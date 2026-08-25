"""mppi_impedance_harness.py — replicate the pipeline's task-space rollout FAITHFULLY on the local box,
in numpy, no MJX compile, no IRL loop. Loads the EXACT pipeline setup (rail, init, reference tables,
gains, cost weights) dumped by mppi_mjx_kinematic.py's TS_DUMP_SETUP hook, so the rollout matches the
mango run's geometry: real rail_dir/rail_len (0.237 m along the demo scrape), real init pose, real
qddot_ff/qdot_ff feedforward, real dt=0.008632, real q_of_s posture table.

Regenerate the dump (fast, setup-only, no compile) with:  TS_DUMP_SETUP=1 <the run command>
Then run this:  RENDER=gif LAW=mppi python experiments/mppi_impedance_harness.py

LAW=mppi (op-space JᵀF, ts_task_pd=1, the real config) | LAW=ct (computed-torque inside-M, comparison)
env: STEPS(=full ref)  NSUB=1  DUMP=data/mppi_setup_dump.npz  KP_LAT/NULL_KP/PEN_KP/... override gains
"""
import os, sys, math
import numpy as np
_RENDER = os.environ.get('RENDER')
if _RENDER:
    os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
os.makedirs('runs', exist_ok=True)
sys.path.insert(0, 'src')
from prune_arm_model import prune

LAW = os.environ.get('LAW', 'mppi')
SANITIZE = os.environ.get('SANITIZE', '1') != '0'
# gains — the nan'd run config (TS_TASK_PD=1 op-space + TS_NULL_KP=50 + coarse dt)
KP_LAT = float(os.environ.get('KP_LAT', 2500.0)); KD_LAT = 60.0
KD_RAILF = 40.0; PEN_KP = float(os.environ.get('PEN_KP', '13800')); PEN_KD = float(os.environ.get('PEN_KD', '300'))
PEN_DES = float(os.environ.get('PEN_DES', 0.005)); PEN_MAX = float(os.environ.get('PEN_MAX', 0.02)); FNMAX = float(os.environ.get('FNMAX', 200.0))
ENGAGE_BAND = float(os.environ.get('ENGAGE_BAND', 0.006))
MODEL_B = os.environ.get('MODEL_B', '1') != '0'
FORCE_FROM_PEN = os.environ.get('FORCE_FROM_PEN', '0') != '0'
TRACK_TAN = os.environ.get('TRACK_TAN', '1' if MODEL_B else '0') != '0'
QOFS_ON_SURF = os.environ.get('QOFS_ON_SURF', '1' if MODEL_B else '0') != '0'
QOFS_SURF_GAP = float(os.environ.get('QOFS_SURF_GAP', 0.0))
PRESS_FROM_SENSOR = os.environ.get('PRESS_FROM_SENSOR', '1' if MODEL_B else '0') != '0'
SOFT_LAND = os.environ.get('SOFT_LAND', '1' if MODEL_B else '0') != '0'
SOFT_STEPS = float(os.environ.get('SOFT_STEPS', 8))
SOFT_TOUCH_GAP = float(os.environ.get('SOFT_TOUCH_GAP', 0.001))
STEADY_START = os.environ.get('STEADY_START', '0') != '0'
DISCOVER_LAND = (os.environ.get('DISCOVER_LAND', '0') != '0') and not STEADY_START
LIFT_H = float(os.environ.get('LIFT_H', 0.015))
LAND_STEPS = float(os.environ.get('LAND_STEPS', 8))
VN_FILT = float(os.environ.get('VN_FILT', 0.6 if MODEL_B else 0.0))         # EMA on the normal velocity feeding the damper (0=raw, →1=heavily smoothed) → breaks the stiff-contact chatter the D-term otherwise amplifies
PRESS_IN_CONTACT = os.environ.get('PRESS_IN_CONTACT', '1' if MODEL_B else '0') != '0'   # apply the impedance press ONLY when actually in contact (gap<0), never above the surface. The old above-surface engage applied an unphysical INWARD PULL ("suction") that yanked the tool at the stiff wall whenever the slide lifted it → the sliding chatter. A contact can't pull.
SEAT_F = float(os.environ.get('SEAT_F', 12.0 if MODEL_B else 0.0))          # AFTER first contact only: gentle constant inward force (N) in a band above the surface → catches any bounce so the tool never jumps back out (gated on the contact latch → no pre-contact yank)
SEAT_BAND = float(os.environ.get('SEAT_BAND', 0.012))
SEAT_DEPTH = float(os.environ.get('SEAT_DEPTH', 0.0005)) # AFTER first contact: commit a bit deeper (add this to the target depth) → margin so a landing bounce can't break contact for the rest of the motion
CCANCEL = 1.0; FF_FRAC = 0.5; TS_KD = 40.0; TS_KP = 100.0
IMPEDANCE_SUBSTEP = os.environ.get('IMPEDANCE_SUBSTEP', '1') != '0'
TERM_DBG = os.environ.get('TERM_DBG', '0') != '0'
TERM_DBG_K = int(os.environ.get('TERM_DBG_K', 33))
NULL_KP = float(os.environ.get('NULL_KP', 50.0))
NULL_RUNTIME = os.environ.get('NULL_RUNTIME', '1') != '0'
FORCE_GAIN = float(os.environ.get('FORCE_GAIN', 0.25))
PRESS_FN = float(os.environ.get('PRESS_FN', 26.0))

DUMP = os.environ.get('DUMP', 'data/mppi_setup_dump.npz')
if not os.path.exists(DUMP):
    sys.exit(f"[harness] {DUMP} not found — regenerate with:  TS_DUMP_SETUP=1 <the run command>")
D = np.load(DUMP, allow_pickle=True)
RAIL_DIR = D['rail_dir'].astype(float); P_START = D['p_start'].astype(float); RAIL_LEN = float(D['rail_len'])
INIT_QPOS = D['init_qpos'].astype(float)
QDDREF = D['qddot_ff'].astype(float); QDREF = D['qdot_ff'].astype(float)
QOFS = D['ts_q_of_s'].astype(float)
STICK_HALF = float(D['stick_half_len']); STICK_R = float(D['stick_radius'])
STICK_TRAJ = D['stick_traj'].astype(float) if D['stick_traj'].size else None
DT = float(D['dt']); NREF = len(QDREF)
STEPS = int(os.environ.get('STEPS', NREF)); NSUB = int(os.environ.get('NSUB', '8'))

# reach" red herring). SUBJECT must match whatever generated mppi_setup_dump.npz.
SUBJECT = os.environ.get('SUBJECT', 's2')
MOCAP_DATE = os.environ.get('MOCAP_DATE', '27_02')
PR = f'config/xml_models/{MOCAP_DATE}/{SUBJECT}_pruned.xml'
if not os.path.exists(PR):
    sys.exit(f"[harness] {PR} not found — run the pipeline once (it writes {SUBJECT}_pruned.xml), or set SUBJECT=")
m = mujoco.MjModel.from_xml_path(PR); d = mujoco.MjData(m)
m.opt.timestep = DT
d.qpos[:len(INIT_QPOS)] = INIT_QPOS
mujoco.mj_forward(m, d)

ARM = np.arange(9)
rock_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'rock')
rock_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'rock_sphere')
if rock_gid < 0:
    rock_gid = list(range(m.body_geomadr[rock_bid], m.body_geomadr[rock_bid] + m.body_geomnum[rock_bid]))[-1]
ROCK_R = float(m.geom_size[rock_gid][0])
stick_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'stick')
stick_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'stick_freejoint')
stick_qadr = m.jnt_qposadr[stick_jid]; stick_vadr = m.jnt_dofadr[stick_jid]
STICK_PIN = d.qpos[stick_qadr:stick_qadr + 7].copy()
STICK_QUAT = STICK_PIN[3:7].copy()
SENSOR_ONSET = 0
STICK_FIXED = os.environ.get('STICK_FIXED', '0') != '0'
def _stick_pos(k):
    if STICK_TRAJ is None:
        return STICK_PIN[:3]
    if STICK_FIXED:
        return STICK_TRAJ[0]
    kk = 0 if k < SENSOR_ONSET else min(k, len(STICK_TRAJ) - 1)
    return STICK_TRAJ[kk]
u_max = m.actuator_ctrlrange[:, 1].copy()
if os.environ.get('THORAX_UMAX'):
    u_max[0] = float(os.environ['THORAX_UMAX'])
_M = np.zeros((m.nv, m.nv))
_stick_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'stick_geom')
_TRACE = {'u': [], 'fcmd': [], 'fact': [], 'gap': [], 'q': [], 'v': []}
def _contact_normal():
    for ci in range(d.ncon):
        g1, g2 = d.contact[ci].geom1, d.contact[ci].geom2
        if {g1, g2} == {rock_gid, _stick_gid}:
            f6 = np.zeros(6); mujoco.mj_contactForce(m, d, ci, f6); return abs(float(f6[0]))
    return 0.0
K_SERIES = PEN_KP
PRESS_GAIN = float(os.environ.get('PRESS_GAIN', '0.55'))
PRESS_FLOOR = float(os.environ.get('PRESS_FLOOR', '0.003'))
PRESS_KCAL = float(os.environ.get('PRESS_KCAL', 1.0))
PEN_CAP = os.environ.get('PEN_CAP', '0') != '0'
PEN_CAP_MARGIN = float(os.environ.get('PEN_CAP_MARGIN', 0.0008))
PEN_CAP_KP = float(os.environ.get('PEN_CAP_KP', 80000.0))          # barrier stiffness (N/m) — must beat the tracking's inward drive
PEN_CAP_KD = float(os.environ.get('PEN_CAP_KD', 500.0))
if MODEL_B:
    # near-RIGID rock-stick contact: the arm must PUSH (torque → reaction) instead of penetrating, so the force
    NSUB = int(os.environ.get('NSUB', '8')); m.opt.timestep = DT / NSUB
    _srfloor = float(os.environ.get('SOLREF', 0.002))
    _sr = max(2.2 * m.opt.timestep, _srfloor)
    _sd = float(os.environ.get('SOLREF_D', 1.0))
    _simp = np.array([0.9, 0.99, 0.001, 0.5, 2.0])
    _setpair = False
    for _pi in range(m.npair):
        if {int(m.pair_geom1[_pi]), int(m.pair_geom2[_pi])} == {rock_gid, _stick_gid}:
            m.pair_solref[_pi] = np.array([_sr, _sd]); m.pair_solimp[_pi] = _simp; _setpair = True
    if not _setpair:
        for _g in (rock_gid, _stick_gid):
            m.geom_solref[_g] = np.array([_sr, _sd]); m.geom_solimp[_g] = _simp
    print(f"[model-B] wall solref={_sr:.4f} {'(PAIR)' if _setpair else '(geom)'}, NSUB={NSUB}, sim dt={m.opt.timestep*1000:.2f}ms")
    _qs = d.qpos.copy(); _pt = 0.003
    _sp0 = (STICK_TRAJ[0] if STICK_TRAJ is not None else STICK_PIN[:3]).copy()
    d.qpos[stick_qadr:stick_qadr + 3] = _sp0; mujoco.mj_forward(m, d)
    _rp = d.geom_xpos[rock_gid]; _sc = d.xpos[stick_bid]; _sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
    _tp = float(np.clip((_rp - _sc) @ _sa, -STICK_HALF, STICK_HALF)); _nv = _rp - (_sc + _tp * _sa); _nn = _nv / max(np.linalg.norm(_nv), 1e-9)
    _gap0 = float(np.linalg.norm(_nv) - STICK_R - ROCK_R)
    d.qpos[stick_qadr:stick_qadr + 3] = _sp0 + _nn * (_gap0 + _pt); mujoco.mj_forward(m, d)
    _fr = _contact_normal(); KC = _fr / _pt if _fr > 1e-6 else 1e9
    K_SERIES = PEN_KP * KC / (PEN_KP + KC)
    _kser_static = K_SERIES; K_SERIES = K_SERIES * PRESS_KCAL
    d.qpos[:] = _qs; mujoco.mj_forward(m, d)
    print(f"[model-B] measured contact kc={KC:.0f} N/m → K_series={_kser_static:.0f} N/m ×KCAL {PRESS_KCAL:g} = {K_SERIES:.0f} N/m  (target depth = F_sensor/K_series; pen@69N ≈ {69/K_SERIES*1000:.1f}mm)"
          + (f"  | PEN_CAP ON: ≤ target+{PEN_CAP_MARGIN*1000:.1f}mm (kp={PEN_CAP_KP:.0f})" if PEN_CAP else ""))
_rp0 = d.geom_xpos[rock_gid]; s0 = float(np.dot(_rp0 - P_START, RAIL_DIR))
print(f"[harness] LAW={LAW} dt={DT:.5f} steps={STEPS}  rail_len={RAIL_LEN*100:.1f}cm  "
      f"init tool s={s0*100:.1f}cm  kp_lat={KP_LAT} null_kp={NULL_KP}")

NULLOFS = D['ts_null_of_s'].astype(float) if D['ts_null_of_s'].ndim == 3 else None
NNULL = NULLOFS.shape[1] if NULLOFS is not None else 0
_offs = {'dnull': np.zeros(max(NNULL, 1)), 'dpress': np.zeros(1), 'dpace': 0.0, 'dlift': 0.0, 'ds0': 0.0, 'dpress_now': 0.0}

HOLD_THORAX = os.environ.get('HOLD_THORAX', '1') != '0'
THORAX_KINEMATIC = os.environ.get('THORAX_KINEMATIC', '0') != '0'
QOFS_ON_CONTACT = os.environ.get('QOFS_ON_CONTACT', '1') != '0'
LAND_COST_W = float(os.environ.get('LAND_COST_W', '100'))
QOFS_PRE_KP = float(os.environ.get('QOFS_PRE_KP', '20'))
QOFS_ENGAGE_GAP = float(os.environ.get('QOFS_ENGAGE_GAP', 0.006))
THORAX_REF = float(os.environ.get('THORAX_VAL', INIT_QPOS[0]))
THORAX_KP = float(os.environ.get('THORAX_KP', 800.0))
if HOLD_THORAX and NULLOFS is not None:
    NULLOFS = NULLOFS.copy(); NULLOFS[:, :, 0] = 0.0
DEMO_THORAX = S_NOM = None
DEMO_NPZ = os.environ.get('DEMO_NPZ', 'data/harness_demo.npz')
if HOLD_THORAX and os.path.exists(DEMO_NPZ):
    _dz = np.load(DEMO_NPZ, allow_pickle=True)
    _sn = D['ts_s_nom'].astype(float)
    if 'demo_xs' in _dz.files and len(_dz['demo_xs']) == len(_sn):
        _dth = _dz['demo_xs'][:, 0].astype(float)
        _o = np.argsort(_sn); S_NOM = _sn[_o]; DEMO_THORAX = _dth[_o]
if HOLD_THORAX:
    _mode = (f"TRACKING demo trajectory ({math.degrees(DEMO_THORAX.min()):.1f}..{math.degrees(DEMO_THORAX.max()):.1f}deg over s)"
             if DEMO_THORAX is not None else f"held CONSTANT at {math.degrees(THORAX_REF):.1f}deg")
    print(f"[hold] thoracic (DOF0) {_mode}; rest tracks the NEUTRAL q_of_s")
DEMO_XS = np.load(DEMO_NPZ, allow_pickle=True)['demo_xs'].astype(float) if os.path.exists(DEMO_NPZ) else None
# is the "reliable demo" — NOT a fresh mj_inverse (which blows up to ~40kN·m on the noisy finite-diff accelerations).
DEMO_US = np.load(DEMO_NPZ, allow_pickle=True)['demo_us'].astype(float) if (os.path.exists(DEMO_NPZ) and 'demo_us' in np.load(DEMO_NPZ, allow_pickle=True).files) else None

DEMO_XS_POOL = DEMO_US_POOL = None
_dpool = os.environ.get('DEMO_POOL', '')
if _dpool and os.path.exists(_dpool):
    _pz = np.load(_dpool, allow_pickle=True)
    if 'demo_xs_pool' in _pz.files:
        DEMO_XS_POOL = np.asarray(_pz['demo_xs_pool']).astype(float)
        DEMO_US_POOL = np.asarray(_pz['demo_us_pool']).astype(float) if 'demo_us_pool' in _pz.files else None
        print(f"[demo] POOL loaded from {_dpool}: {len(DEMO_XS_POOL)} demos → φ_demo will be the pool mean")

DEMO_SENSOR_F = None
_scsv = os.environ.get('SENSOR_CSV', f'trajectories_from_mocap/27_02_sensor/{SUBJECT.capitalize()}/down_long/elaborated/force_slices/down/cycle_05.csv')
_savg = os.environ.get('SENSOR_AVG', '1') != '0'
if os.environ.get('SENSOR_FORCE', '1') != '0' and DEMO_XS is not None:
    def _load_fmag(_p):
        _sd = np.genfromtxt(_p, delimiter=',', names=True, usecols=('fx', 'fy', 'fz'))
        _m = np.atleast_1d(np.sqrt(_sd['fx'] ** 2 + _sd['fy'] ** 2 + _sd['fz'] ** 2))
        if _m.size < 2:
            return None
        return np.interp(np.linspace(0, 1, len(DEMO_XS)), np.linspace(0, 1, len(_m)), _m)
    if _savg:
        import glob
        _dir = os.path.dirname(_scsv); _files = sorted(glob.glob(os.path.join(_dir, 'cycle_*.csv')))
        _mags = [_mg for _f in _files if (_mg := _load_fmag(_f)) is not None]
        if _mags:
            DEMO_SENSOR_F = np.mean(_mags, axis=0)
            print(f"[sensor] AVERAGED |F| over {len(_mags)}/{len(_files)} cycles in {_dir}")
    elif os.path.exists(_scsv):
        DEMO_SENSOR_F = _load_fmag(_scsv)
    if DEMO_SENSOR_F is not None:
        _ssm = int(os.environ.get('SENSOR_SMOOTH', '15'))
        if _ssm >= 3:
            from scipy.signal import savgol_filter
            _w = _ssm + (1 - _ssm % 2); _w = min(_w, len(DEMO_SENSOR_F) - (1 - len(DEMO_SENSOR_F) % 2))
            if _w >= 3:
                DEMO_SENSOR_F = np.clip(savgol_filter(DEMO_SENSOR_F, _w, 2), 0.0, None)
        _fscale = float(os.environ.get('SENSOR_SCALE', 1.0))               # scale the demo press force (both demo φ + challenger base) — a LIGHT presser's shallow base lets the sampled depth dip below the surface (negative → floats/blows up); scaling up deepens the press so sampling stays positive
        if _fscale != 1.0:
            DEMO_SENSOR_F = DEMO_SENSOR_F * _fscale
            print(f"[sensor] SENSOR_SCALE={_fscale} → press force ×{_fscale}")
        print(f"[sensor] demo press_force |F|: {DEMO_SENSOR_F.min():.1f}..{DEMO_SENSOR_F.max():.1f}N "
              f"(mean {DEMO_SENSOR_F.mean():.1f}, {'AVERAGED' if _savg else 'single cyc'}, smooth w={_ssm})")
    _F_ON = float(os.environ.get('SENSOR_CONTACT_N', '5.0'))
    _hit = np.where(DEMO_SENSOR_F > _F_ON)[0]
    SENSOR_ONSET = int(_hit[0]) if len(_hit) else 0
    _omask = int(os.environ.get('ONSET_MASK', 0))
    if _omask > 0:
        SENSOR_ONSET = max(SENSOR_ONSET, _omask)
        DEMO_SENSOR_F[:SENSOR_ONSET] = 0.0
        print(f"[sensor] ONSET_MASK={_omask}: premature press masked → onset frame {SENSOR_ONSET}")
    print(f"[sensor] contact onset (>{_F_ON:.0f}N): frame {SENSOR_ONSET}  (mocap-gap contact was frame 32) -> stick fixed until then")

# tens of mm "above" a stick that never followed it (S1 floats +60mm). The rock CANNOT be above the stick when the
if os.environ.get('STICK_ON_ROCK') and STICK_TRAJ is not None and DEMO_XS is not None:
    _seat = float(os.environ.get('STICK_ON_ROCK_SEAT', 0.0))
    _st = STICK_TRAJ.copy(); _ns = 0; _gmax = 0.0
    for _t in range(min(len(STICK_TRAJ), len(DEMO_XS))):
        _inc = (DEMO_SENSOR_F[min(_t, len(DEMO_SENSOR_F) - 1)] > _F_ON) if DEMO_SENSOR_F is not None else (_t >= SENSOR_ONSET)
        if not _inc:
            continue
        d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = DEMO_XS[_t, 0:9]
        d.qpos[stick_qadr:stick_qadr + 3] = STICK_TRAJ[_t]; d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
        mujoco.mj_forward(m, d)
        _rp = d.geom_xpos[rock_gid]; _sc = d.xpos[stick_bid]; _sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
        _tp = np.clip((_rp - _sc) @ _sa, -STICK_HALF, STICK_HALF); _nv = _rp - (_sc + _tp * _sa); _dn = np.linalg.norm(_nv)
        _n = _nv / max(_dn, 1e-9); _gap = _dn - STICK_R - ROCK_R; _gmax = max(_gmax, _gap)
        _st[_t] = STICK_TRAJ[_t] + (_gap + _seat) * _n
        _ns += 1
    STICK_TRAJ = _st
    print(f"[stick] STICK_ON_ROCK: re-seated {_ns} contact frames onto the demo rock (was up to +{_gmax*1000:.0f}mm off → gap {-_seat*1000:.0f}mm)")

if os.environ.get('QOFS_HOLD_TRUNK', '1') and DEMO_THORAX is not None:
    _sgrid = D['ts_s_grid'].astype(float) if 'ts_s_grid' in D.files and D['ts_s_grid'].size else np.linspace(0, RAIL_LEN, len(QOFS))
    _jkp = np.zeros((3, m.nv))
    def _ik_fix_trunk(q0, th0, ptgt, iters=60, damp=2e-3, knull=0.05):
        q = q0.copy().astype(float); q[0] = th0
        for _ in range(iters):
            d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = q; mujoco.mj_forward(m, d)
            e = ptgt - d.geom_xpos[rock_gid]
            if np.linalg.norm(e) < 1e-5:
                break
            mujoco.mj_jac(m, d, _jkp, None, d.geom_xpos[rock_gid], rock_bid)
            J = _jkp[:, 0:9].copy(); J[:, 0] = 0.0
            Hs = J @ J.T + damp * np.eye(3); Jpi = J.T @ np.linalg.inv(Hs)
            dq = Jpi @ e + (np.eye(9) - Jpi @ J) @ (knull * (q0 - q))
            dq[0] = 0.0; q = q + dq; q[0] = th0                    # never move the thoracic
        return q
    _QDEMO = bool(os.environ.get('QOFS_DEMO_PRESS', '1')) and DEMO_XS is not None
    if _QDEMO:
        _dtp = np.zeros((len(DEMO_XS), 3)); _drs = np.zeros(len(DEMO_XS))
        for _t in range(len(DEMO_XS)):
            d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = DEMO_XS[_t, 0:9]; mujoco.mj_forward(m, d)
            _dtp[_t] = d.geom_xpos[rock_gid].copy(); _drs[_t] = float((_dtp[_t] - P_START) @ RAIL_DIR)
        _oo = np.argsort(_drs); _drs_s = _drs[_oo]; _dtp_s = _dtp[_oo]
        print(f"[qofs] RETARGET to DEMO tool path (press incl.), demo rail-s {_drs_s[0]*100:.1f}..{_drs_s[-1]*100:.1f}cm")
    def _stick_at_s(_sv):
        if STICK_TRAJ is None:
            return None
        if STICK_FIXED or S_NOM is None:
            return STICK_TRAJ[0]
        _f = int(np.clip(np.argmin(np.abs(np.asarray(S_NOM) - _sv)), 0, len(STICK_TRAJ) - 1))
        _f = 0 if _f < SENSOR_ONSET else _f
        return STICK_TRAJ[min(_f, len(STICK_TRAJ) - 1)]
    _Q2 = QOFS.copy(); _errs = []
    for _i in range(len(QOFS)):
        d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = QOFS[_i]; mujoco.mj_forward(m, d)
        if _QDEMO:
            _ptgt = np.array([np.interp(_sgrid[_i], _drs_s, _dtp_s[:, _c]) for _c in range(3)])
        else:
            _ptgt = d.geom_xpos[rock_gid].copy()
        if QOFS_ON_SURF:
            if STICK_TRAJ is not None:
                d.qpos[stick_qadr:stick_qadr + 3] = _stick_at_s(_sgrid[_i]); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT; mujoco.mj_forward(m, d)
            _scc = d.xpos[stick_bid].copy(); _saa = d.xmat[stick_bid].reshape(3, 3)[:, 2].copy()
            _tpp = float(np.clip((_ptgt - _scc) @ _saa, -STICK_HALF, STICK_HALF)); _axp = _scc + _tpp * _saa
            _dvv = _ptgt - _axp; _nnn = _dvv / max(np.linalg.norm(_dvv), 1e-9)
            _ptgt = _axp + (STICK_R + ROCK_R + QOFS_SURF_GAP) * _nnn
        _th0 = float(np.interp(_sgrid[_i], S_NOM, DEMO_THORAX)) if S_NOM is not None else float(np.mean(DEMO_THORAX))
        _seed = INIT_QPOS[0:9].astype(float).copy() if _QDEMO else QOFS[_i]
        _Q2[_i] = _ik_fix_trunk(_seed, _th0, _ptgt)
        d.qpos[0:9] = _Q2[_i]; mujoco.mj_forward(m, d); _errs.append(float(np.linalg.norm(_ptgt - d.geom_xpos[rock_gid])))
    print(f"[qofs] re-solved with TRUNK HELD: thoracic {math.degrees(_Q2[:,0].min()):.1f}..{math.degrees(_Q2[:,0].max()):.1f}deg, "
          f"ELBOW {math.degrees(_Q2[:,6].min()):.1f}..{math.degrees(_Q2[:,6].max()):.1f}deg (demo branch ~35-52° | old neutral ~78°); tool-pos err mean={np.mean(_errs)*1000:.2f}mm")
    # DIAGNOSTIC: does the reference itself PENETRATE? (min gap < 0 ⇒ q_of_s presses; the challenger must track into this)
    _qg = []
    for _i in range(len(_Q2)):
        d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = _Q2[_i]
        if STICK_TRAJ is not None:
            d.qpos[stick_qadr:stick_qadr + 3] = _stick_at_s(_sgrid[_i]); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
        mujoco.mj_forward(m, d)
        _rp = d.geom_xpos[rock_gid]; _sc = d.xpos[stick_bid]; _sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
        _tp = np.clip((_rp - _sc) @ _sa, -STICK_HALF, STICK_HALF); _qg.append((np.linalg.norm(_rp - (_sc + _tp * _sa)) - STICK_R - ROCK_R) * 1000)
    _qg = np.array(_qg)
    print(f"[qofs] reference GAP over arc-length: {_qg.min():.1f}..{_qg.max():.1f}mm  (min<0 ⇒ q_of_s PENETRATES → tracking must follow it in)")
    QOFS = _Q2

DISCOVER = os.environ.get('DISCOVER', '0') != '0'
DISCOVER_LIFT = os.environ.get('DISCOVER_LIFT', '1') != '0'
LIFT_OF_S = D['ts_lift_of_s'].astype(float) if 'ts_lift_of_s' in D.files and D['ts_lift_of_s'].size else None
if LIFT_OF_S is not None and HOLD_THORAX:
    LIFT_OF_S = LIFT_OF_S.copy(); LIFT_OF_S[:, 0] = 0.0
N_APP = float(os.environ.get('TS_N_APP', 8.0))
LIFT_SCALE = float(os.environ.get('TS_LIFT_SCALE', 0.01)); LIFT_MAX = float(os.environ.get('TS_LIFT_MAX', 0.04))
LIFT_SIG = float(os.environ.get('LIFT_SIG', 3.0))
NDISC = 1 if (DISCOVER and LIFT_OF_S is not None) else 0
S0_SCALE = float(os.environ.get('TS_S0_SCALE', 0.03)); S0_MAX = float(os.environ.get('TS_S0_MAX', 0.12))
S0_SIG = float(os.environ.get('S0_SIG', 2.0))
TS_S_NOM = D['ts_s_nom'].astype(float) if 'ts_s_nom' in D.files and np.asarray(D['ts_s_nom']).size else None
PRESS_GATE = os.environ.get('PRESS_GATE', '0') != '0'
if DISCOVER:
    print(f"[discover] contact-discovery ON: sampled approach-lift (N_app={N_APP:.0f} steps, scale={LIFT_SCALE}, "
          f"max={LIFT_MAX*1000:.0f}mm, σ={LIFT_SIG}) — press gated off during the descent")

FEATURES = os.environ.get('FEATURES', '1') != '0'
_KEYS = ['Tau', 'JV', 'JA', 'JTC', 'Eng_thoracic', 'Eng_clavicle', 'Eng_shoulder', 'Eng_elbow', 'Eng_wrist',
         'Geo', 'press_force', 'surface', 'progress_vel', 'rail_lat', 'rock_ori', 'approach', 'traveled',
         'Tau_thoracic', 'Tau_clavicle', 'Tau_shoulder', 'Tau_elbow', 'Tau_wrist', 'press_capacity', 'brace',
         'JV_thoracic', 'JV_clavicle', 'JV_shoulder', 'JV_elbow', 'JV_wrist', 'ElbowTrack']
_wd = dict(zip([str(k) for k in D['w_run_keys']], D['w_run_vals'].astype(float)))
W_VEC = np.array([_wd.get(k, 0.0) for k in _KEYS])
if os.environ.get('WEIGHTS'):
    for kv in os.environ['WEIGHTS'].split(','):
        _k, _v = kv.split('='); W_VEC[_KEYS.index(_k.strip())] = float(_v)
FEAT_DROP = os.environ.get('FEAT_DROP', 'JTC,JV_thoracic,Eng_thoracic,Tau_thoracic,ElbowTrack')   # JTC is a hardcoded 0 placeholder (never computed) → dead like brace
DROP_IDX = [_KEYS.index(_k.strip()) for _k in FEAT_DROP.split(',') if _k.strip() in _KEYS] if FEAT_DROP else []
if DROP_IDX:
    W_VEC[DROP_IDX] = 0.0
    print(f"[feat] dropped (held-thoracic → uninformative): {[_KEYS[i] for i in DROP_IDX]}")
N_W = int(os.environ.get('N_W', 12))
WIN_SIZE = max(1, STEPS // N_W)
WIN_OF_K = np.minimum(np.arange(STEPS + 1) // WIN_SIZE, N_W - 1)
PRESS_TV = os.environ.get('PRESS_TV', '1') != '0'
NP = N_W if PRESS_TV else 1
W_TV = np.tile(W_VEC, (N_W, 1)).astype(float)
if os.environ.get('WEIGHTS_NPZ') and os.path.exists(os.environ['WEIGHTS_NPZ']):
    _wz = np.load(os.environ['WEIGHTS_NPZ'])
    W_TV = np.asarray(_wz['w_tv'], float).reshape(N_W, 30)
    print(f"[weights] deploying recovered W_TV from {os.environ['WEIGHTS_NPZ']} (mean top: " +
          " ".join(f"{_KEYS[i]}={W_TV.mean(0)[i]:.1f}" for i in np.argsort(-W_TV.mean(0))[:4]) + ")")
BASIS = os.environ.get('BASIS', '0') != '0'
K_BASIS = int(os.environ.get('K_BASIS', N_W))
_T = STEPS; _centers = np.linspace(0, _T, K_BASIS)
_sig_b = max(_T / max(K_BASIS - 1, 1), 1.0)
_ts = np.arange(_T + 1)
_Braw = np.exp(-((_ts[:, None] - _centers[None, :]) ** 2) / (2.0 * _sig_b ** 2))
B_STEP = _Braw / (_Braw.sum(axis=1, keepdims=True) + 1e-12)
B_WINDOW = np.zeros((N_W, K_BASIS))
for _k in range(N_W):
    _lo = _k * WIN_SIZE; _hi = (_k + 1) * WIN_SIZE if _k < N_W - 1 else _T + 1
    B_WINDOW[_k] = B_STEP[_lo:_hi].mean(axis=0)
THETA = np.tile(W_VEC, (K_BASIS, 1)).astype(float)
if BASIS:
    W_TV = (B_WINDOW @ THETA).astype(float)
ENG_G = [[0], [1], [2, 3, 4], [5, 6], [7, 8]]
VT_KP = float(os.environ.get('VT_KP', PEN_KP))
FORCE_MAX = float(D['force_max']); FORCE_EMA = float(D['force_ema'])
VEL_EMA = float(D['vel_feat_ema']); TGT_F = float(D['target_force']); ROCK_QREF = D['rock_quat_ref'].astype(float)
ELB_QADR = D['elbow_qposadr'].astype(int); ELB_REF = D['elbow_ref'].astype(float)
SITE_ID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'rock_contact_point')
ROCK_BID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'rock')
_Mg = np.zeros((m.nv, m.nv))
def compute_phi(u, prev_v, prev_f, f_override=None, k=None):
    v = VEL_EMA * np.asarray(d.qvel)[ARM] + (1 - VEL_EMA) * prev_v
    ddq = (v - prev_v) / DT
    rp = d.geom_xpos[rock_gid]; sc = d.xpos[stick_bid]; sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
    tp = np.clip((rp - sc) @ sa, -STICK_HALF, STICK_HALF); gap = np.linalg.norm(rp - (sc + tp * sa)) - STICK_R - ROCK_R
    fc = float(f_override) if f_override is not None else VT_KP * max(0.0, -gap)
    f_ema = FORCE_EMA * fc + (1 - FORCE_EMA) * prev_f
    mujoco.mj_fullM(m, _Mg, d.qM); vq = np.asarray(d.qvel); geo = np.sqrt(max(0.0, vq @ _Mg @ vq))
    sp = d.site_xpos[SITE_ID] if SITE_ID >= 0 else rp; trav = (sp - P_START) @ RAIL_DIR
    diff = sp - P_START; lat = diff - (diff @ RAIL_DIR) * RAIL_DIR
    qc = d.xquat[ROCK_BID]; pw = v * u
    phi = np.array([
        np.linalg.norm(u) / 50, np.linalg.norm(v), np.linalg.norm(ddq) / 100, 0.0,
        *[np.linalg.norm(pw[g]) / 20 for g in ENG_G], geo,
        abs(f_ema - TGT_F) / max(TGT_F, 1.0), max(0.0, gap) / STICK_R, 0.0, np.linalg.norm(lat) / STICK_R,
        1 - (qc @ ROCK_QREF) ** 2, (max(0.0, -gap) / STICK_R if (k is not None and k < SENSOR_ONSET) else 0.0), -trav / max(RAIL_LEN, 1e-9),   # approach = PREMATURE contact: penetration BEFORE the SENSOR onset (frame 10), 0 after. Demo never trips it (hovers till sensor contact); challenger diving early does → early-window weight penalizes it. NOT keyed to the frame-32 mocap artifact
        *[np.linalg.norm(u[g]) / 50 for g in ENG_G],
        (np.clip(f_ema, -2 * FORCE_MAX, 2 * FORCE_MAX) - FORCE_MAX) ** 2 / FORCE_MAX ** 2, 0.0,
        *[np.linalg.norm(v[g]) for g in ENG_G], np.sum((np.asarray(d.qpos)[ELB_QADR] - ELB_REF) ** 2)])
    if DROP_IDX:
        phi[DROP_IDX] = 0.0
    return phi, v, f_ema

def _demo_phi_one(dq, dv, dus):
    """Windowed feature sum (N_W,30) for ONE demo joint trajectory (dq, dv) with torque dus (or mj_inverse)."""
    dacc = np.gradient(dv, DT, axis=0)
    prev_v = np.zeros(9); prev_f = 0.0; acc = np.zeros((N_W, 30)); N = len(dq)
    _offs['dpress_now'] = 0.0
    _fallback = dus is None
    for t in range(N):
        d.qpos[0:9] = dq[t]; d.qpos[stick_qadr:stick_qadr + 3] = _stick_pos(t); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
        d.qvel[:] = 0.0; d.qvel[0:9] = dv[t]; d.qacc[:] = 0.0; d.qacc[0:9] = dacc[t]
        if _fallback:
            mujoco.mj_inverse(m, d); tau = np.asarray(d.qfrc_inverse)[0:9]
        else:
            mujoco.mj_forward(m, d)
            tau = dus[min(t, len(dus) - 1)][:9]
        _fs = float(DEMO_SENSOR_F[min(t, len(DEMO_SENSOR_F) - 1)]) if DEMO_SENSOR_F is not None else None
        phi, prev_v, prev_f = compute_phi(tau, prev_v, prev_f, f_override=_fs, k=t)
        acc[WIN_OF_K[min(t, STEPS)]] += phi
    return acc

def compute_demo_phi():
    """The DEMO's mean feature vector φ_demo — the IRL target. Roll the demo trajectory (arm=demo_xs),
    get its torque via inverse dynamics, and run the SAME compute_phi as the challenger.
    DEMO_POOL (an npz with demo_xs_pool/demo_us_pool) → AVERAGE φ over the pool (per-subject-pooled / combined model)."""
    if DEMO_XS is None:
        return None
    if DEMO_XS_POOL is not None:
        accs = [_demo_phi_one(x[:, 0:9], x[:, 10:19],
                              (DEMO_US_POOL[i] if DEMO_US_POOL is not None else None))
                for i, x in enumerate(DEMO_XS_POOL)]
        print(f"[demo] POOLED φ over {len(accs)} demos (combined/per-subject-pooled target)")
        return np.mean(accs, axis=0)
    return _demo_phi_one(DEMO_XS[:, 0:9], DEMO_XS[:, 10:19], DEMO_US)

PRESS_RAMP = os.environ.get('PRESS_RAMP', '0') != '0'
DEMO_PEN_S = DEMO_PEN_VAL = None
if PRESS_RAMP and DEMO_XS is not None:
    _ss = []; _pp = []
    for _t in range(len(DEMO_XS)):
        d.qpos[:len(INIT_QPOS)] = INIT_QPOS; d.qpos[0:9] = DEMO_XS[_t, 0:9]
        d.qpos[stick_qadr:stick_qadr + 3] = _stick_pos(_t); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
        mujoco.mj_forward(m, d)
        rp = d.geom_xpos[rock_gid]; sc = d.xpos[stick_bid]; sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
        tp = np.clip((rp - sc) @ sa, -STICK_HALF, STICK_HALF); g = np.linalg.norm(rp - (sc + tp * sa)) - STICK_R - ROCK_R
        _ss.append(float((rp - P_START) @ RAIL_DIR)); _pp.append(max(0.0, -g))
    _o = np.argsort(_ss); DEMO_PEN_S = np.array(_ss)[_o]; DEMO_PEN_VAL = np.clip(np.array(_pp)[_o], 0.0, PEN_MAX)
    print(f"[press-ramp] demo penetration schedule vs s: 0 → {DEMO_PEN_VAL.max()*1000:.0f}mm (gentle early, deep late)")

_NO_FN = bool(os.environ.get('NO_FN'))
_NO_FF = bool(os.environ.get('NO_FF'))
def control(k):
    kk = min(k, NREF - 1)
    qd_ref = QDREF[kk] * (1.0 + _offs['dpace'])
    qdd_ref = QDDREF[kk]
    mujoco.mj_fullM(m, _M, d.qM); Marm = _M[np.ix_(ARM, ARM)]
    ff = np.zeros(9) if _NO_FF else np.clip(Marm @ qdd_ref, -FF_FRAC * u_max, FF_FRAC * u_max)
    c = np.asarray(d.qfrc_constraint)[ARM]
    if SANITIZE:
        c = np.where(np.isnan(c), 0.0, c)
    dq = np.asarray(d.qvel)[ARM]
    rp = d.geom_xpos[rock_gid]; sc = d.xpos[stick_bid]; sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
    tp = np.clip(np.dot(rp - sc, sa), -STICK_HALF, STICK_HALF)
    dvec = rp - (sc + tp * sa); dn = max(np.linalg.norm(dvec), 1e-9); n_geo = dvec / dn
    gap = dn - STICK_R - ROCK_R
    s = float(np.dot(rp - P_START, RAIL_DIR))
    jacp = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, jacp, None, rp, rock_bid); Ja = jacp[:, ARM]
    _s0 = float(np.clip(_offs['ds0'] * S0_SCALE, -S0_MAX, S0_MAX))
    _s_ref = (float(TS_S_NOM[min(k, len(TS_S_NOM) - 1)]) + _s0) if (DISCOVER and TS_S_NOM is not None) else s
    gi = int(np.clip(_s_ref / max(RAIL_LEN, 1e-6) * (len(QOFS) - 1), 0, len(QOFS) - 1))
    q_ref_p = QOFS[gi].copy()
    if HOLD_THORAX:
        q_ref_p[0] = float(np.interp(s, S_NOM, DEMO_THORAX)) if DEMO_THORAX is not None else THORAX_REF
    if NULLOFS is not None:
        _dq = _offs['dnull'] @ NULLOFS[gi]
        if NULL_RUNTIME:
            _Jf = Ja.copy(); _Jf[:, 0] = 0.0
            _dq = _dq - _Jf.T @ np.linalg.solve(_Jf @ _Jf.T + 1e-6 * np.eye(3), _Jf @ _dq); _dq[0] = 0.0
        q_ref_p = q_ref_p + _dq
    _pgate = 1.0
    if DISCOVER and DISCOVER_LIFT and LIFT_OF_S is not None:
        _appf = max(0.0, 1.0 - k / max(N_APP, 1.0))
        _lift0 = min(max(_offs['dlift'] * LIFT_SCALE, 0.0), LIFT_MAX)
        q_ref_p = q_ref_p + (_lift0 * _appf) * LIFT_OF_S[gi]
        if PRESS_GATE:
            _pgate = 1.0 - _appf
    Pt = np.eye(3) - np.outer(n_geo, n_geo)
    railt = Pt @ RAIL_DIR; railt = railt / max(np.linalg.norm(railt), 1e-9); railt = railt * np.sign(np.dot(railt, RAIL_DIR) + 1e-9)
    latd = np.cross(n_geo, railt); latd = latd / max(np.linalg.norm(latd), 1e-9)
    vtool = jacp @ np.asarray(d.qvel); vref = Ja @ qd_ref
    frail = float(np.clip(KD_RAILF * (np.dot(vref, railt) - np.dot(vtool, railt)), -60, 60)); Frail = frail * railt
    latoff = float(np.dot(rp - P_START, latd)); Flat = (KP_LAT * (-latoff) - KD_LAT * np.dot(vtool, latd)) * latd
    _win = int(WIN_OF_K[min(k, STEPS)]); _dp = _offs['dpress']
    _offs['dpress_now'] = float(_dp[min(_win, len(_dp) - 1)])
    if PRESS_FROM_SENSOR and DEMO_SENSOR_F is not None:
        _pen_base = float(np.clip(PRESS_GAIN * DEMO_SENSOR_F[min(k, len(DEMO_SENSOR_F) - 1)] / K_SERIES, 0.0, PEN_MAX))
        if k >= SENSOR_ONSET and PRESS_FLOOR > 0.0:
            _pen_base = max(_pen_base, PRESS_FLOOR)
    else:
        _pen_base = float(np.interp(s, DEMO_PEN_S, DEMO_PEN_VAL)) if (PRESS_RAMP and DEMO_PEN_S is not None) else PEN_DES
    vn = float(np.dot(vtool, n_geo))
    _offs['vn_f'] = VN_FILT * _offs['vn_f'] + (1.0 - VN_FILT) * vn
    _k0 = max(0.0, SENSOR_ONSET - LAND_STEPS)
    _descending = DISCOVER_LAND and (k < _k0 + LAND_STEPS) and (_offs['cur_depth'] <= 0.0)
    if _descending:
        if k <= _k0:
            _z = LIFT_H
        else:
            _t = (k - _k0) / LAND_STEPS
            _z = -PRESS_FLOOR + (LIFT_H + PRESS_FLOOR) * 0.5 * (1.0 + np.cos(np.pi * _t))
        pen = -_z
    elif SOFT_LAND:
        _rate = (PEN_MAX if SOFT_STEPS < 1 else 0.005 / SOFT_STEPS)
        if _offs['cur_depth'] <= 0.0 and gap < SOFT_TOUCH_GAP:
            _offs['cur_depth'] = max(PRESS_FLOOR, _rate)
        if gap < SOFT_TOUCH_GAP:
            _offs['cur_depth'] = float(min(_offs['cur_depth'] + _rate, _pen_base))
        pen = _offs['cur_depth']
    else:
        pen = _pen_base
    pen = pen + _offs['dpress_now']
    _offs['contacted'] = _offs['contacted'] or (gap < 0.0)
    if _offs['contacted']:
        pen = pen + SEAT_DEPTH
    if _NO_FN:
        coef = 0.0
    elif _descending:
        coef = float(np.clip(PEN_KP * (-pen - gap) - PEN_KD * _offs['vn_f'], -FNMAX, FNMAX)) * _pgate
    elif PRESS_IN_CONTACT:
        if gap < 0.0:
            coef = float(np.clip(PEN_KP * (-pen - gap) - PEN_KD * _offs['vn_f'], -FNMAX, FNMAX)) * _pgate
        elif _offs['contacted'] and SEAT_F > 0.0 and gap < SEAT_BAND:
            coef = -SEAT_F * _pgate
        else:
            coef = 0.0
    else:
        _engage = (gap < 0.0) or (pen > 1e-4 and gap < ENGAGE_BAND)
        coef = float(np.clip(PEN_KP * (-pen - gap) - PEN_KD * _offs['vn_f'], -FNMAX, FNMAX)) * _pgate if _engage else 0.0
    Fn = coef * n_geo
    u_trk = ff + Ja.T @ (Frail + Flat) + Marm @ (TS_KD * (qd_ref - dq)) - CCANCEL * c
    _nkp = np.full(9, NULL_KP)
    if HOLD_THORAX:
        _nkp[0] = THORAX_KP
    if QOFS_ON_CONTACT and not (_offs['contacted'] or gap < QOFS_ENGAGE_GAP):
        _nkp[1:] = QOFS_PRE_KP
    q_err = q_ref_p - np.asarray(d.qpos)[ARM]
    _dmp = Marm @ (np.where(_nkp > NULL_KP, 2 * np.sqrt(_nkp) * dq, 0.0))
    _post = Marm @ (_nkp * q_err)
    u_trk = u_trk + _post - _dmp
    if TERM_DBG and k == TERM_DBG_K:
        _Mi = np.linalg.inv(Marm + 1e-6 * np.eye(9)); _L = np.linalg.inv(Ja @ _Mi @ Ja.T + 1e-9 * np.eye(3))
        _nf = lambda tau: float(n_geo @ (_L @ (Ja @ _Mi @ tau)))
        print(f"[term k={k} gap={gap*1000:+.1f}mm] ff={_nf(ff):+6.0f} rail+lat={_nf(Ja.T@(Frail+Flat)):+6.0f} "
              f"veltrack={_nf(Marm@(TS_KD*(qd_ref-dq))):+6.0f} coriolis={_nf(-CCANCEL*c):+6.0f} "
              f"posturePD={_nf(_post):+6.0f} damping={_nf(-_dmp):+6.0f} | u_trk_before_TT={_nf(u_trk):+6.0f} impedance={float(n_geo@Fn):+6.0f} N")
    if TRACK_TAN:
        _Minv = np.linalg.inv(Marm + 1e-6 * np.eye(9))
        _Lam = np.linalg.inv(Ja @ _Minv @ Ja.T + 1e-9 * np.eye(3))
        _Ftrk = _Lam @ (Ja @ _Minv @ u_trk)
        u_trk = u_trk - Ja.T @ (n_geo * float(n_geo @ _Ftrk))
        if TERM_DBG and k == TERM_DBG_K:
            print(f"[term k={k}] u_trk_AFTER_TRACK_TAN normal={_nf(u_trk):+.0f}N  (should be ~0 if the projection cancels it)")
    u = u_trk + Ja.T @ Fn
    if THORAX_KINEMATIC:
        u[0] = 0.0
    if PEN_CAP and gap < 0.0:
        _pen_act = -gap
        _cap = max(pen, 0.0) + PEN_CAP_MARGIN
        if _pen_act > _cap:
            _fc = PEN_CAP_KP * (_pen_act - _cap) - PEN_CAP_KD * _offs['vn_f']
            u = u + Ja.T @ (n_geo * float(np.clip(_fc, 0.0, FNMAX)))
    return u, dict(s=s, gap=gap, latoff=latoff, Fn=coef)

def roll(xq, xv, k0, n, dnull, dpress, dpace, dlift=0.0, ds0=0.0, record=False):
    """Roll n steps FROM state (xq,xv), reference index starting at k0. Returns (cost, xq_next, xv_next,
    traj, summary). The primitive both sampling rollouts and executed segments share → receding-horizon
    MPC = repeated solve()+roll()."""
    _offs['dnull'] = np.atleast_1d(dnull); _offs['dpress'] = np.atleast_1d(dpress).astype(float); _offs['dpace'] = float(dpace)
    _offs['dlift'] = float(dlift); _offs['ds0'] = float(ds0); _offs['cur_depth'] = 0.0; _offs['vn_f'] = 0.0; _offs['contacted'] = False
    d.qpos[:] = xq; d.qvel[:] = xv
    d.qpos[stick_qadr:stick_qadr + 3] = _stick_pos(k0); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
    mujoco.mj_forward(m, d)
    s_start = float(np.dot(d.geom_xpos[rock_gid] - P_START, RAIL_DIR))
    lat = press = eff = surf = 0.0; traj = []; blew = False; diag = None
    qlo = np.array(d.qpos[0:9], float); qhi = qlo.copy(); tmax = np.zeros(9)
    prev_v = np.zeros(9); prev_f = 0.0; feat_cost = 0.0; phi_win = np.zeros((N_W, 30))
    land = 0.0; _landed = False
    for kk in range(n):
        k = k0 + kk
        u, diag = control(k)
        if not np.all(np.isfinite(u)):
            blew = True; break
        _uc = np.clip(u, -u_max, u_max); tmax = np.maximum(tmax, np.abs(_uc))
        d.ctrl[:] = _uc
        for _ in range(NSUB):
            d.qpos[stick_qadr:stick_qadr + 3] = _stick_pos(k); d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
            d.qvel[stick_vadr:stick_vadr + 6] = 0.0
            if THORAX_KINEMATIC and DEMO_XS is not None:
                _kf = min(k, len(DEMO_XS) - 1); d.qpos[0] = DEMO_XS[_kf, 0]; d.qvel[0] = DEMO_XS[_kf, 10]
            if IMPEDANCE_SUBSTEP:
                mujoco.mj_forward(m, d); _u2, _ = control(k)
                d.ctrl[:] = np.clip(_u2, -u_max, u_max)
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            blew = True; break
        _q9 = np.asarray(d.qpos[0:9]); qlo = np.minimum(qlo, _q9); qhi = np.maximum(qhi, _q9)
        lat += diag['latoff'] ** 2; press += (abs(diag['Fn']) - PRESS_FN) ** 2 * (diag['gap'] < 0.02)
        eff += float(np.sum(_uc ** 2)); surf += max(0.0, diag['gap']) ** 2
        if not _landed:
            land += max(0.0, diag['gap']); _landed = diag['gap'] < 0.0
        if FEATURES:
            _fov = None if FORCE_FROM_PEN else _contact_normal()
            _phi, prev_v, prev_f = compute_phi(_uc, prev_v, prev_f, k=k, f_override=_fov)
            _win = WIN_OF_K[min(k, STEPS)]
            phi_win[_win] += _phi; feat_cost += float(W_TV[_win] @ _phi)
        if record:
            traj.append((d.qpos[0:9].copy(), _stick_pos(k).copy()))
            _TRACE['u'].append(_uc.copy()); _TRACE['fcmd'].append(float(diag['Fn']))
            _TRACE['fact'].append(_contact_normal()); _TRACE['gap'].append(float(diag['gap']) * 1000)
            _TRACE['q'].append(np.asarray(d.qpos[0:9]).copy()); _TRACE['v'].append(np.asarray(d.qvel[0:9]).copy())
    xqn, xvn = d.qpos.copy(), d.qvel.copy()
    sm = {'tmax': tmax, 'qlo': qlo, 'qhi': qhi, 'blew': blew, 'lat': (lat / max(n, 1)) ** 0.5 * 1000, 'land': land,
          'gapf': (diag['gap'] * 1000 if diag else 0.0), 's_end': float(np.dot(d.geom_xpos[rock_gid] - P_START, RAIL_DIR))}
    if blew:
        return 1e9, xqn, xvn, traj, sm                          # nan/blowup → high cost (pipeline masks these too)
    travel = sm['s_end'] - s_start
    if FEATURES:
        cost = feat_cost + 120.0 * max(0.0, sm['s_end'] - RAIL_LEN) * n + LAND_COST_W * land
        sm['phi_win'] = phi_win; sm['phi'] = phi_win.sum(0) / max(n, 1)
    else:
        overshoot = max(0.0, sm['s_end'] - RAIL_LEN)
        cost = (-15.0 * min(travel, RAIL_LEN) + 120.0 * overshoot + 8.0 * (lat / n) * 1000
                + 40.0 * (surf / n) * 5000 + 0.5 * (press / n) * 1e-3 + (eff / n) * 2e-6 + LAND_COST_W * land)
    sm['travel'] = travel * 100
    return cost, xqn, xvn, traj, sm

def _dials(v):
    _b = NNULL + NP + 1
    return v[:NNULL], v[NNULL:NNULL + NP], v[NNULL + NP], (v[_b] if NDISC else 0.0), (v[_b + 1] if NDISC else 0.0)
NDIAL_TOT = NNULL + NP + 1 + 2 * NDISC
_XINIT = np.zeros(m.nq); _XINIT[:len(INIT_QPOS)] = INIT_QPOS
if STEADY_START and QOFS is not None:
    _XINIT[0:9] = QOFS[0]
    print("[steady] STEADY_START: rock placed on the stick at q_of_s[0], no approach — isolating the impedance-press jitter")
def run_rollout(dnull, dpress, dpace, dlift=0.0, ds0=0.0, record=False):
    cost, _, _, traj, sm = roll(_XINIT, np.zeros(m.nv), 0, STEPS, dnull, dpress, dpace, dlift, ds0, record=record)
    sm['jrange'] = np.degrees(sm['qhi'] - sm['qlo']); sm['trms'] = np.zeros(9)
    return cost, traj, sm

PHINORM = FEATURES and os.environ.get('PHINORM', '1') != '0'
def cost_of(results):
    """Sample costs for the softmax. With FEATURES+PHINORM: cost = Σ (w/σ)·φ — each feature divided by its
    per-solve std across the K samples, so no single large-magnitude feature (Tau/JA/press) dominates."""
    if not (FEATURES and PHINORM):
        return np.array([r[0] for r in results])
    phis = np.array([r[4].get('phi_win', np.zeros((N_W, 30))) for r in results])
    blew = np.array([bool(r[4].get('blew', False)) for r in results])
    over = np.array([max(0.0, r[4].get('s_end', 0.0) - RAIL_LEN) for r in results])
    sig = phis.std(axis=0) + 1e-9
    lands = np.array([r[4].get('land', 0.0) for r in results])
    costs = np.einsum('kwf,wf->k', phis, W_TV / sig) + 120.0 * over + LAND_COST_W * lands
    return np.where(blew, 1e9, costs)

def solve(xq, xv, k0, H, Uinit, ndial, K, sig, LAM):
    P = Uinit.copy(); rng = np.random.RandomState(1000 + k0)
    for dial in range(ndial):
        eps = rng.standard_normal((K, len(P))) * sig * (0.6 ** dial)
        costs = cost_of([roll(xq, xv, k0, H, *_dials(P + e)) for e in eps])
        zc = (costs - costs.mean()) / (costs.std() + 1e-9)
        w = np.exp(-zc / max(LAM, 1e-3) * 0.5); w /= w.sum()
        P = P + w @ eps
    return P

if os.environ.get('IRL', '1'):
    assert FEATURES, "IRL needs FEATURES=1"
    K = int(os.environ.get('KOVER', '56')); LAM = float(D['mppi_lambda']); NDS = int(os.environ.get('IRL_DIALS', 3))
    IRL_MPC = os.environ.get('IRL_MPC', '1') != '0'
    MPC_H = int(os.environ.get('IRL_MPC_H', 30)); MPC_NAPP = int(os.environ.get('IRL_MPC_NAPPLY', '12')); MPC_NDIAL = int(os.environ.get('IRL_MPC_NDIAL', '3'))
    NULL_SIG = float(os.environ.get('NULL_SIG', D['ts_null_sigma']))
    sig = np.array([NULL_SIG] * NNULL + [float(os.environ.get('PRESS_SIG', '0.003'))] * NP + [float(os.environ.get('PACE_SIG', '0.4'))]
                   + ([LIFT_SIG, S0_SIG] if NDISC else []))
    LR = float(os.environ.get('IRL_LR', 0.4)); N_IRL = int(os.environ.get('IRL_ITERS', '1'))
    phi_demo = compute_demo_phi()
    W_TV[:] = np.maximum(W_TV, 0.0)
    _bmode = f"GAUSSIAN BASIS (K={K_BASIS}, σ={_sig_b:.0f})" if BASIS else "independent windows"
    print(f"[IRL] φ_demo computed | {N_IRL} iters, lr={LR}, K={K}, n_w={N_W} windows | weights: {_bmode}")
    W_hist = []; err_hist = []; trav_hist = []; gap_hist = []; qn_hist = []; last_traj = None
    CX_hist = []; CU_hist = []
    USE_LBFGS = os.environ.get('LBFGS', '1') != '0'
    MHIST = int(os.environ.get('LBFGS_HIST', 5)); GAMMA_CAP = float(os.environ.get('LBFGS_GAMMA_CAP', 200.0))
    LS_STEPS = int(os.environ.get('LS_STEPS', 4)); LS_BASE = float(os.environ.get('LS_BASE', 0.5))
    LBFGS_ACCEPT = os.environ.get('LBFGS_ACCEPT', '1') != '0'
    # UNREALIZABLE → the feature-match gradient never vanishes → L-BFGS inflates the weights without bound
    REG_L1 = float(os.environ.get('REG_LAMBDA', 0.01)); REG_L2 = float(os.environ.get('REG_BETA', 0.02))
    NVAR0 = K_BASIS if BASIS else N_W
    def _set_w(vf):
        v = np.clip(vf, 0.0, None).reshape(NVAR0, 30)
        W_TV[:] = (B_WINDOW @ v) if BASIS else v
        return v
    def _solve_chal(vf, seed):
        _set_w(vf)
        for _l in _TRACE.values():
            _l.clear()
        if IRL_MPC:
            xq = _XINIT.copy(); xv = np.zeros(m.nv); U = np.zeros(NDIAL_TOT)
            traj = []; phi_win = np.zeros((N_W, 30)); sm = {}; s0 = None; k = 0
            while k < STEPS:
                H = min(MPC_H, STEPS - k); napp = min(MPC_NAPP, STEPS - k)
                U = solve(xq, xv, k, H, U, MPC_NDIAL, K, sig, LAM)
                _c, xq, xv, seg, sm = roll(xq, xv, k, napp, *_dials(U), record=True)
                if s0 is None:
                    s0 = sm['s_end'] - sm['travel'] / 100.0
                traj += seg; phi_win += sm.get('phi_win', np.zeros((N_W, 30)))
                k += napp
            sm['phi_win'] = phi_win; sm['phi'] = phi_win.sum(0) / max(STEPS, 1)
            sm['travel'] = (sm['s_end'] - (s0 or 0.0)) * 100.0
            return traj, sm
        P = np.zeros(NDIAL_TOT); rng = np.random.RandomState(seed)
        for dial in range(NDS):
            eps = rng.standard_normal((K, NDIAL_TOT)) * sig * (0.5 ** dial)
            costs = cost_of([roll(_XINIT, np.zeros(m.nv), 0, STEPS, *_dials(P + e)) for e in eps])
            zc = (costs - costs.mean()) / (costs.std() + 1e-9); w = np.exp(-zc / max(LAM, 1e-3) * 0.5); w /= w.sum(); P = P + w @ eps
        _, _, _, traj, sm = roll(_XINIT, np.zeros(m.nv), 0, STEPS, *_dials(P), record=True)
        return traj, sm
    FORCE_FIT_W = float(os.environ.get('FORCE_FIT_W', 1.0))
    _FORCE_IDX = [_KEYS.index('press_force'), _KEYS.index('press_capacity')]
    def _ascent_grad(sm):
        _sc = np.abs(phi_demo) + np.abs(sm['phi_win']) + 0.1
        gf = (sm['phi_win'] - phi_demo) / _sc
        if FORCE_FIT_W != 1.0:
            gf = gf.copy(); gf[:, _FORCE_IDX] *= FORCE_FIT_W
        A = (B_WINDOW.T @ gf) if BASIS else gf
        return A.flatten(), float(np.linalg.norm(gf))
    def _reg_grad(vf):
        return REG_L1 * np.sign(vf) + 2.0 * REG_L2 * vf
    def _lossgrad(A_flat, vf):
        return -A_flat + _reg_grad(vf)
    def _merit(A_flat, vf):
        return float(np.linalg.norm(_lossgrad(A_flat, vf)))
    def _lbfgs_dir(GL, S, Y):
        q = GL.copy(); al = []
        for s, y in zip(reversed(S), reversed(Y)):
            rho = 1.0 / (float(y @ s) + 1e-12); a = rho * float(s @ q); q = q - a * y; al.append((rho, a))
        gamma = (float(S[-1] @ Y[-1]) / (float(Y[-1] @ Y[-1]) + 1e-12)) if S else LR
        r = min(max(gamma, 1e-6), GAMMA_CAP) * q
        for (s, y), (rho, a) in zip(zip(S, Y), reversed(al)):
            b = rho * float(y @ r); r = r + s * (a - b)
        return r
    def _cap():
        return np.array(_TRACE['q']), np.array(_TRACE['v']), np.array(_TRACE['u'])
    def _capture(it, traj, sm, cap, fm):
        _set_w(var)
        _carm = np.array([a for a, _ in traj]); _Tm = min(len(_carm), len(DEMO_XS))
        qn = float(np.degrees(np.sqrt(np.mean((_carm[:_Tm] - DEMO_XS[:_Tm, 0:9]) ** 2, axis=0))).mean())
        _tq, _tv, _tu = cap
        _cx = np.zeros((len(_tq) + 1, 20)); _cx[0, 0:9] = _XINIT[0:9]; _cx[1:, 0:9] = _tq; _cx[1:, 10:19] = _tv
        CX_hist.append(_cx[:len(DEMO_XS)]); CU_hist.append(_tu[:len(DEMO_XS) - 1])
        _wm = W_TV.mean(0)
        W_hist.append(W_TV.copy()); err_hist.append(fm); trav_hist.append(sm['travel']); gap_hist.append(sm['gapf']); qn_hist.append(qn)
        print(f"  [IRL {it:2d}] q_norm={qn:5.2f}deg  ‖φ_chal−φ_demo‖={fm:5.2f}  travel={sm['travel']:5.1f}cm gap={sm['gapf']:+.1f}mm  "
              f"| top-w̄: " + " ".join(f"{_KEYS[i]}={_wm[i]:.1f}" for i in np.argsort(-_wm)[:4]))
    print(f"[IRL] outer optimizer: {'L-BFGS (hist=%d, line-search=%d)' % (MHIST, LS_STEPS) if USE_LBFGS else 'SGD'}")
    var = (THETA if BASIS else W_TV).flatten().astype(float)
    Sh = []; Yh = []
    last_traj, sm = _solve_chal(var, 0); cap = _cap()
    if os.environ.get('DUMP_PHI'):
        _carm = np.array([a for a, _ in last_traj]); _Tm = min(len(_carm), len(DEMO_XS))
        _qn = float(np.degrees(np.sqrt(np.mean((_carm[:_Tm] - DEMO_XS[:_Tm, 0:9]) ** 2, axis=0))).mean())
        np.savez(os.environ['DUMP_PHI'], phi_demo=phi_demo, phi_chal=sm['phi_win'], q_norm=_qn,
                 s_end=sm.get('s_end', 0.0), gapf=sm.get('gapf', 0.0))
        print(f"[DUMP_PHI] wrote {os.environ['DUMP_PHI']}  q_norm={_qn:.2f} gap={sm.get('gapf',0):+.1f}mm")
        sys.exit(0)
    A, fm = _ascent_grad(sm); GL = _lossgrad(A, var)
    for it in range(N_IRL):
        _capture(it, last_traj, sm, cap, fm)
        if it == N_IRL - 1:
            break
        pdir = -_lbfgs_dir(GL, Sh, Yh) if USE_LBFGS else -LR * GL
        if not np.all(np.isfinite(pdir)) or float(np.linalg.norm(pdir)) < 1e-12:
            pdir = -LR * GL
        cur = _merit(A, var); alpha = 1.0; best = None; _seed = 100000 + it
        for _ls in range(max(1, LS_STEPS) if USE_LBFGS else 1):
            vtry = np.clip(var + alpha * pdir, 0.0, None)
            tr2, sm2 = _solve_chal(vtry, _seed); cap2 = _cap()
            A2, fm2 = _ascent_grad(sm2); mrt = _merit(A2, vtry)
            if best is None or mrt < best[0]:
                best = (mrt, vtry, tr2, sm2, cap2, A2, fm2)
            if mrt < cur:
                break
            alpha *= LS_BASE
        mrt, vtry, tr2, sm2, cap2, A2, fm2 = best
        if LBFGS_ACCEPT and fm2 >= fm - 1e-6:
            print(f"  [IRL {it:2d}] REJECT: best fm {fm2:.2f} ≥ current {fm:.2f} — weights held, L-BFGS reset")
            Sh.clear(); Yh.clear()
            continue
        GL2 = _lossgrad(A2, vtry)
        s_step = vtry - var; y_step = GL2 - GL
        if USE_LBFGS and float(s_step @ y_step) > 1e-10:
            Sh.append(s_step); Yh.append(y_step)
            if len(Sh) > MHIST:
                Sh.pop(0); Yh.pop(0)
        var, last_traj, sm, cap, A, GL, fm = vtry, tr2, sm2, cap2, A2, GL2, fm2
    _set_w(var)
    if BASIS:
        THETA[:] = np.clip(var.reshape(NVAR0, 30), 0.0, None)
    _wm = W_TV.mean(0)
    print("[IRL] recovered (window-mean) weights:", {_KEYS[i]: round(float(_wm[i]), 2) for i in np.argsort(-_wm)[:8]})
    if os.environ.get('WRITE_ANIM'):
        _src = np.load(DEMO_NPZ, allow_pickle=True)
        _srcf = set(_src.files)
        if 'keys_run' in _srcf:
            _kr = [str(k) for k in _src['keys_run']]
            _idx = [_KEYS.index(k) for k in _kr]
        else:
            _kr = list(_KEYS); _idx = list(range(len(_KEYS)))
        _wit = np.array([Wi[:, _idx] for Wi in W_hist])
        _cx_obj = np.empty(len(CX_hist), object); _cu_obj = np.empty(len(CU_hist), object)
        for _i in range(len(CX_hist)): _cx_obj[_i] = CX_hist[_i]; _cu_obj[_i] = CU_hist[_i]
        _ap = os.environ['WRITE_ANIM']
        _z3 = np.zeros((0, 3), np.float32)
        _get = lambda k, d: (_src[k] if k in _srcf else d)
        np.savez(_ap, keys_run=np.array(_kr), n_w=N_W, w_iters=_wit, q_norm_iters=np.array(qn_hist),
                 chal_xs=_cx_obj, chal_us=_cu_obj,
                 demo_xs=_get('demo_xs', DEMO_XS), demo_us=_get('demo_us', np.zeros((0, 9))),
                 stick_traj=(STICK_TRAJ if STICK_TRAJ is not None else _z3),
                 phis_samples=_get('phis_samples', np.zeros(0)), sample_qpos=_get('sample_qpos', _z3),
                 sample_phis=_get('sample_phis', _z3), sample_costs=_get('sample_costs', np.zeros(0)),
                 chal_react_sim=(np.array(_TRACE['fact'], float) if len(_TRACE['fact']) else np.zeros(0)))
        print(f"[IRL] wrote anim npz → {_ap}  ({len(W_hist)} iters, keys_run={len(_kr)}, n_w={N_W})")
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    Wh = np.array(W_hist); Wm = Wh.mean(axis=1); ii = np.arange(len(err_hist))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8)); ax = ax.ravel()
    ax[0].plot(ii, qn_hist, '-o', color='C0', label='q_norm (deg) — MO_IRL metric')
    ax[0].plot(ii, err_hist, '-s', color='C3', label='feature-match err')
    ax[0].set_title('convergence (both should ↓)'); ax[0].set_xlabel('IRL iteration'); ax[0].legend(fontsize=8)
    for i in np.argsort(-_wm)[:6]:
        ax[1].plot(ii, Wm[:, i], '-o', label=_KEYS[i])
    ax[1].set_title('window-mean weights per iteration'); ax[1].set_xlabel('IRL iteration'); ax[1].legend(fontsize=7)
    ax[2].plot(ii, trav_hist, '-o', label='travel (cm)'); ax[2].plot(ii, gap_hist, '-o', label='final gap (mm)')
    ax[2].axhline(RAIL_LEN * 100, ls='--', color='gray', lw=1, label='rail len'); ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set_title('challenger trajectory vs iteration'); ax[2].set_xlabel('IRL iteration'); ax[2].legend(fontsize=8)
    for i in np.argsort(-_wm)[:6]:
        ax[3].plot(np.arange(N_W), W_TV[:, i], '-o', label=_KEYS[i])
    _btag = f'Gaussian basis K={K_BASIS}' if BASIS else f'{N_W} independent windows'
    ax[3].set_title(f'RECOVERED time-varying weights ({_btag})'); ax[3].set_xlabel('window (early → late scrape)')
    ax[3].set_ylabel('weight'); ax[3].legend(fontsize=7)
    fig.tight_layout(); _ip = os.environ.get('IRL_PLOT', 'runs/irl_iterations.png'); fig.savefig(_ip, dpi=110)
    print(f"[IRL] wrote {_ip}")
    _traj_out = last_traj; summ = {**sm, 'jrange': np.degrees(sm['qhi'] - sm['qlo']), 'trms': np.zeros(9)}

elif os.environ.get('DIAG_NULL') is not None:
    import sys
    _val = float(os.environ.get('DIAG_NULL') or '2.0')
    print(f"[DIAG] posture authority: force null-dir 0 in {{0, {_val:+.1f}, {-_val:+.1f}}} (press=0, pace=0)")
    for _coef in (0.0, _val, -_val):
        _dn = np.zeros(max(NNULL, 1)); _dn[0] = _coef
        _c, _tj, _sm = run_rollout(_dn, 0.0, 0.0, record=True)
        _cells = []
        for _kk in (0, 6, 12, 20, 30, 45, 66):
            _q9, _sp = _tj[min(_kk, len(_tj) - 1)]
            d.qpos[0:9] = _q9; d.qpos[stick_qadr:stick_qadr + 3] = _sp; d.qpos[stick_qadr + 3:stick_qadr + 7] = STICK_QUAT
            mujoco.mj_forward(m, d)
            _rp = d.geom_xpos[rock_gid]; _sc = d.xpos[stick_bid]; _sa = d.xmat[stick_bid].reshape(3, 3)[:, 2]
            _tp = np.clip((_rp - _sc) @ _sa, -STICK_HALF, STICK_HALF); _g = (np.linalg.norm(_rp - (_sc + _tp * _sa)) - STICK_R - ROCK_R) * 1000
            _cells.append(f"t{_kk:2d}[elb {math.degrees(_q9[6]):5.1f} gap {_g:+5.1f}]")
        print(f"  dnull0={_coef:+.1f}:  " + "  ".join(_cells))
    sys.exit(0)
elif os.environ.get('MPPI'):
    K = int(os.environ.get('KOVER', D['K'])); LAM = float(D['mppi_lambda'])
    NULL_SIG = float(os.environ.get('NULL_SIG', D['ts_null_sigma']))
    PACE_SIG = float(os.environ.get('PACE_SIG', '0.4')); PRESS_SIG = float(os.environ.get('PRESS_SIG', '0.003'))
    sig = np.array([NULL_SIG] * NNULL + [PRESS_SIG] * NP + [PACE_SIG] + ([LIFT_SIG, S0_SIG] if NDISC else []))
    HORIZON = int(os.environ.get('HORIZON', 30)); N_APPLY = int(os.environ.get('N_APPLY', 10))
    NDIAL = int(os.environ.get('NDIAL', 4)); REPLAN = os.environ.get('REPLAN', '1') != '0'
    if REPLAN:
        print(f"[MPC] receding-horizon: solve H={HORIZON}, execute {N_APPLY} then re-solve | K={K}, {NDIAL} dials/solve")
        xq = _XINIT.copy(); xv = np.zeros(m.nv); U = np.zeros(NDIAL_TOT); _traj_out = []; k = 0
        TMAX = np.zeros(9); QLO = xq[:9].copy(); QHI = xq[:9].copy(); sm = {}
        while k < STEPS:
            H = min(HORIZON, STEPS - k); napp = min(N_APPLY, STEPS - k)
            U = solve(xq, xv, k, H, U, NDIAL, K, sig, LAM)
            cost, xq, xv, seg, sm = roll(xq, xv, k, napp, *_dials(U), record=_RENDER)
            _traj_out += seg; TMAX = np.maximum(TMAX, sm['tmax']); QLO = np.minimum(QLO, sm['qlo']); QHI = np.maximum(QHI, sm['qhi'])
            print(f"  [replan t={k:2d}→{k+napp:2d}] null={np.round(U[:NNULL], 2)} press̄={np.mean(U[NNULL:NNULL + NP]):+.3f} pace={U[NNULL + NP]:+.2f}"
                  + (f" lift={U[NNULL + NP + 1]:+.2f} s0={np.clip(U[NNULL + NP + 2]*S0_SCALE,-S0_MAX,S0_MAX)*100:+.1f}cm" if NDISC else "")
                  + f"  →  tool s={sm['s_end'] * 100:5.1f}cm  gap={sm['gapf']:+5.1f}mm")
            k += napp
        summ = {'travel': sm['s_end'] * 100, 'lat': sm['lat'], 'gapf': sm['gapf'],
                'jrange': np.degrees(QHI - QLO), 'tmax': TMAX, 'trms': np.zeros(9)}
    else:
        P = np.zeros(NDIAL_TOT); rng = np.random.RandomState(0); NDS = max(1, int(D['n_dial']))
        print(f"[MPPI] single-shot open-loop: K={K}, {NDS} dials")
        for dial in range(NDS):
            eps = rng.standard_normal((K, NDIAL_TOT)) * sig * (0.5 ** dial)
            costs = cost_of([roll(_XINIT, np.zeros(m.nv), 0, STEPS, *_dials(P + e)) for e in eps])
            zc = (costs - costs.mean()) / (costs.std() + 1e-9); w = np.exp(-zc / max(LAM, 1e-3) * 0.5); w /= w.sum(); P = P + w @ eps
            print(f"  dial {dial}: cost min={costs.min():7.2f} mean={costs.mean():7.2f}")
        _dn, _dp, _dpc, _dl, _ds0 = _dials(P)
        cost, _traj_out, summ = run_rollout(_dn, _dp, _dpc, _dl, _ds0, record=_RENDER)
else:
    cost, _traj_out, summ = run_rollout(np.zeros(NNULL), 0.0, 0.0, record=_RENDER)
    print(f"[tracking] nominal rollout (no sampling)")
_traj = _traj_out if _RENDER else None
nan_at = None; travel = summ.get('travel', 0.0); gapf = summ.get('gapf', 0.0) / 1000.0
if 'jrange' in summ:
    _jr = summ['jrange']
    print(f"[joint ranges deg] THORACIC={_jr[0]:.1f}  clavicle={_jr[1]:.1f}  shZ={_jr[2]:.1f} shX={_jr[3]:.1f} "
          f"shY={_jr[4]:.1f}  elbowZ={_jr[5]:.1f}  wristZ={_jr[7]:.1f}")
    _lbl = ['thor', 'clav', 'shZ', 'shX', 'shY', 'elZ', 'elY', 'wrZ', 'wrX']
    print("[torque N·m  peak|limit] " + " ".join(f"{_lbl[i]}={summ['tmax'][i]:.0f}|{u_max[i]:.0f}" for i in range(9)))
if _TRACE['u']:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    U = np.array(_TRACE['u']); t = np.arange(len(U)) * DT; _g = np.array(_TRACE['gap'])
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    a0.plot(t, _g, color='C2', lw=1.8, label='rock–stick gap (mm)')
    a0.axhline(0, color='k', lw=0.8, label='stick surface (gap=0)')
    a0.fill_between(t, _g, 0, where=(_g < 0), color='C2', alpha=0.15)
    a0.set_ylabel('normal position (mm)'); a0.legend(fontsize=8, loc='upper right')
    a0.set_title('Position — tool relative to the stick surface (below 0 = pressing in)')
    for i in range(9):
        a1.plot(t, U[:, i], label=_lbl[i], lw=1.4)
    a1.axhline(0, color='k', lw=0.5); a1.set_ylabel('applied torque (N·m)'); a1.legend(ncol=9, fontsize=7, loc='upper center')
    a1.set_title('Torque — applied joint torques')
    a2.plot(t, np.abs(np.array(_TRACE['fcmd'])), label='commanded impedance |Fₙ|', color='C3', lw=1.6)
    a2.plot(t, _TRACE['fact'], label='emergent MuJoCo reaction', color='C0', lw=1.8)
    if DEMO_SENSOR_F is not None:
        a2.plot(t, DEMO_SENSOR_F[:len(U)], label='demo sensor |F| (target)', color='k', ls='--', lw=1.4)
    a2.set_ylabel('normal force (N)'); a2.set_xlabel('time (s)'); a2.legend(fontsize=8); a2.set_title('Force — emergent reaction vs commanded vs sensor')
    fig.tight_layout(); _pp = os.environ.get('PLOT', 'runs/torque_force_trace.png'); fig.savefig(_pp, dpi=110)
    _gmm = np.array(_TRACE['gap']); _fa = np.array(_TRACE['fact'])
    _ncon = _gmm < 0.0
    _tail = np.zeros(len(_gmm), bool); _tail[len(_gmm)//3:] = True
    _sust = _ncon & _tail
    _fc = np.array(_TRACE['fcmd'])
    _sf = float(np.median(_fa[_sust])) if _sust.any() else 0.0
    _sc = float(np.median(np.abs(_fc[_sust]))) if _sust.any() else 0.0
    _sp = float(np.median(-_gmm[_sust])) if _sust.any() else 0.0
    _2h = _fa[2*len(_fa)//5:]; _jit = float(np.abs(np.diff(_2h)).mean()) if len(_2h) > 1 else 0.0
    _contact_frac = float((_gmm[2*len(_gmm)//5:] < 0.2).mean())
    _sens_med = _track = 0.0
    if DEMO_SENSOR_F is not None and _sust.any():
        _ss = DEMO_SENSOR_F[:len(_fa)]; _sens_med = float(np.median(_ss[_sust])); _track = float(np.sqrt(np.mean((_fa[_sust] - _ss[_sust])**2)))
    print(f"[trace] wrote {_pp}  ({len(U)} steps)  peak|u|={np.abs(U).max():.0f}N·m  "
          f"SUSTAINED: reaction(a)={_sf:.0f}N  vs sensor {_sens_med:.0f}N  (track RMS {_track:.0f}N)  @ pen {_sp:.1f}mm (max {max(0.0,-_gmm.min()):.1f}) | "
          f"JITTER {_jit:.1f}  contact {_contact_frac*100:.0f}%  {'CLEAN' if _jit < 8 else 'JITTERY'}")
    _above = _gmm > 0.1
    _imp_above = int((_above & (np.abs(_fc) > 0.5)).sum())
    print(f"[gate] impedance commanded ABOVE surface: {_imp_above}/{int(_above.sum())} above-frames"
          + (f"  (max |fcmd| there {np.abs(_fc[_above]).max():.0f}N)" if _above.any() else "")
          + f"  | actual contact(fact>0.5): {int((_fa>0.5).sum())}/{len(_fa)} frames")
    _qdep = np.array(_TRACE['q']) if _TRACE['q'] else None
    if _qdep is not None and DEMO_XS is not None:
        _Tq = min(len(_qdep), len(DEMO_XS))
        _qn = float(np.degrees(np.sqrt(np.mean((_qdep[:_Tq] - DEMO_XS[:_Tq, 0:9]) ** 2, axis=0))).mean())
        print(f"[eval] deployed q_norm vs demo = {_qn:.2f} deg")
    if os.environ.get('TRACE_DUMP'):
        np.savez(os.environ['TRACE_DUMP'], gap=_gmm, fact=_fa, fcmd=_fc, q=(_qdep if _qdep is not None else np.zeros((0, 9))))
        print(f"[gate] dumped gap/fact/fcmd/q → {os.environ['TRACE_DUMP']}")
if _RENDER and _traj:
    import PIL.Image as _PIL
    mr = mujoco.MjModel.from_xml_path(PR); dr = mujoco.MjData(mr)
    dr.qpos[:len(INIT_QPOS)] = INIT_QPOS; mujoco.mj_forward(mr, dr)
    _pb = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_BODY, 'middle_pelvis')
    _sj = mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_JOINT, 'stick_freejoint')
    _sb = mr.jnt_bodyid[_sj]; _sq = mr.jnt_qposadr[_sj]
    up = dr.xmat[_pb].reshape(3, 3)[:, 1].copy(); up /= np.linalg.norm(up); tgt = np.array([0., 0, 1.])
    ax = np.cross(up, tgt); s_ = np.linalg.norm(ax); c_ = float(up @ tgt); ax /= s_; ang = math.atan2(s_, c_)
    qaa = np.zeros(4); mujoco.mju_axisAngle2Quat(qaa, ax, ang); Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, qaa); Rm = Rm.reshape(3, 3)
    qR = np.zeros(4); mujoco.mju_mat2Quat(qR, np.ascontiguousarray(Rm.flatten()))
    swp = dr.xpos[_sb].copy(); swq = dr.xquat[_sb].copy()
    nq = np.zeros(4); mujoco.mju_mulQuat(nq, qR, mr.body_quat[_pb].copy()); mr.body_quat[_pb] = nq; mr.body_pos[_pb] = Rm @ mr.body_pos[_pb]
    mr.body_pos[_sb] = [0, 0, 0]; mr.body_quat[_sb] = [1, 0, 0, 0]
    nsq = np.zeros(4); mujoco.mju_mulQuat(nsq, qR, swq); dr.qpos[_sq:_sq + 3] = Rm @ swp; dr.qpos[_sq + 3:_sq + 7] = nsq
    _stick_up_quat = dr.qpos[_sq + 3:_sq + 7].copy()
    def _set_render(arm, spos):
        dr.qpos[0:9] = arm; dr.qpos[_sq:_sq + 3] = Rm @ spos; dr.qpos[_sq + 3:_sq + 7] = _stick_up_quat
        dr.qvel[:] = 0.0; mujoco.mj_forward(mr, dr)
    ren = mujoco.Renderer(mr, 480, 640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    _midarm, _midstk = _traj[len(_traj) // 2]; _set_render(_midarm, _midstk)
    _pts = np.array([dr.xpos[b] for b in range(1, mr.nbody)])
    cam.lookat[:] = 0.5 * (_pts.max(0) + _pts.min(0))
    cam.distance = float(os.environ.get('CAM_DIST', 1.4 * np.linalg.norm(_pts.max(0) - _pts.min(0))))
    cam.elevation = -8; cam.azimuth = float(os.environ.get('CAM_AZ', 210))
    frames = []
    for arm, spos in _traj:
        _set_render(arm, spos); ren.update_scene(dr, cam); frames.append(_PIL.fromarray(ren.render()))
    ren.close()
    _out = os.environ.get('OUT', f'runs/rollout_{LAW}.gif')
    frames[0].save(_out, save_all=True, append_images=frames[1:], duration=60, loop=0)
    print(f"[render] wrote {_out} ({len(frames)} frames, upright display)")
print(f"\nRESULT LAW={LAW}: " + (f"NAN at step {nan_at}" if nan_at is not None else f"ran {STEPS} steps clean") +
      f"  |  travel along rail = {travel:.1f}cm of {RAIL_LEN*100:.1f}cm  |  final gap = {gapf*1000:.1f}mm")
