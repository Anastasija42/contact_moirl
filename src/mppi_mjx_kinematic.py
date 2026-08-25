"""mppi_mjx_kinematic.py — MJX/JAX port of KinematicMPPI_CPU.

Drops in as the IRL solver replacement, same install_as_solver API:

    from mppi_mjx_kinematic import KinematicMPPI_MJX
    kin = KinematicMPPI_MJX(human_mppi, horizon=40, num_samples=512)
    kin.install_as_solver(human_mppi)
    IRL = MO_IRL(mo_args, irl_args)
    IRL.solve()

Design notes
============
* Keeps the **same 17-feature KEYS_RUN ordering** as mppi_cpu.KEYS_RUN
  so IRL weights / phi_scale / line search require no retuning.
* Per-feature formulas mirror `_compute_phi_fast` exactly (linear norms,
  divisions by phi-scale constants like /50, /100, /20, /stick_radius,
  /rail_len) — NOT Crocoddyl-quadratic. Goal: numerical parity with CPU.
* Rollout: jax.vmap over K samples, jax.lax.scan over H horizon steps.
  JIT-compiled on first solve() call.

TODOs (clearly marked in _running_phi):
* `surface`, `approach`, `traveled`: same formulas as CPU once the core
  rollout is validated. They reuse the rock-stick geometry that
  `rail_lat` and `press_force` already compute.
* `JTC`: kept at 0 to match CPU's `phi_jtc = 0.0` (skipped for speed).
"""
from __future__ import annotations

import os
import time
import numpy as np

try:
    from jax.extend import backend as _jax_extend_backend
    if not hasattr(_jax_extend_backend, "backends"):
        from jax._src.xla_bridge import backends as _xb_backends
        _jax_extend_backend.backends = _xb_backends   # type: ignore[attr-defined]
except Exception:
    pass

import mujoco

from mppi_cpu import KEYS_RUN, ENG_GROUPS, ELBOW_TRACK_REF, ELBOW_LOCAL

def _mppi_phi_norm_enabled():
    """MPPI_PHI_NORM env flag (default OFF).

    When ON, the MPPI softmax GENERATION cost is evaluated on σ-NORMALIZED
    features — the per-feature weights are divided by their σ so that
        Σ_k w_k · φ_k   becomes   Σ_k (w_k/σ_k) · φ_k = Σ_k w_k · (φ_k/σ_k)
    i.e. the nominal is selected in the SAME σ-normalized coordinates the
    MO-IRL gradient reasons in (see MO_IRL._apply_per_feature_scale). This
    fixes the feature-normalization inconsistency where the IRL optimizes in
    σ-space but the raw softmax was dominated by large-σ features (JA/Tau/…)
    regardless of their learned weight. OFF ⇒ every code path below is a
    no-op ⇒ byte-identical to the pre-flag behavior.
    """
    return os.environ.get('MPPI_PHI_NORM', '').strip().lower() in (
        '1', 'true', 'on', 'yes')

def _mppi_anneal_enabled():
    """MPPI_ANNEAL env flag (default OFF).

    When ON, the dial loop's per-iteration noise scale AND softmax temperature
    follow a HIGH→LOW geometric ramp (explore-hot → converge-cold), tuned by
        MPPI_ANNEAL_NDIAL   # of dial iters              (default 5)
                            # NOTE: the dial loop is UNROLLED at trace time, so
                            # each +1 here is another full mjx rollout in the XLA
                            # graph → linearly slower compile. Keep it small.
        MPPI_ANNEAL_SIG_HI / _SIG_LO   noise-scale ramp   (default 1.0 → 0.04)
        MPPI_ANNEAL_LAM_HI / _LAM_LO   softmax temperature(default 6·λ → λ)
    This deepens the fixed 3-level ``dial_decay`` stub into a real anneal and
    adds the λ-sharpening the fixed-λ softmax lacks — the sampler-side analog
    of the reach-toy schedule that let a full-horizon MPPI match the OCP, here
    run INSIDE each receding replan so every re-solve explores from its fresh
    state then converges to the argmin. OFF ⇒ dial loop byte-identical to before.
    """
    return os.environ.get('MPPI_ANNEAL', '').strip().lower() in (
        '1', 'true', 'on', 'yes')

class KinematicMPPI_MJX:
    """MJX-backed MPPI kinematic solver. Same public interface as
    KinematicMPPI_CPU.

    The IRL pipeline calls `solve(human_mppi, T)` which runs a
    receding-horizon MPPI loop. Each MPPI step batches K rollouts on GPU.
    """

    def __init__(self, human_mppi, *, horizon=40, num_samples=256,
                 alpha=1.0, beta=0.1, lambda_=0.5,
                 n_dial=3, dial_decay=0.5,
                 target_force=40.0, target_rail_vel=0.5,
                 noise_scale=1.0, spline_M=None,
                 force_control=False, fc_kp=0.5, fc_kd=0.0, fc_sign=1.0,
                 fc_kp_close=0.0, fc_sign_close=1.0, fc_kv=0.0,
                 contact_solref=None, bound_progress_vel=False,
                 stick_follow=False, force_ema=1.0, blend_steps=0,
                 press_control=False, press_sigma=15.0, press_max=120.0,
                 press_ema=1.0, press_ref=0.0,
                 rail_walk_pace=False, rail_walk_pace_sigma=0.15,
                 virtual_target=False, vt_kp=2600.0, vt_kd=0.0,
                 vt_dz_sigma=0.006, vt_dz_max=0.03):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
        self.jax = jax
        self.jnp = jnp
        self.mjx = mjx

        model = human_mppi.mj_model
        self._mj_model = model
        from MPPI_MJX import _prepare_model_for_mjx
        _prepare_model_for_mjx(model)

        model.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        if __import__('os').environ.get('DISABLE_EQ'):
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
            print("[DISABLE_EQ] equality constraints OFF")
        _jit_eps = __import__('os').environ.get('JITTER')
        if _jit_eps:
            from mujoco.mjx._src import smooth as _mjx_smooth
            import jax.numpy as _jnpj
            _eps = float(_jit_eps)
            if not getattr(_mjx_smooth, '_jitter_patched', False):
                _orig_fm = _mjx_smooth.factor_m
                def _fm_jitter(m, d, _orig=_orig_fm, _e=_eps):
                    _qM = d._impl.qM
                    _qMj = _qM + _e * _jnpj.eye(_qM.shape[-1], dtype=_qM.dtype)
                    return _orig(m, d.replace(_impl=d._impl.replace(qM=_qMj)))
                _mjx_smooth.factor_m = _fm_jitter
                _mjx_smooth._jitter_patched = True
                print(f"[JITTER] factor_m qM += {_eps:.0e}*I (diagonal regularization)")
        # Diagnostics for the GPU-only cho_factor nan (fused-graph codegen hypothesis):
        #   QMCHK=1 → print isfinite(qM) going INTO cho_factor (finite-in/nan-out vs nan-in?)
        #             nan clears on GPU, it's a fusion/codegen defect (and we keep the GPU).
        _qmchk = __import__('os').environ.get('QMCHK')
        _optbar = __import__('os').environ.get('OPTBAR')
        if _qmchk or _optbar:
            from mujoco.mjx._src import smooth as _mjx_smooth
            import jax as _jaxd, jax.numpy as _jnpd
            if not getattr(_mjx_smooth, '_diag_patched', False):
                _orig_fm2 = _mjx_smooth.factor_m
                def _fm_diag(m, d, _orig=_orig_fm2):
                    _qM = d._impl.qM
                    if _qmchk:
                        _diag = _jnpd.diagonal(_jnpd.nan_to_num(_qM), axis1=-2, axis2=-1)
                        _jaxd.debug.print(
                            "[qM] all_finite={f} nan_ct={n} absmax={mx} min_abs_diag={md} argmin_dof={ad}",
                            f=_jnpd.isfinite(_qM).all(), n=_jnpd.isnan(_qM).sum(),
                            mx=_jnpd.max(_jnpd.abs(_jnpd.nan_to_num(_qM))),
                            md=_jnpd.min(_jnpd.abs(_diag)),
                            ad=_jnpd.argmin(_jnpd.abs(_diag)))
                    if _optbar:
                        _qM = _jaxd.lax.optimization_barrier(_qM)
                        d = d.replace(_impl=d._impl.replace(qM=_qM))
                    _out = _orig(m, d)
                    if _optbar:
                        _out = _jaxd.lax.optimization_barrier(_out)
                    return _out
                _mjx_smooth.factor_m = _fm_diag
                _mjx_smooth._diag_patched = True
                print(f"[DIAG] factor_m patched: QMCHK={bool(_qmchk)} OPTBAR={bool(_optbar)}")

        if contact_solref is not None:
            sr = [float(x) for x in str(contact_solref).split()]
            g1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
            g2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom")
            n_patched = 0
            for p in range(model.npair):
                pg = {int(model.pair_geom1[p]), int(model.pair_geom2[p])}
                if pg == {g1, g2}:
                    model.pair_solref[p, :len(sr)] = sr
                    n_patched += 1
            print(f"[KinematicMPPI_MJX] contact_solref override → {sr} "
                  f"on {n_patched} rock-stick pair(s)")
        self.contact_solref = contact_solref

        # CLOCK: 1 horizon step must advance one CONTROL frame (~8.6ms) so H spans the whole
        # scrape — but the SIM must integrate at the native FINE step for stable contact. So
        # big step: reached the full rail but nan'd aggressive samples + softened contact.
        # stiff contact must use fine EULER substeps. EXACT substepping: sim step = ctrl/N so N
        # (N=4→2.16ms stable/4×; N=2→4.3ms ~2×, try it; N=1→8.6ms fast but nan-prone).
        self._sim_dt = float(model.opt.timestep)
        _ctrl_ov = __import__('os').environ.get('MPPI_DT')
        if _ctrl_ov is not None:
            _ctrl_dt = float(_ctrl_ov)
            _nsub_ov = __import__('os').environ.get('MPPI_NSUB')
            self._rollout_nsub = int(_nsub_ov) if _nsub_ov else max(1, int(round(_ctrl_dt / self._sim_dt)))
            model.opt.timestep = _ctrl_dt / self._rollout_nsub
            self._ctrl_dt_ov = _ctrl_dt
            print(f"[KinematicMPPI_MJX] clock: ctrl {_ctrl_dt*1000:.2f}ms = {self._rollout_nsub} × "
                  f"{model.opt.timestep*1000:.2f}ms Euler substeps ({'override' if _nsub_ov else 'auto'})")
        else:
            self._rollout_nsub = 1; self._ctrl_dt_ov = None
        self.mx = mjx.put_model(model)
        self.model = model
        self.H = horizon
        self.K = num_samples
        self.lam = float(lambda_)
        self.dt = float(self._ctrl_dt_ov) if self._ctrl_dt_ov is not None else float(model.opt.timestep)
        self.target_force = float(target_force)
        self.target_force_profile = None
        self.target_rail_vel = float(target_rail_vel)
        self.n_dial = int(n_dial)
        self.dial_decay = float(dial_decay)
        self._anneal_on = _mppi_anneal_enabled()
        self._anneal_sig = None
        self._anneal_lam = None
        if self._anneal_on:
            self.n_dial = int(os.environ.get('MPPI_ANNEAL_NDIAL', '5'))
            _sig_hi = float(os.environ.get('MPPI_ANNEAL_SIG_HI', '1.0'))
            _sig_lo = float(os.environ.get('MPPI_ANNEAL_SIG_LO', '0.04'))
            _lam_hi = float(os.environ.get('MPPI_ANNEAL_LAM_HI', str(self.lam * 6.0)))
            _lam_lo = float(os.environ.get('MPPI_ANNEAL_LAM_LO', str(self.lam)))
            _N = max(1, self.n_dial)
            def _geom(hi, lo, i, n):
                return float(lo) if n <= 1 else float(hi * (lo / hi) ** (i / (n - 1)))
            self._anneal_sig = [_geom(_sig_hi, _sig_lo, d, _N) for d in range(_N)]
            self._anneal_lam = [_geom(_lam_hi, _lam_lo, d, _N) for d in range(_N)]
            print(f"[mppi] MPPI_ANNEAL ON: n_dial={self.n_dial} "
                  f"sig {_sig_hi:g}->{_sig_lo:g}  lam {_lam_hi:g}->{_lam_lo:g}\n"
                  f"       sig_sched={[round(s, 3) for s in self._anneal_sig]}\n"
                  f"       lam_sched={[round(l, 3) for l in self._anneal_lam]}")
        self.spline_M = spline_M
        self._B_spline = None
        self._spline_gain = 1.0
        self.use_fixed_samples = False
        self._fixed_samples = None

        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv
        self.nr_run = len(KEYS_RUN)

        self.press_control = bool(press_control)
        self.virtual_target = bool(virtual_target)
        self.vt_kp = float(vt_kp); self.vt_kd = float(vt_kd)
        if self.virtual_target:
            self.press_control = True
            press_sigma = float(vt_dz_sigma)
            press_max   = float(vt_dz_max)
        self.press_sigma   = float(press_sigma)
        self.press_max     = float(press_max)
        self.press_ema     = float(press_ema)
        self.press_ref     = float(press_ref)
        self.rail_walk_pace = bool(rail_walk_pace)
        self.rail_walk_pace_sigma = float(rail_walk_pace_sigma)
        self.n_ctrl = self.nu + (1 if self.press_control else 0) + (1 if self.rail_walk_pace else 0)
        self._pace_idx = self.nu + (1 if self.press_control else 0)

        ctrl = human_mppi.controller
        self.arm_dofadr = np.asarray(ctrl.arm_dofadr, dtype=np.int32)
        self._vt_free_mask = (np.asarray(model.dof_armature)[self.arm_dofadr] < 1e6).astype(np.float64)
        self._vt_proj = None

        self.arm_qposadr = np.asarray(ctrl.arm_qposadr, dtype=np.int32)
        self._elbow_qposadr = self.arm_qposadr[ELBOW_LOCAL]
        self._elbow_ref = jnp.asarray(ELBOW_TRACK_REF, dtype=jnp.float32)
        _dpd = mujoco.MjData(model)
        if hasattr(human_mppi, 'mujoco_init_q'):
            _dpd.qpos[:] = np.asarray(human_mppi.mujoco_init_q)
        self._dump_init_qpos = _dpd.qpos.copy()
        self._dump_init_qvel = np.asarray(getattr(human_mppi, 'v0', np.zeros(model.nv)), dtype=float)
        mujoco.mj_forward(model, _dpd)
        _Mpd = np.zeros((model.nv, model.nv)); mujoco.mj_fullM(model, _Mpd, _dpd.qM)
        _Mdiag = np.diag(_Mpd)[self.arm_dofadr]
        _lim = np.abs(model.actuator_ctrlrange[:len(self.arm_dofadr), 1])
        _dt_pd = float(model.opt.timestep)
        self._pd_kp = np.minimum(_lim * 5.0, 0.5 * 4.0 * _Mdiag / _dt_pd**2).astype(np.float32)
        self._pd_kd = (self._pd_kp * 0.15).astype(np.float32)

        u_min = model.actuator_ctrlrange[:, 0].copy()
        u_max = model.actuator_ctrlrange[:, 1].copy()
        if np.all(u_min == 0) and np.all(u_max == 0):
            u_min = np.full(self.nu, -500.0)
            u_max = np.full(self.nu, 500.0)
        if self.press_control:
            u_min = np.concatenate([u_min, [-self.press_max]]).astype(u_min.dtype)
            u_max = np.concatenate([u_max, [ self.press_max]]).astype(u_max.dtype)
        if self.rail_walk_pace:
            _pace_max = float(getattr(self, 'rail_walk_pace_max', 1.0))
            u_min = np.concatenate([u_min, [-_pace_max]]).astype(u_min.dtype)
            u_max = np.concatenate([u_max, [ _pace_max]]).astype(u_max.dtype)
        self.u_min = u_min
        self.u_max = u_max

        lj = human_mppi.locked_joint_constraints
        self.locked_qpos_ids = np.array([c[0] for c in lj], dtype=np.int32) if lj else np.array([], dtype=np.int32)
        self.locked_dof_ids  = np.array([c[1] for c in lj], dtype=np.int32) if lj else np.array([], dtype=np.int32)
        self.locked_vals     = np.array([c[2] for c in lj], dtype=np.float32) if lj else np.array([], dtype=np.float32)

        self.has_stick = (not human_mppi.stick_static
                          and human_mppi.stick_traj is not None
                          and human_mppi._stick_qposadr is not None)
        self.stick_traj = (np.asarray(human_mppi.stick_traj, dtype=np.float32)
                           if self.has_stick else None)
        self.stick_qposadr = int(human_mppi._stick_qposadr) if self.has_stick else 0

        self.stick_body_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        self.rock_geom_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
        self.rock_body_id     = int(model.geom_bodyid[self.rock_geom_id]) if self.rock_geom_id >= 0 else -1
        self.contact_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        self.stick_half_len   = 0.4
        self.stick_radius     = 0.02
        self._stick_body_id   = self.stick_body_id
        self._rock_geom_id    = self.rock_geom_id
        self._rock_body_id    = self.rock_body_id
        self._contact_site_id = self.contact_site_id
        self._stick_half_len  = self.stick_half_len
        self._stick_radius    = self.stick_radius
        self._rock_quat_ref   = None

        p_start = np.asarray(human_mppi.p_start_world, dtype=np.float32)
        p_end   = np.asarray(human_mppi.p_end_world,   dtype=np.float32)
        rail_vec = p_end - p_start
        self.rail_len = float(np.linalg.norm(rail_vec))
        self.rail_dir = (rail_vec / max(self.rail_len, 1e-9)).astype(np.float32)
        self.p_start  = p_start

        self.rock_quat_ref = None
        rb_id_named = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        if rb_id_named >= 0:
            d_ref = mujoco.MjData(model)
            from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
            for i, name in enumerate(_AAJ):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and i < len(human_mppi.q0):
                    d_ref.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
            mujoco.mj_forward(model, d_ref)
            self.rock_quat_ref = d_ref.xquat[rb_id_named].copy().astype(np.float32)
            self._rock_quat_ref = self.rock_quat_ref

        self.noise_scale = float(noise_scale)
        self.force_control = bool(force_control)
        self.fc_kp = float(fc_kp)
        self.fc_kd = float(fc_kd)
        self.fc_sign = float(fc_sign)
        self.fc_kp_close = float(fc_kp_close)
        self.fc_sign_close = float(fc_sign_close)
        self.fc_kv = float(fc_kv)
        # to the setpoint, then flat (no incentive to exceed it). MUST be
        self.bound_progress_vel = bool(bound_progress_vel)
        self.stick_follow = bool(stick_follow)
        self.force_ema = float(force_ema)
        self.blend_steps = int(blend_steps)
        self._u_blend_from = None
        self._fc_prev_N = 0.0
        if self.force_control:
            model.opt.jacobian = int(mujoco.mjtJacobian.mjJAC_DENSE)
            print(f"[KinematicMPPI_MJX] force_control ON "
                  f"(fc_kp={self.fc_kp}, fc_kd={self.fc_kd}, "
                  f"fc_sign={self.fc_sign}); host efc Jacobian → DENSE")
            if self.fc_kp_close != 0.0:
                print(f"[KinematicMPPI_MJX] gap-closer ON "
                      f"(fc_kp_close={self.fc_kp_close}, "
                      f"fc_sign_close={self.fc_sign_close})")
        self.verbose_solve = True
        d_tmp = mujoco.MjData(model)
        d_tmp.qpos[:] = human_mppi.mj_data.qpos.copy()
        d_tmp.qvel[:] = 0.0
        mujoco.mj_forward(model, d_tmp)
        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d_tmp, J, None, self.contact_site_id)
        J_arm = J[:, self.arm_dofadr]
        Sigma = alpha * (J_arm.T @ J_arm) + beta * np.eye(self.nu)
        self.L = (np.linalg.cholesky(Sigma).astype(np.float32)
                  * np.float32(self.noise_scale))
        if abs(self.noise_scale - 1.0) > 1e-6:
            print(f"[KinematicMPPI_MJX] noise_scale={self.noise_scale:.2f} "
                  f"applied to Cholesky factor")
        if self.press_control or self.rail_walk_pace:
            L_aug = np.zeros((self.n_ctrl, self.n_ctrl), dtype=np.float32)
            L_aug[:self.nu, :self.nu] = self.L
            if self.press_control:
                L_aug[self.nu, self.nu] = np.float32(self.press_sigma)
            if self.rail_walk_pace:
                L_aug[self._pace_idx, self._pace_idx] = np.float32(self.rail_walk_pace_sigma)
            self.L = L_aug
            print(f"[KinematicMPPI_MJX] control dim {self.nu}"
                  f"{'+press' if self.press_control else ''}{'+pace' if self.rail_walk_pace else ''}"
                  f" (press_sigma={self.press_sigma}, pace_sigma={self.rail_walk_pace_sigma})")
        if getattr(self, 'task_space', False):
            _nn = int(getattr(self, 'ts_n_null', 2))
            _Lts = np.zeros((self.n_ctrl, self.n_ctrl), dtype=np.float32)
            _Lts[0, 0] = np.float32(getattr(self, 'ts_pace_sigma', 1.0))
            for _j in range(_nn):
                _Lts[1 + _j, 1 + _j] = np.float32(getattr(self, 'ts_null_sigma', 0.05))
            if self.press_control:
                _Lts[self.nu, self.nu] = np.float32(self.press_sigma)
            self.L = _Lts
            print(f"[KinematicMPPI_MJX] TASK-SPACE control: slot0=pace(σ={getattr(self,'ts_pace_sigma',1.0)}) "
                  f"slots1..{_nn}=null(σ={getattr(self,'ts_null_sigma',0.05)}) press(σ={self.press_sigma})")

        self._jnt_qposadr = model.jnt_qposadr.copy().astype(np.int32)
        self._jnt_limited = model.jnt_limited.astype(bool)
        self._jnt_range   = model.jnt_range.copy().astype(np.float32)
        _nq = int(model.nq)
        _qlo = np.full(_nq, -1e9, np.float32); _qhi = np.full(_nq, 1e9, np.float32)
        _qlm = np.zeros(_nq, np.float32)
        _HINGE = int(mujoco.mjtJoint.mjJNT_HINGE); _SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for _j in range(model.njnt):
            if bool(model.jnt_limited[_j]) and int(model.jnt_type[_j]) in (_HINGE, _SLIDE):
                _a = int(model.jnt_qposadr[_j])
                _qlo[_a] = model.jnt_range[_j, 0]; _qhi[_a] = model.jnt_range[_j, 1]; _qlm[_a] = 1.0
        self._q_lim_lo = _qlo; self._q_lim_hi = _qhi; self._q_lim_mask = _qlm
        self.jnt_limit_w = 0.0

        self.w_run = {k: 0.0 for k in KEYS_RUN}
        self._phi_scale = np.ones(self.nr_run, dtype=np.float32)
        self._phi_norm = np.ones(self.nr_run, dtype=np.float32)
        self._phi_norm_on = _mppi_phi_norm_enabled()
        self._w_fixed   = np.zeros(self.nr_run, dtype=np.float32)
        self._w_run_windows = None
        self.U = np.zeros((self.H, self.n_ctrl), dtype=np.float32)

        self.current_t = 0
        self._rng_key  = jax.random.PRNGKey(42)

        self._jit_step = None

        self._eng_groups_jax = {
            g: jnp.asarray([i for i in gidx if i < len(self.arm_dofadr)],
                            dtype=jnp.int32)
            for g, gidx in ENG_GROUPS.items()
        }

        print(f"[KinematicMPPI_MJX] K={self.K}, H={self.H}, n_dial={self.n_dial}")

    def set_weights(self, w_run):
        if w_run is not None:
            self.w_run.update(w_run)

    def set_phi_norm(self, phi_norm):
        """Store the per-feature σ used to σ-normalize the softmax GENERATION
        cost (MPPI_PHI_NORM). ``phi_norm`` is either

          * a ``{feature_key: σ}`` dict — mapped BY KEY into KEYS_RUN order
            (NOT by position: the IRL feature order ``sorted(keys)`` differs
            from KEYS_RUN, so a positional copy would silently mis-scale every
            feature), or
          * an array already in KEYS_RUN order.

        Non-positive / non-finite σ are clamped to 1.0 (no scaling for that
        feature). Effective only when MPPI_PHI_NORM is set; otherwise stored
        but never consulted (``_softmax_weights`` short-circuits)."""
        if isinstance(phi_norm, dict):
            arr = np.array([float(phi_norm.get(k, 1.0)) for k in KEYS_RUN],
                           dtype=np.float32)
        else:
            arr = np.asarray(phi_norm, dtype=np.float32).reshape(-1)[:self.nr_run]
        arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 1.0).astype(np.float32)
        self._phi_norm = arr

    def _softmax_weights(self, w_arr_np):
        """σ-normalize a host-side softmax weight array of shape (..., nr_run)
        by dividing each feature weight by its σ, turning the generation cost
        ``Σ w·φ`` into ``Σ (w/σ)·φ = Σ w·(φ/σ)``. Applied to the *weights*
        (a runtime input to the jitted rollout) rather than the jitted feature
        computation, so there is NO JIT rebuild. Returns the input unchanged
        when MPPI_PHI_NORM is off ⇒ byte-identical."""
        if self._phi_norm_on:
            return (np.asarray(w_arr_np, dtype=np.float32)
                    / self._phi_norm).astype(np.float32)
        return w_arr_np

    def set_force_target_profile(self, profile):
        self.target_force_profile = (
            None if profile is None
            else np.asarray(profile, dtype=np.float32)
        )

    def reset(self):
        _ws = getattr(self, '_U_warmstart', None)
        self.U = (_ws.copy() if _ws is not None
                  else np.zeros((self.H, self.n_ctrl), dtype=np.float32))
        self.current_t = 0
        self._rng_key  = self.jax.random.PRNGKey(42)

    @staticmethod
    def _cubic_bspline_basis(H, M, degree=3):
        """Clamped-uniform cubic B-spline basis, (H, M) numpy. Each column is
        one basis function at H uniform time points; rows sum to 1 (partition
        of unity) so control magnitude is preserved. U = B @ CP maps M control
        points to an H-step control trajectory, smooth by construction."""
        from scipy.interpolate import BSpline
        n_interior = M - degree - 1
        interior = (np.linspace(0, 1, n_interior + 2)[1:-1]
                    if n_interior > 0 else np.array([]))
        knots = np.concatenate([np.zeros(degree + 1), interior,
                                np.ones(degree + 1)])
        t_eval = np.linspace(0.0, 1.0, H)
        B = np.zeros((H, M))
        for j in range(M):
            c = np.zeros(M); c[j] = 1.0
            B[:, j] = BSpline(knots, c, degree, extrapolate=False)(t_eval)
        B = np.nan_to_num(B, nan=0.0)
        rs = B.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        return B / rs

    @staticmethod
    def _colored_knot_matrix(M, beta):
        """(M×M) coloring matrix A for z_colored = A·z: imposes a 1/f^β spectrum ACROSS the
        M spline control points, so the SAMPLED control path is temporally smooth (fixes the
        residual jitter the spline basis alone can't). Built as Φᵀ·diag(f^{-β/2})·Φ with Φ the
        orthonormal DCT-II basis → Cov(z_colored)=Φᵀ·diag(f^{-β})·Φ. Overall power rescaled so
        the AVERAGE marginal variance = 1 (exploration magnitude / noise_scale unchanged; the
        1/f^β SHAPE is preserved). beta=0 → identity (byte-identical); larger β → smoother."""
        M = int(M)
        k = np.arange(M); n = np.arange(M)
        Phi = np.sqrt(2.0 / M) * np.cos(np.pi * (n[None, :] + 0.5) * k[:, None] / M)
        Phi[0, :] *= 1.0 / np.sqrt(2.0)
        f = np.maximum(k.astype(np.float64), 0.5)
        amp = f ** (-0.5 * float(beta))
        A = Phi.T @ (amp[:, None] * Phi)
        A *= np.sqrt(M / max(float(np.trace(A @ A.T)), 1e-12))
        return A.astype(np.float32)

    def _build_jit(self):
        jnp = self.jnp
        jax = self.jax
        mjx = self.mjx
        mx  = self.mx

        arm        = jnp.asarray(self.arm_dofadr)
        u_min      = jnp.asarray(self.u_min)
        u_max      = jnp.asarray(self.u_max)
        locked_q   = jnp.asarray(self.locked_qpos_ids)
        locked_dof = jnp.asarray(self.locked_dof_ids)
        locked_vals = jnp.asarray(self.locked_vals)
        _ts_pin_qpos = jnp.asarray(getattr(self, 'ts_pin_qpos', np.array([], dtype=np.int32)))
        _ts_pin_dof  = jnp.asarray(getattr(self, 'ts_pin_dof',  np.array([], dtype=np.int32)))
        _ts_pin_vals = jnp.asarray(getattr(self, 'ts_pin_vals', np.array([], dtype=np.float32)))
        _ts_pin_n_py = int(np.asarray(getattr(self, 'ts_pin_qpos', np.array([], dtype=np.int32))).shape[0])
        geo_keep_mask = jnp.asarray(np.asarray(self.mx.dof_armature) < 1e6)
        rail_dir   = jnp.asarray(self.rail_dir)
        p_start    = jnp.asarray(self.p_start)
        rail_len   = jnp.float32(self.rail_len)
        stick_radius = jnp.float32(self.stick_radius)
        stick_half_len = jnp.float32(self.stick_half_len)
        rock_quat_ref = (jnp.asarray(self.rock_quat_ref)
                          if self.rock_quat_ref is not None
                          else jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
        rock_body_id_named = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        stick_body_id = self.stick_body_id
        rock_geom_id  = self.rock_geom_id
        rock_body_geom = self.rock_body_id
        #   The geometric gap-closer must Jacobian the ROCK body, not the
        site_id       = self.contact_site_id
        eng_groups    = self._eng_groups_jax
        dt            = jnp.float32(self.dt)
        has_stick     = self.has_stick
        stick_qposadr = self.stick_qposadr
        stick_follow  = bool(self.stick_follow)
        if has_stick and self.stick_traj is not None:
            stick_traj_j = jnp.asarray(self.stick_traj, dtype=jnp.float32)
            T_stick      = int(self.stick_traj.shape[0])
        else:
            stick_traj_j = jnp.zeros((1, 3), dtype=jnp.float32)
            T_stick      = 1
        _trav_ref = getattr(self, 'traveled_ref', None)
        if _trav_ref is not None and np.asarray(_trav_ref).size > 1:
            traveled_ref_j = jnp.asarray(np.asarray(_trav_ref, dtype=np.float32))
            T_trav         = int(traveled_ref_j.shape[0])
            use_trav_track = True
        else:
            traveled_ref_j = jnp.zeros((1,), dtype=jnp.float32)
            T_trav         = 1
            use_trav_track = False
        force_control = bool(self.force_control)
        press_control = bool(self.press_control)
        virtual_target = bool(self.virtual_target)
        vt_kp_j       = jnp.float32(self.vt_kp)
        vt_kd_j       = jnp.float32(self.vt_kd)
        # tangential projection out of the locked root (else it blows up, as on the host).
        vt_free_mask  = jnp.asarray(
            (np.asarray(self._mj_model.dof_armature)[self.arm_dofadr] < 1e6).astype(np.float32))
        jlim_w   = float(getattr(self, 'jnt_limit_w', 0.0))
        use_jlim = jlim_w > 0.0
        q_lim_lo   = jnp.asarray(self._q_lim_lo)
        q_lim_hi   = jnp.asarray(self._q_lim_hi)
        q_lim_mask = jnp.asarray(self._q_lim_mask)
        press_ema_a   = jnp.float32(self.press_ema)
        press_ref_on  = bool(self.press_ref > 0)
        press_ref_j   = jnp.float32(self.press_ref if self.press_ref > 0 else 1.0)
        press_emergent = bool(getattr(self, 'press_emergent', False))
        confine_motion = bool(getattr(self, 'confine_motion', False))
        depth_impedance = bool(getattr(self, 'depth_impedance', False))
        depth_kn = jnp.float32(getattr(self, 'depth_kn', 3000.0))
        depth_target = jnp.float32(getattr(self, 'depth_target', -0.0015))
        contact_latch = bool(getattr(self, 'contact_latch', False))
        latch_project = bool(getattr(self, 'latch_project', False))
        latch_k = jnp.float32(getattr(self, 'latch_k', 4000.0))
        latch_gap = jnp.float32(getattr(self, 'latch_gap', 0.001))
        latch_vrail = jnp.float32(getattr(self, 'latch_vrail', self.target_rail_vel))
        latch_rail_kd = jnp.float32(getattr(self, 'latch_rail_kd', 40.0))
        latch_rail_ff = jnp.float32(getattr(self, 'latch_rail_ff', 0.0))
        # with the reference projected tangential ALWAYS so the PD never fights
        force_press = bool(getattr(self, 'force_press', False))
        press_force_des = jnp.float32(getattr(self, 'press_force_des', 26.0))
        task_space_control = bool(getattr(self, 'task_space_control', False))
        ts_kp = jnp.float32(getattr(self, 'ts_kp', 600.0))
        ts_kd = jnp.float32(getattr(self, 'ts_kd', 60.0))
        ts_fdes = jnp.float32(getattr(self, 'ts_fdes', 30.0))
        ts_pstart = jnp.asarray(getattr(self, 'p_start', np.zeros(3)), dtype=jnp.float32)
        ts_railn = jnp.float32(max(float(getattr(self, 'rail_len', 0.26)), 1e-6))
        ts_Hm1 = jnp.float32(max(int(self.H) - 1, 1))
        _fp = np.asarray(getattr(self, 'ts_fdes_profile', [float(getattr(self, 'ts_fdes', 30.0))]),
                         dtype=np.float32)
        ts_fdes_profile = jnp.asarray(_fp, dtype=jnp.float32)
        ts_prof_len = jnp.int32(len(_fp))
        ts_vrail = jnp.float32(getattr(self, 'ts_vrail', 0.46))
        ts_kd_rail = jnp.float32(getattr(self, 'ts_kd_rail', 40.0))
        ts_vrail_ramp = jnp.float32(getattr(self, 'ts_vrail_ramp', 0.0))
        ts_dtctrl = jnp.float32(getattr(self, 'dt', 0.0086))
        rollout_nsub = int(getattr(self, '_rollout_nsub', 1))
        ts_fdes_floor = jnp.float32(getattr(self, 'ts_fdes_floor', 15.0))
        _ts_post_kp_py = float(getattr(self, 'ts_post_kp', 0.0))
        ts_post_kp = jnp.float32(_ts_post_kp_py)
        ts_post_kd = jnp.float32(getattr(self, 'ts_post_kd', 6.0))
        ts_impedance_normal = bool(getattr(self, 'ts_impedance_normal', False))
        ts_pen_kp   = jnp.float32(getattr(self, 'ts_pen_kp', 8000.0))
        ts_pen_kd   = jnp.float32(getattr(self, 'ts_pen_kd', 120.0))
        ts_pen_des  = jnp.float32(getattr(self, 'ts_pen_des', 0.003))
        ts_pen_scale= jnp.float32(getattr(self, 'ts_pen_scale', 0.0003))
        ts_pen_max  = jnp.float32(getattr(self, 'ts_pen_max', 0.02))
        ts_fnmax    = jnp.float32(getattr(self, 'ts_fnmax', 200.0))
        ts_press_depth = jnp.float32(getattr(self, 'ts_press_depth', 0.0))
        _ts_press_ct_on = bool(float(getattr(self, 'ts_press_depth', 0.0)) > 0.0)
        # cap |damping| to frac·|spring| so it still kills slam/eject but can NEVER cancel the press.
        _ts_pen_damp_frac_py = float(getattr(self, 'ts_pen_damp_frac', 0.0))
        # the singular config → mjx.step factors a singular qM → the nan. rail_walk
        # instead of blowing up → kills the nan). Same wall, both fixes in one block.
        rail_walk   = bool(getattr(self, 'rail_walk', False))
        rw_lambda   = jnp.float32(getattr(self, 'rail_walk_lambda', 0.05))
        rw_wlim     = jnp.float32(getattr(self, 'rail_walk_wlim', 4.0))
        rw_margin   = jnp.float32(getattr(self, 'rail_walk_margin', 0.15))
        q_lim_lo_arm = jnp.asarray(self._q_lim_lo[self.arm_qposadr])
        q_lim_hi_arm = jnp.asarray(self._q_lim_hi[self.arm_qposadr])
        rw_pace     = bool(self.rail_walk_pace)
        pace_idx    = int(self._pace_idx)
        _rw_post_kp_py = float(getattr(self, 'rail_walk_post_kp', 0.0))
        rw_post_kp  = jnp.float32(_rw_post_kp_py)
        rw_post_kd  = jnp.float32(getattr(self, 'rail_walk_post_kd', 4.0))
        rail_force  = bool(getattr(self, 'rail_force', False))
        rf_kv       = jnp.float32(getattr(self, 'rf_kv', 400.0))
        rf_fmax     = jnp.float32(getattr(self, 'rf_fmax', 120.0))
        rf_kpn      = jnp.float32(getattr(self, 'rf_kpn', 8000.0))
        rf_kdn      = jnp.float32(getattr(self, 'rf_kdn', 120.0))
        rf_fnmax    = jnp.float32(getattr(self, 'rf_fnmax', 200.0))
        rf_near     = jnp.float32(getattr(self, 'rf_near', 0.015))
        rf_fapp     = jnp.float32(getattr(self, 'rf_fapp', 25.0))
        rf_kplat    = jnp.float32(getattr(self, 'rf_kplat', 3000.0))
        rf_kdlat    = jnp.float32(getattr(self, 'rf_kdlat', 40.0))
        rf_pen      = jnp.float32(getattr(self, 'rf_pen', 0.003))
        rf_vbase    = jnp.float32(getattr(self, 'rf_vbase', float(self.target_rail_vel)))
        task_space  = bool(getattr(self, 'task_space', False))
        ts_discover  = bool(getattr(self, 'ts_discover', False)) and task_space
        ts_reanchor  = bool(getattr(self, 'ts_reanchor', False)) and task_space
        ts_reanchor_exact = bool(getattr(self, 'ts_reanchor_exact', False)) and task_space
        ts_null_runtime = bool(getattr(self, 'ts_null_runtime', False)) and task_space
        ts_reanchor_iters = int(getattr(self, 'ts_reanchor_iters', 2))
        ts_n_app_f   = jnp.float32(getattr(self, 'ts_n_app', 8.0))
        _nn0         = int(getattr(self, 'ts_n_null', 2))
        ts_s0_idx    = 1 + _nn0
        ts_lift_idx  = 2 + _nn0
        ts_s0_scale  = jnp.float32(getattr(self, 'ts_s0_scale', 0.03))
        ts_s0_max    = jnp.float32(getattr(self, 'ts_s0_max', 0.06))
        ts_lift_scale= jnp.float32(getattr(self, 'ts_lift_scale', 0.01))
        ts_lift_max  = jnp.float32(getattr(self, 'ts_lift_max', 0.04))
        if task_space:
            ts_q_of_s    = jnp.asarray(self.ts_q_of_s)
            ts_null_of_s = jnp.asarray(self.ts_null_of_s)
            ts_grid      = jnp.asarray(self.ts_s_grid)
            ts_s_nom     = jnp.asarray(self.ts_s_nom)
            ts_sched_len = int(self.ts_s_nom.shape[0])
            ts_n_null    = int(self.ts_null_of_s.shape[1])
            ts_narm      = int(self.ts_q_of_s.shape[1])
            ts_pace_scale= jnp.float32(getattr(self, 'ts_pace_scale', 0.03))
            ts_pace_fwd  = bool(getattr(self, 'ts_pace_fwd', False))   # FORWARD-ONLY sweep: clamp the sampled pace ≥0 so s never retreats (the sweep is one-directional; backward samples are wasted + a glitchy warmstart can execute one)
            ts_qref_dev_ema = jnp.float32(getattr(self, 'ts_qref_ema', 1.0))
            _ts_qref_warm = bool(getattr(self, 'ts_qref_warm', False))
            _ts_dev_ema_on = bool(float(getattr(self, 'ts_qref_ema', 1.0)) < 1.0) and not _ts_qref_warm
            ts_smin      = jnp.float32(float(np.min(self.ts_s_grid)))
            ts_smax      = jnp.float32(float(np.max(self.ts_s_grid)))
            ts_KP        = jnp.asarray(self.ts_KP, dtype=jnp.float32)
            ts_KD        = jnp.asarray(self.ts_KD, dtype=jnp.float32)
            ts_null_kp_scale = jnp.float32(getattr(self, 'ts_null_kp_scale', 1.0))
            ts_null_kp   = jnp.asarray(getattr(self, 'ts_null_kp', 0.0), dtype=jnp.float32)
            _ts_null_kp_on = bool(np.any(np.asarray(getattr(self, 'ts_null_kp', 0.0), dtype=float) > 0.0))
            ts_task_pd  = bool(getattr(self, 'ts_task_pd', False))
            ts_kp_lat   = jnp.float32(getattr(self, 'ts_kp_lat', 2500.0))
            ts_kd_lat   = jnp.float32(getattr(self, 'ts_kd_lat', 60.0))
            ts_kd_railf = jnp.float32(getattr(self, "ts_kd_railf", 40.0))
            ts_ccancel   = jnp.float32(getattr(self, 'ts_ccancel', 1.0))
            ts_ff_frac   = jnp.float32(getattr(self, 'ts_ff_frac', 0.5))
            ts_mu_ff     = jnp.float32(getattr(self, 'ts_mu_ff', 0.25))
            # DOFs (the masked GN below). Breaks the null-channel reach cap that trapped the wrist at
            ts_posture_joint = bool(getattr(self, 'ts_posture_joint', False))
            ts_posture_free  = bool(getattr(self, 'ts_posture_free', False))
            ts_posture_sample = bool(getattr(self, 'ts_posture_sample', False))
            _ts_proj_winv = jnp.asarray(np.asarray(getattr(self, 'ts_proj_winv', np.ones(ts_narm)), dtype=np.float32))
            ts_joint_gain    = jnp.float32(getattr(self, 'ts_joint_gain', 0.11))
            _jd_ts = [int(x) for x in getattr(self, 'ts_joint_dofs', [7])]
            ts_n_joint     = int(len(_jd_ts))
            _ts_joint_rows = jnp.asarray(np.eye(ts_narm, dtype=np.float32)[_jd_ts])
            _ts_joint_mask = jnp.asarray(np.isin(np.arange(ts_narm), _jd_ts).astype(np.float32))
            _ts_joint_center = jnp.asarray(np.asarray(getattr(self, 'ts_joint_center', np.zeros(ts_narm)), dtype=np.float32))
            _ts_nullflat = ts_null_of_s.reshape(ts_null_of_s.shape[0], -1)
            ts_lift_of_s = jnp.asarray(getattr(self, 'ts_lift_of_s', self.ts_q_of_s * 0.0))
            if ts_reanchor or ts_reanchor_exact:
                _ts_reJpTflat = jnp.asarray(self.ts_reanchor_JpT).reshape(ts_q_of_s.shape[0], -1)
            _ts_ptab_np = getattr(self, 'ts_p_of_s', None)
            ts_p_table = (_ts_ptab_np is not None
                          and os.environ.get('TS_P_TABLE', '0').strip().lower()
                          not in ('0', '', 'false', 'off', 'no'))
            if ts_p_table:
                ts_p_of_s = jnp.asarray(np.asarray(_ts_ptab_np, dtype=np.float32))
                print("[KinematicMPPI_MJX] TS_P_TABLE ON — GN reanchor target from the "
                      "p_of_s table (saves 1 FK pass per rollout step)")
            if ts_reanchor:
                _ts_reBflat   = jnp.asarray(self.ts_reanchor_B).reshape(ts_q_of_s.shape[0], -1)
            if ts_reanchor_exact or ts_posture_joint or ts_posture_sample:
                from mujoco.mjx._src import smooth as _ts_smooth_mod
                from mujoco.mjx._src import support as _ts_support_mod
                _ts_kin_fn    = getattr(mjx, 'kinematics', None) or _ts_smooth_mod.kinematics
                _ts_compos_fn = getattr(mjx, 'com_pos', None)    or _ts_smooth_mod.com_pos
                print(f"[KinematicMPPI_MJX] runtime Gauss-Newton tool re-anchor "
                      f"({ts_reanchor_iters} FK iters/step, J⁺ RECOMPUTED at q_ref via mjx.jac)"
                      + (" — JOINT-SPACE posture (led DOFs protected)" if ts_posture_joint else " — kills ALL-order drift, wide σ stays on-rail"))
            def _ts_interp(table, s):
                i = jnp.clip(jnp.searchsorted(ts_grid, s), 1, ts_grid.shape[0] - 1)
                g0 = ts_grid[i - 1]; g1 = ts_grid[i]
                w = jnp.clip((s - g0) / jnp.maximum(g1 - g0, 1e-9), 0.0, 1.0)
                return (1.0 - w) * table[i - 1] + w * table[i]
        pd_track = bool(getattr(self, 'pd_track', False))
        pd_free_normal = bool(getattr(self, 'pd_free_normal', False))
        pd_feedforward = bool(getattr(self, 'pd_feedforward', False))
        pd_kp_j  = jnp.asarray(self._pd_kp)
        pd_kd_j  = jnp.asarray(self._pd_kd)
        _ws_ff = getattr(self, '_U_warmstart', None)
        if getattr(self, 'task_space', False):
            _snom = np.asarray(self.ts_s_nom); _qg = np.asarray(self.ts_q_of_s); _sg = np.asarray(self.ts_s_grid)
            _qnom = np.stack([np.interp(_snom, _sg, _qg[:, c]) for c in range(_qg.shape[1])], axis=1)
            _sm = _qnom.copy()
            for _ in range(int(getattr(self, 'ff_smooth_passes', 8))):
                _sm[1:-1] = 0.25*_sm[:-2] + 0.5*_sm[1:-1] + 0.25*_sm[2:]
            _qddf = np.zeros_like(_qnom); _qddf[1:-1] = (_sm[2:] - 2.0*_sm[1:-1] + _sm[:-2]) / (self.dt**2)
            _qddf = _qddf.astype(np.float32)
            self._qddot_ff = _qddf; qddot_ff = jnp.asarray(_qddf, dtype=jnp.float32)
            _qdf = np.gradient(_sm, self.dt, axis=0).astype(np.float32)
            self._qdot_ff = _qdf; qdot_ff = jnp.asarray(_qdf, dtype=jnp.float32)
        elif pd_feedforward and _ws_ff is not None:
            _wsu = np.asarray(_ws_ff)[:, :self.nu]
            _sm = _wsu.copy()
            for _ in range(int(getattr(self, 'ff_smooth_passes', 8))):
                _sm[1:-1] = 0.25 * _sm[:-2] + 0.5 * _sm[1:-1] + 0.25 * _sm[2:]
            _qdd_np = np.zeros_like(_wsu)
            _qdd_np[1:-1] = (_sm[2:] - 2.0 * _sm[1:-1] + _sm[:-2]) / (self.dt ** 2)
            self._qddot_ff = _qdd_np
            qddot_ff = jnp.asarray(_qdd_np, dtype=jnp.float32)
            qdot_ff = jnp.zeros((self.H, self.nu), dtype=jnp.float32)
        else:
            self._qddot_ff = np.zeros((self.H, self.nu), dtype=np.float32)
            qddot_ff = jnp.zeros((self.H, self.nu), dtype=jnp.float32)
            qdot_ff = jnp.zeros((self.H, self.nu), dtype=jnp.float32)
        id_feedforward = bool(getattr(self, 'id_feedforward', False))
        _u_ff_id = getattr(self, '_u_feedforward_id', None)
        if id_feedforward and _u_ff_id is not None:
            u_ff_id = jnp.asarray(np.asarray(_u_ff_id)[:, :self.nu], dtype=jnp.float32)
        else:
            u_ff_id = jnp.zeros((self.H, self.nu), dtype=jnp.float32)
        arm_qpos = jnp.asarray(self.arm_qposadr)
        force_max_j    = jnp.float32(max(float(getattr(self, 'force_max', 80.0)), 1.0))
        nu_j          = int(self.nu)
        ema_alpha     = jnp.float32(self.force_ema)
        ema_on        = bool(self.force_ema < 1.0)
        fc_kp         = jnp.float32(self.fc_kp)
        fc_kd         = jnp.float32(self.fc_kd)
        fc_sign       = jnp.float32(self.fc_sign)
        fc_kp_close   = jnp.float32(self.fc_kp_close)
        fc_sign_close = jnp.float32(self.fc_sign_close)
        gap_close_max = jnp.float32(getattr(self, 'gap_close_max', 1.0))
        use_gap_closer = bool(self.fc_kp_close != 0.0)
        # travel to 200%+; target_rail_vel is stored but never entered the
        fc_kv         = jnp.float32(self.fc_kv)
        v_rail_target = jnp.float32(self.target_rail_vel)
        use_rail_brake = bool(self.fc_kv != 0.0)
        bound_pv = bool(self.bound_progress_vel)
        pv_cap   = jnp.float32(self.target_rail_vel)
        prog_vel_sq = bool(getattr(self, 'progress_vel_sq', False))
        prog_vel_track = bool(getattr(self, 'progress_vel_track', False))
        use_rock_jac   = bool(use_gap_closer or use_rail_brake or press_control or task_space_control or rail_walk or rail_force or task_space)
        # the raw jittery u still drives the dynamics. MUST be mirrored on the
        tau_feat_ema = jnp.float32(getattr(self, 'tau_feat_ema', 1.0))
        vel_feat_ema = jnp.float32(getattr(self, 'vel_feat_ema', 1.0))
        from mujoco.mjx._src import support as _mjx_support

        use_spline = self.spline_M is not None
        if use_spline:
            if self._B_spline is None:
                self._B_spline = self._cubic_bspline_basis(self.H, self.spline_M)
                self._spline_gain = 1.0 / max(
                    float(np.sqrt((self._B_spline ** 2).sum(axis=1).mean())), 1e-6)
            B_spline = jnp.asarray(self._B_spline, dtype=jnp.float32)
            spline_gain = jnp.float32(self._spline_gain)
            spline_M_val = int(self.spline_M)
            print(f"[KinematicMPPI_MJX] SPLINE mode: M={spline_M_val} control "
                  f"points (dim {spline_M_val}×{self.nu} vs discrete "
                  f"{self.H}×{self.nu}), gain={self._spline_gain:.3f}")
        else:
            B_spline = None
            spline_gain = jnp.float32(1.0)
            spline_M_val = 0

        _cbeta = float(getattr(self, 'color_beta', 0.0))
        use_color = use_spline and (_cbeta > 0.0)
        if use_color:
            color_A = jnp.asarray(self._colored_knot_matrix(spline_M_val, _cbeta), dtype=jnp.float32)
            print(f"[KinematicMPPI_MJX] COLORED noise ON: 1/f^{_cbeta:g} across "
                  f"{spline_M_val} knots (smooth control path)")
        else:
            color_A = None

        use_fixed = self.use_fixed_samples
        if use_fixed:
            noise_dim = spline_M_val if use_spline else self.H
            if self._fixed_samples is None or self._fixed_samples.shape != (self.K, noise_dim, self.n_ctrl):
                rng_fixed = np.random.default_rng(0)
                raw = rng_fixed.standard_normal((self.K, noise_dim, self.n_ctrl)).astype(np.float32)
                norms = np.linalg.norm(raw.reshape(self.K, -1), axis=1, keepdims=True)
                raw_norm = (raw.reshape(self.K, -1) / np.maximum(norms, 1e-8)).reshape(self.K, noise_dim, self.n_ctrl)
                raw_norm *= float(np.sqrt(noise_dim * self.n_ctrl))
                _cs = int(getattr(self, 'ctrl_smooth', 0))
                if _cs > 1 and not use_spline:
                    _k = np.ones(_cs, dtype=np.float32) / _cs
                    for _a in range(self.K):
                        for _c in range(self.n_ctrl):
                            raw_norm[_a, :, _c] = np.convolve(raw_norm[_a, :, _c], _k, mode='same')
                    raw_norm *= float(np.sqrt(_cs))
                    print(f"[KinematicMPPI_MJX] ctrl_smooth={_cs}: low-pass noise filter ON "
                          f"(toy _maybe_smooth) → smooth control (deterministic MJX)")
                self._fixed_samples = raw_norm
                print(f"[KinematicMPPI_MJX] FIXED samples: shape={self._fixed_samples.shape}  "
                      f"std={self._fixed_samples.std():.3f} (target≈1.0)")
            fixed_z = jnp.asarray(self._fixed_samples, dtype=jnp.float32)
            print(f"[KinematicMPPI_MJX] Deterministic sample mode ON "
                  f"(noise_dim={noise_dim}, K={self.K})")
        else:
            fixed_z = None

        def _running_phi(dx, u, target_f_h, prev_f_ema, prev_v_arm, dz_press=0.0):
            """Per-step φ vector — mirrors _compute_phi_fast formulas.

            Returns (phi (nr_run,) matching KEYS_RUN order, v_feat, f_ema).
            """
            # → v_feat=raw (byte-identical). MUST match the demo (human_base).
            v_arm = vel_feat_ema * dx.qvel[arm] + (1.0 - vel_feat_ema) * prev_v_arm

            # Demo MUST mirror these exactly — see
            phi_tau = jnp.linalg.norm(u) / 50.0
            phi_jv  = jnp.linalg.norm(v_arm)
            phi_ja  = jnp.float32(0.0)
            phi_jtc = jnp.float32(0.0)

            u_arm = u[:v_arm.shape[0]]
            power = v_arm * u_arm
            phi_eng = []
            phi_tau_g = []
            phi_jv_g = []
            for g in ['thoracic', 'clavicle', 'shoulder', 'elbow', 'wrist']:
                gidx = eng_groups[g]
                if gidx.size > 0:
                    phi_eng.append(jnp.linalg.norm(power[gidx]) / 20.0)
                    phi_tau_g.append(jnp.linalg.norm(u_arm[gidx]) / 50.0)
                    phi_jv_g.append(jnp.linalg.norm(v_arm[gidx]))
                else:
                    phi_eng.append(jnp.float32(0.0))
                    phi_tau_g.append(jnp.float32(0.0))
                    phi_jv_g.append(jnp.float32(0.0))

            q_elb = dx.qpos[self._elbow_qposadr]
            phi_elbtrack = jnp.sum((q_elb - self._elbow_ref) ** 2)

            qM = dx._impl.qM
            v_geo = jnp.where(geo_keep_mask, dx.qvel, 0.0)
            phi_geo = jnp.sqrt(jnp.maximum(0.0, v_geo @ qM @ v_geo))

            if virtual_target and rock_geom_id >= 0 and stick_body_id >= 0:
                # (human_mppi get_tau). We do NOT read dx.contact.dist / efc_force —
                _rp_f = dx.geom_xpos[rock_geom_id]; _sc_f = dx.xpos[stick_body_id]
                _sa_f = dx.xmat[stick_body_id].reshape(3, 3)[:, 2]
                _tp_f = jnp.clip(jnp.dot(_rp_f - _sc_f, _sa_f), -stick_half_len, stick_half_len)
                _gap_f = jnp.linalg.norm(_rp_f - (_sc_f + _tp_f * _sa_f)) - stick_radius - 0.02
                f_contact = jnp.where(_gap_f < 0.05,
                                      jnp.maximum(vt_kp_j * (_gap_f + dz_press), 0.0), 0.0)
            else:
                # nan/undefined read. Clamp the index and nan_to_num the read so
                efc_addr = jnp.clip(dx._impl.contact.efc_address[0],
                                    0, dx._impl.efc_force.shape[0] - 1)
                cdist    = dx._impl.contact.dist[0]
                f_contact = jnp.where(cdist < 0,
                                       jnp.abs(jnp.nan_to_num(dx._impl.efc_force[efc_addr])), 0.0)
            f_ema = ema_alpha * f_contact + (1.0 - ema_alpha) * prev_f_ema
            tgt = jnp.abs(target_f_h)
            # original, which blows up at low-target steps and washes out the
            if press_emergent:
                phi_pf_raw = jnp.abs(f_ema) / force_max_j
            else:
                pf_denom = press_ref_j if press_ref_on else jnp.maximum(tgt, 1.0)
                phi_pf_raw = jnp.abs(f_ema - tgt) / pf_denom
            phi_pf = jnp.where(target_f_h < 0, 0.0, phi_pf_raw)
            # Force capacity |f_n - Fmax|/Fmax — same contact gate. MUST MATCH the
            _f_clip = jnp.clip(f_ema, -2.0 * force_max_j, 2.0 * force_max_j)
            phi_cap_raw = (_f_clip - force_max_j) ** 2 / (force_max_j ** 2)
            phi_cap = jnp.where(target_f_h < 0, 0.0, phi_cap_raw)

            if rock_geom_id >= 0 and stick_body_id >= 0:
                rp = dx.geom_xpos[rock_geom_id]
                sc = dx.xpos[stick_body_id]
                sa = dx.xmat[stick_body_id].reshape(3, 3)[:, 2]
                tp = jnp.clip(jnp.dot(rp - sc, sa), -stick_half_len, stick_half_len)
                nearest = sc + tp * sa
                raw_dist = jnp.linalg.norm(rp - nearest)
                gap = raw_dist - stick_radius - 0.02
            else:
                gap = jnp.float32(0.0)
            phi_surf     = jnp.maximum(0.0, gap) / stick_radius
            phi_approach = jnp.abs(gap)            / stick_radius

            sp = dx.site_xpos[site_id]
            traveled = jnp.dot(sp - p_start, rail_dir)
            phi_progress_remaining = jnp.maximum(0.0, 1.0 - traveled / jnp.maximum(rail_len, 1e-9))
            phi_traveled  = -traveled / jnp.maximum(rail_len, 1e-9)

            diff = sp - p_start
            lat  = diff - jnp.dot(diff, rail_dir) * rail_dir
            phi_rail_lat = jnp.linalg.norm(lat) / stick_radius

            phi_prog_vel = jnp.float32(0.0)

            if rock_body_id_named >= 0:
                qc  = dx.xquat[rock_body_id_named]
                dot = jnp.dot(qc, rock_quat_ref)
                phi_rock_ori = 1.0 - dot ** 2
            else:
                phi_rock_ori = jnp.float32(0.0)

            # Order MUST match KEYS_RUN (incl. the appended Tau_<group> tail)
            phi = jnp.stack([
                phi_tau, phi_jv, phi_ja, phi_jtc,
                phi_eng[0], phi_eng[1], phi_eng[2], phi_eng[3], phi_eng[4],
                phi_geo,
                phi_pf, phi_surf,
                phi_prog_vel, phi_rail_lat,
                phi_rock_ori, phi_approach, phi_traveled,
                phi_tau_g[0], phi_tau_g[1], phi_tau_g[2], phi_tau_g[3], phi_tau_g[4],
                phi_cap,
                phi_tau * 0.0,
                phi_jv_g[0], phi_jv_g[1], phi_jv_g[2], phi_jv_g[3], phi_jv_g[4],
                phi_elbtrack,
            ])
            return phi, traveled, v_arm, f_ema

        def rollout_k(qpos0, qvel0, U_k, w_arr, target_sched, stick_t0, u_prev_exec):
            """One sample. lax.scan over horizon. Returns cost + per-step phis.

            stick_t0: absolute base timestep of this rollout (current MPC step).
            The stick frame at horizon step idx is stick_traj[stick_t0+idx]
            (only used when stick_follow is on). Indexed like target_sched.
            """
            _mdt = mx.qpos0.dtype
            qpos0 = qpos0.astype(_mdt); qvel0 = qvel0.astype(_mdt)
            U_k = U_k.astype(_mdt); u_prev_exec = u_prev_exec.astype(_mdt)
            dx0 = mjx.make_data(mx).replace(qpos=qpos0, qvel=qvel0,
                                              ctrl=jnp.zeros(self.nu, _mdt))
            dx0 = mjx.forward(mx, dx0)
            q0_arm = qpos0[arm_qpos]

            ja_idx = KEYS_RUN.index('JA')
            jtc_idx = KEYS_RUN.index('JTC')
            pv_idx = KEYS_RUN.index('progress_vel')
            trav_idx = KEYS_RUN.index('traveled')
            brace_idx = KEYS_RUN.index('brace')
            use_hnorm  = bool(getattr(self, 'horizon_norm_traveled', True))
            href       = float(getattr(self, 'traveled_href', 45.0))
            trav_scale = jnp.float32(href / max(int(self.H), 1))
            crw       = jnp.float32(getattr(self, 'ctrl_rate_w', 0.0))
            use_crate = bool(float(getattr(self, 'ctrl_rate_w', 0.0)) > 0.0)
            use_jtc   = bool(getattr(self, 'jtc_rate', False))

            if __import__('os').environ.get('ROLLDBG'):
                print(f"[build-branch] pd_track={pd_track} virtual_target={virtual_target} "
                      f"force_press={force_press} task_space={task_space_control} "
                      f"press_control={press_control} id_ff={id_feedforward} pd_ff={pd_feedforward} "
                      f"confine={confine_motion} contact_latch={contact_latch}")

            def step_fn(carry, idx):
                dx, prev_traveled, prev_v_arm, prev_N, prev_f_ema, prev_press_ema, prev_u, prev_u_ema, prev_touched, prev_pdev, cost_acc, phi_acc = carry
                latched = prev_touched
                _pdev_next = prev_pdev
                tau_grav = dx.qfrc_bias[arm]
                if force_control:
                    efc_addr = jnp.clip(dx._impl.contact.efc_address[0],
                                        0, dx._impl.efc_force.shape[0] - 1)
                    N_now = jnp.where(
                        dx._impl.contact.dist[0] < 0,
                        jnp.abs(jnp.nan_to_num(dx._impl.efc_force[efc_addr])), 0.0)
                    J_n = jnp.nan_to_num(dx._impl.efc_J[efc_addr])
                    N_des = jnp.abs(target_sched[idx])
                    dN = (N_now - prev_N) / dt
                    tau_press_full = fc_sign * (
                        fc_kp * (N_des - N_now) - fc_kd * dN) * J_n
                    tau_press = tau_press_full[arm]
                else:
                    N_now = prev_N
                    tau_press = jnp.zeros_like(tau_grav)
                tau_close = jnp.zeros_like(tau_grav)
                tau_brake = jnp.zeros_like(tau_grav)
                tau_task  = jnp.zeros_like(tau_grav)
                tau_rf    = jnp.zeros_like(tau_grav)
                tau_rail_latch = jnp.zeros_like(tau_grav)
                J_n_geom  = jnp.zeros(mx.nv)
                if use_rock_jac and rock_geom_id >= 0 and stick_body_id >= 0:
                    rp_c = dx.geom_xpos[rock_geom_id]
                    sc_c = dx.xpos[stick_body_id]
                    sa_c = dx.xmat[stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = jnp.clip(jnp.dot(rp_c - sc_c, sa_c),
                                    -stick_half_len, stick_half_len)
                    nearest_c = sc_c + tp_c * sa_c
                    dvec = rp_c - nearest_c
                    dn = jnp.linalg.norm(dvec)
                    n_geo = dvec / jnp.maximum(dn, 1e-9)
                    gap_geo = dn - stick_radius - 0.02
                    if contact_latch:
                        _touch_now = jnp.logical_or(
                            gap_geo < latch_gap,
                            dx._impl.contact.dist[0] < 0)
                        latched = jnp.maximum(
                            prev_touched, jnp.where(_touch_now, 1.0, 0.0))
                    jacp_rock, _ = _mjx_support.jac(
                        mx, dx, rp_c, rock_body_geom)
                    J_n_geom = jacp_rock @ n_geo
                    if contact_latch:
                        Pt_l = jnp.eye(3) - jnp.outer(n_geo, n_geo)
                        railg_l = Pt_l @ rail_dir
                        railg_l = railg_l / jnp.maximum(jnp.linalg.norm(railg_l), 1e-9)
                        v_rail_now_l = jnp.dot(jacp_rock.T @ dx.qvel, railg_l)
                        F_rail_l = (latch_rail_kd * (latch_vrail - v_rail_now_l) + latch_rail_ff) * railg_l
                        tau_rail_latch = (jacp_rock[arm] @ F_rail_l) * latched
                    if task_space_control:
                        Ja = jacp_rock[arm]
                        v_ball = jacp_rock.T @ dx.qvel
                        Pt = jnp.eye(3) - jnp.outer(n_geo, n_geo)
                        _railg = Pt @ sa_c
                        _railg = _railg / jnp.maximum(jnp.linalg.norm(_railg), 1e-9)
                        _railg = _railg * jnp.sign(jnp.dot(_railg, rail_dir) + 1e-9)
                        _latd = jnp.cross(n_geo, _railg)
                        _latd = _latd / jnp.maximum(jnp.linalg.norm(_latd), 1e-9)
                        _lat_off = jnp.dot(rp_c - ts_pstart, _latd)
                        _vr_ramp = jnp.where(ts_vrail_ramp > 0.5,
                                             jnp.clip((stick_t0 + idx) / jnp.maximum(ts_vrail_ramp, 1.0), 0.0, 1.0),
                                             1.0)
                        F_rail = ts_kd_rail * (ts_vrail * _vr_ramp - jnp.dot(v_ball, _railg)) * _railg
                        F_lat = (ts_kp * (-_lat_off) - ts_kd * jnp.dot(v_ball, _latd)) * _latd
                        F_t = F_rail + F_lat
                        if ts_impedance_normal:
                            _pen_des = jnp.clip(ts_pen_des
                                                + jnp.where(press_control, U_k[idx][nu_j] * ts_pen_scale, 0.0),
                                                0.0, ts_pen_max)
                            _vn = jnp.dot(v_ball, n_geo)
                            _coef = ts_pen_kp * (-_pen_des - gap_geo) - ts_pen_kd * _vn
                            _coef = jnp.clip(_coef, -ts_fnmax, ts_fnmax)
                            F_n = _coef * n_geo
                        else:
                            _fdes_t = ts_fdes_profile[jnp.minimum(idx, ts_prof_len - 1)]
                            _fdes_t = _fdes_t + jnp.where(press_control, U_k[idx][nu_j], 0.0)
                            F_n = -jnp.maximum(_fdes_t, ts_fdes_floor) * n_geo
                        tau_task = Ja @ (F_t + F_n)
                        if _ts_post_kp_py != 0.0:
                            Jt = Ja.T
                            Nproj = jnp.eye(Ja.shape[0]) - jnp.linalg.pinv(Jt) @ Jt
                            tau_task = tau_task + Nproj @ (ts_post_kp * U_k[idx][:nu_j]
                                                           - ts_post_kd * dx.qvel[arm])
                    if rail_force:
                        Ja_rf = jacp_rock[arm]
                        vball_rf = jacp_rock.T @ dx.qvel
                        Pt_rf = jnp.eye(3) - jnp.outer(n_geo, n_geo)
                        railg_rf = Pt_rf @ sa_c
                        railg_rf = railg_rf / jnp.maximum(jnp.linalg.norm(railg_rf), 1e-9)
                        railg_rf = railg_rf * jnp.sign(jnp.dot(railg_rf, rail_dir) + 1e-9)
                        latd_rf = jnp.cross(n_geo, railg_rf)
                        latd_rf = latd_rf / jnp.maximum(jnp.linalg.norm(latd_rf), 1e-9)
                        _sdot = jnp.dot(vball_rf, railg_rf); _vn = jnp.dot(vball_rf, n_geo); _vl = jnp.dot(vball_rf, latd_rf)
                        _latoff = jnp.dot(rp_c - ts_pstart, latd_rf)
                        _vtgt = rf_vbase + (U_k[idx][pace_idx] if rw_pace else 0.0)
                        _Fpress = (U_k[idx][nu_j] if press_control else 0.0)
                        _truegap = gap_geo + 0.02 - mx.geom_size[rock_geom_id][0]
                        _Ft = jnp.clip(rf_kv * (_vtgt - _sdot), -rf_fmax, rf_fmax)
                        _farcap = jnp.where(_truegap < rf_near, rf_fnmax, rf_fapp)
                        _Fn = jnp.clip(rf_kpn * (_truegap + rf_pen) - rf_kdn * _vn + _Fpress, 0.0, _farcap)
                        _Fl = -rf_kplat * _latoff - rf_kdlat * _vl
                        F_rf = _Ft * railg_rf + _Fn * (-n_geo) + _Fl * latd_rf
                        tau_rf = Ja_rf @ F_rf
                        if _rw_post_kp_py != 0.0:
                            _Jtrf = Ja_rf.T
                            _Nrf = jnp.eye(arm.shape[0]) - jnp.linalg.pinv(_Jtrf) @ _Jtrf
                            tau_rf = tau_rf + _Nrf @ (rw_post_kp * U_k[idx][:nu_j] - rw_post_kd * dx.qvel[arm])
                    if use_gap_closer:
                        gc_gap = jnp.where(gap_geo < gap_close_max,
                                           jnp.maximum(gap_geo, 0.0), 0.0)
                        tau_close = (fc_sign_close * fc_kp_close
                                     * gc_gap * J_n_geom)[arm]
                    if use_rail_brake:
                        J_rail = jacp_rock @ rail_dir
                        v_rail_now = (dx.qvel @ jacp_rock) @ rail_dir
                        tau_brake = (-fc_kv * (v_rail_now - v_rail_target)
                                     * J_rail)[arm]
                if press_control:
                    efc_addr_p = jnp.clip(dx._impl.contact.efc_address[0],
                                          0, dx._impl.efc_force.shape[0] - 1)
                    J_n_p      = jnp.nan_to_num(dx._impl.efc_J[efc_addr_p])
                    press_raw  = U_k[idx][nu_j]
                    press_ema_val = (press_ema_a * press_raw
                                     + (1.0 - press_ema_a) * prev_press_ema)
                    if virtual_target:
                        gap_vt = gap_geo
                        vn_vt  = J_n_geom @ dx.qvel
                        Fmag_vt = vt_kp_j * (gap_vt + press_ema_val) - vt_kd_j * vn_vt
                        Fmag_vt = jnp.clip(Fmag_vt, 0.0, 120.0)
                        tau_press_amp = jnp.where(
                            gap_vt < 0.05, Fmag_vt * J_n_geom, 0.0)[arm]
                        _gt_vt = J_n_geom[arm] * vt_free_mask
                        _g2_vt = jnp.dot(_gt_vt, _gt_vt) + 1e-9
                        _uk_vt = U_k[idx][:nu_j]
                        Uk_apply_vt = jnp.where(
                            gap_vt < 0.015,
                            _uk_vt - (jnp.dot(_uk_vt, _gt_vt) / _g2_vt) * _gt_vt,
                            _uk_vt)
                    elif force_press:
                        # (so the IRL can learn press_force); clamp ≥0 so it never pulls off.
                        _fdes_ts = jnp.maximum(press_force_des + press_ema_val, 0.0)
                        if ts_discover:
                            _appf_p = jnp.clip(1.0 - (stick_t0 + idx) / jnp.maximum(ts_n_app_f, 1.0), 0.0, 1.0)
                            _fdes_ts = _fdes_ts * (_appf_p <= 0.0).astype(_fdes_ts.dtype)
                        tau_press_amp = (-_fdes_ts * J_n_geom)[arm]
                        Uk_apply_em = U_k[idx][:nu_j]
                    elif depth_impedance:
                        tau_press_amp = (depth_kn * (depth_target - gap_geo)
                                         * J_n_geom)[arm]
                        Uk_apply_em = U_k[idx][:nu_j]
                    else:
                        _ungate = bool(getattr(self, 'press_ungated', False))
                        tau_press_amp = jnp.where(
                            dx._impl.contact.dist[0] < 0,
                            press_ema_val * J_n_p,
                            (press_ema_val * J_n_geom if _ungate else 0.0))[arm]
                        if contact_latch:
                            if not latch_project:
                                tau_hold = jnp.where(
                                    latched > 0.5,
                                    depth_kn * (depth_target - gap_geo), 0.0)
                                tau_press_amp = tau_press_amp + (tau_hold * J_n_geom)[arm]
                            _gt_lt = J_n_geom[arm] * vt_free_mask
                            _g2_lt = jnp.dot(_gt_lt, _gt_lt) + 1e-9
                            _uk_lt = U_k[idx][:nu_j]
                            Uk_apply_em = jnp.where(
                                latched > 0.5,
                                _uk_lt - (jnp.dot(_uk_lt, _gt_lt) / _g2_lt) * _gt_lt,
                                _uk_lt)
                        elif confine_motion:
                            _gt_em = J_n_geom[arm] * vt_free_mask
                            _g2_em = jnp.dot(_gt_em, _gt_em) + 1e-9
                            _uk_em = U_k[idx][:nu_j]
                            Uk_apply_em = jnp.where(
                                gap_geo < 0.015,
                                _uk_em - (jnp.dot(_uk_em, _gt_em) / _g2_em) * _gt_em,
                                _uk_em)
                        else:
                            Uk_apply_em = U_k[idx][:nu_j]
                else:
                    press_ema_val = prev_press_ema
                    tau_press_amp = jnp.zeros_like(tau_grav)
                if task_space:
                    _abs = jnp.clip(stick_t0 + idx, 0, ts_sched_len - 1)
                    _s0_off = (jnp.clip(U_k[0][ts_s0_idx] * ts_s0_scale, -ts_s0_max, ts_s0_max)
                               if ts_discover else jnp.float32(0.0))
                    _pace_k = jnp.maximum(U_k[idx][0], 0.0) if ts_pace_fwd else U_k[idx][0]
                    _s_t = jnp.clip(ts_s_nom[_abs] + _s0_off + _pace_k * ts_pace_scale, ts_smin, ts_smax)
                    q_base = _ts_interp(ts_q_of_s, _s_t)
                    null_t = _ts_interp(_ts_nullflat, _s_t).reshape(ts_n_null, ts_narm)
                    _ncoef = U_k[idx][1:1 + ts_n_null]
                    if ts_posture_sample:
                        q_ref = jnp.clip(q0_arm + _ncoef @ null_t, q_lim_lo_arm, q_lim_hi_arm)
                    elif ts_posture_joint:
                        _q_led = q_base * (1.0 - _ts_joint_mask) + _ts_joint_center * _ts_joint_mask
                        q_ref = jnp.clip(_q_led + ts_joint_gain * (U_k[idx][1:1 + ts_n_joint] @ _ts_joint_rows),
                                         q_lim_lo_arm, q_lim_hi_arm)
                    elif ts_null_runtime:
                        # the DRIVE before integrating, so posture moves and the tool never leaves.
                        _Jn_rt = jacp_rock[arm]
                        _Nrt = jnp.eye(ts_narm) - _Jn_rt @ jnp.linalg.inv(
                            _Jn_rt.T @ _Jn_rt + 1e-4 * jnp.eye(3)) @ _Jn_rt.T
                        q_ref = jnp.clip(q_base + _Nrt @ (_ncoef @ null_t),
                                         q_lim_lo_arm, q_lim_hi_arm)
                    else:
                        q_ref = q_base + _ncoef @ null_t
                    if (ts_reanchor_exact and not ts_null_runtime) or ts_posture_joint or ts_posture_sample:
                        if _ts_qref_warm:
                            _pdev0 = jnp.where(idx == 0, q0_arm - q_base, prev_pdev)
                            _dev_seed = ts_qref_dev_ema * (q_ref - q_base) + (1.0 - ts_qref_dev_ema) * _pdev0
                            q_ref = jnp.clip(q_base + _dev_seed, q_lim_lo_arm, q_lim_hi_arm)
                        _sel = (_ts_joint_mask if ts_posture_joint else jnp.zeros((ts_narm,), dtype=jnp.float32))
                        # offset is wide, so it stalled short of the demo posture AND never cleaned the
                        # rail → the leak survived → the posture cost could never select the demo sample.
                        _q_lin = q_ref
                        if ts_p_table:
                            _p_tgt = _ts_interp(ts_p_of_s, _s_t)
                        else:
                            _p_tgt = _ts_kin_fn(
                                mx, dx.replace(qpos=dx.qpos.at[arm_qpos].set(q_base))
                            ).geom_xpos[rock_geom_id]
                        if _ts_press_ct_on:
                            _pdep = jnp.clip(ts_press_depth
                                             + jnp.where(press_control, U_k[idx][nu_j] * ts_pen_scale, 0.0),
                                             0.0, ts_pen_max)
                            _p_tgt = _p_tgt - _pdep * n_geo
                        _Wi = _ts_proj_winv * (1.0 - _sel)
                        def _gn_body(_gi, _qr):
                            _dxr = _ts_compos_fn(mx, _ts_kin_fn(
                                mx, dx.replace(qpos=dx.qpos.at[arm_qpos].set(_qr))))
                            _p_now = _dxr.geom_xpos[rock_geom_id]
                            _jacp, _ = _ts_support_mod.jac(mx, _dxr, _p_now, rock_body_geom)
                            _JaW = _Wi[:, None] * _jacp[arm]
                            _dq_gn = jnp.clip(_JaW @ jnp.linalg.inv(_jacp[arm].T @ _JaW + 1e-4 * jnp.eye(3))
                                              @ (_p_now - _p_tgt), -0.2, 0.2)
                            return jnp.clip(_qr - _dq_gn, q_lim_lo_arm, q_lim_hi_arm)
                        q_ref = jax.lax.fori_loop(0, ts_reanchor_iters, _gn_body, q_ref)
                        q_ref = jnp.where(jnp.all(jnp.isfinite(q_ref)), q_ref, _q_lin)
                        if _ts_qref_warm:
                            # approach lift is never fed back into the seed (it decays on its own).
                            _pdev_next = q_ref - q_base
                    elif ts_reanchor:
                        _reB_t = _ts_interp(_ts_reBflat, _s_t).reshape(3, ts_n_null, ts_n_null)
                        _drift = 0.5 * jnp.einsum('i,j,dij->d', _ncoef, _ncoef, _reB_t)
                        _reJ_t = _ts_interp(_ts_reJpTflat, _s_t).reshape(ts_narm, 3)
                        q_ref = q_ref - _reJ_t @ _drift
                    if ts_discover:
                        _lift0 = jnp.clip(U_k[0][ts_lift_idx] * ts_lift_scale, 0.0, ts_lift_max)
                        _appf = jnp.clip(1.0 - _abs / jnp.maximum(ts_n_app_f, 1.0), 0.0, 1.0)
                        q_ref = q_ref + (_lift0 * _appf) * _ts_interp(ts_lift_of_s, _s_t)
                    if _ts_dev_ema_on:
                        # (the forward sweep) is left untouched → the sweep never lags. Toy-validated
                        _pdev_next = ts_qref_dev_ema * (q_ref - q_base) + (1.0 - ts_qref_dev_ema) * prev_pdev
                        q_ref = q_base + _pdev_next
                    _Marm = dx._impl.qM[arm][:, arm]
                    _ff_ts = jnp.clip(_Marm @ qddot_ff[_abs],
                                      -ts_ff_frac * u_max[:nu_j], ts_ff_frac * u_max[:nu_j])
                    if __import__('os').environ.get('ROLLDBG'):
                        jax.lax.cond(idx == 0, lambda: (jax.debug.print(
                            "[ctrl0] qref_nan={a} q_nan={q} Marm_nan={b} qddotff_nan={c} ff_nan={d}",
                            a=jnp.isnan(q_ref).any(), q=jnp.isnan(dx.qpos[arm_qpos]).any(),
                            b=jnp.isnan(_Marm).any(), c=jnp.isnan(qddot_ff[_abs]).any(),
                            d=jnp.isnan(_ff_ts).any()), 0)[1], lambda: 0)
                    # equality/locked-joint solve emit nan; nan→0 (byte-identical when finite).
                    _qfrc_c = jnp.where(jnp.isnan(dx.qfrc_constraint[arm]), 0.0,
                                        dx.qfrc_constraint[arm])
                    if __import__('os').environ.get('ROLLDBG'):
                        jax.lax.cond((idx == 0) | (idx == 6) | (idx == 11), lambda: (jax.debug.print(
                            "[qfrc] idx={i} was_nan={n} gap={g} maxabs={m}", i=idx,
                            n=jnp.isnan(dx.qfrc_constraint[arm]).any(), g=gap_geo,
                            m=jnp.max(jnp.abs(_qfrc_c))), 0)[1], lambda: 0)
                    if ts_task_pd:
                        _Ja2 = jacp_rock[arm]
                        _Pt2 = jnp.eye(3) - jnp.outer(n_geo, n_geo)
                        _railt2 = _Pt2 @ rail_dir
                        _railt2 = _railt2 / jnp.maximum(jnp.linalg.norm(_railt2), 1e-9)
                        _railt2 = _railt2 * jnp.sign(jnp.dot(_railt2, rail_dir) + 1e-9)
                        _latd2 = jnp.cross(n_geo, _railt2)
                        _latd2 = _latd2 / jnp.maximum(jnp.linalg.norm(_latd2), 1e-9)
                        _vtool2 = jacp_rock.T @ dx.qvel
                        _vref2  = _Ja2.T @ qdot_ff[_abs]
                        _frail_s2 = jnp.clip(ts_kd_railf * (jnp.dot(_vref2, _railt2)
                                                            - jnp.dot(_vtool2, _railt2)), -60.0, 60.0)
                        _Frail2 = _frail_s2 * _railt2
                        _latoff2 = jnp.dot(rp_c - ts_pstart, _latd2)
                        _Flat2 = (ts_kp_lat * (-_latoff2)
                                  - ts_kd_lat * jnp.dot(_vtool2, _latd2)) * _latd2
                        _pen_des2 = jnp.clip(ts_pen_des
                                             + jnp.where(press_control, U_k[idx][nu_j] * ts_pen_scale, 0.0),
                                             0.0, ts_pen_max)
                        _vn2 = jnp.dot(_vtool2, n_geo)
                        _spring2 = ts_pen_kp * (-_pen_des2 - gap_geo)
                        _damp2 = -(ts_pen_kd * _vn2)
                        if _ts_pen_damp_frac_py > 0.0:
                            _dcap2 = jnp.float32(_ts_pen_damp_frac_py) * jnp.abs(_spring2)
                            _damp2 = jnp.clip(_damp2, -_dcap2, _dcap2)
                        _coef2 = jnp.clip(_spring2 + _damp2, -ts_fnmax, ts_fnmax)
                        _Fn2 = _coef2 * n_geo
                        # undamped → 17mm slam → 106N eject → nan.
                        if __import__('os').environ.get('ROLLDBG'):
                            jax.lax.cond(idx == 0, lambda: (jax.debug.print(
                                "[term0] qdotff_nan={a} qvel_nan={b} qfrc_nan={c}",
                                a=jnp.isnan(qdot_ff[_abs]).any(), b=jnp.isnan(dx.qvel[arm]).any(),
                                c=jnp.isnan(dx.qfrc_constraint[arm]).any()), 0)[1], lambda: 0)
                        _uk_motion = (_ff_ts + _Ja2 @ (_Frail2 + _Flat2 + _Fn2)
                                      + _Marm @ (ts_KD * (qdot_ff[_abs] - dx.qvel[arm]))
                                      - ts_ccancel * _qfrc_c)
                        # untouched). Without this the sampled/biased null coeffs never move the elbow
                        if _ts_null_kp_on and not ts_posture_free:
                            if ts_posture_joint or ts_posture_sample:
                                _e_null = q_ref - dx.qpos[arm_qpos]
                            elif ts_null_runtime:
                                _e_null = _Nrt @ (q_ref - dx.qpos[arm_qpos])
                            else:
                                _e_null = null_t.T @ (null_t @ (q_ref - dx.qpos[arm_qpos]))
                            _uk_motion = _uk_motion + _Marm @ (ts_null_kp * _e_null)
                            if __import__('os').environ.get('ROLLDBG'):
                                jax.lax.cond(idx == 0, lambda: (jax.debug.print(
                                    "[force0] Frail_nan={a} Flat_nan={b} Fn_nan={c} enull_nan={d} Ja_nan={j} uk_nan={e}",
                                    a=jnp.isnan(_Frail2).any(), b=jnp.isnan(_Flat2).any(),
                                    c=jnp.isnan(_Fn2).any(), d=jnp.isnan(_e_null).any(),
                                    j=jnp.isnan(_Ja2).any(), e=jnp.isnan(_uk_motion).any()), 0)[1], lambda: 0)
                        if ts_posture_free:
                            _Jn_f = jacp_rock[arm]
                            _Nf   = jnp.eye(ts_narm) - _Jn_f @ jnp.linalg.inv(
                                _Jn_f.T @ _Jn_f + 1e-4 * jnp.eye(3)) @ _Jn_f.T
                            _drive_f = ts_joint_gain * (U_k[idx][1:1 + ts_n_joint] @ _ts_joint_rows)
                            _uk_motion = _uk_motion + _Marm @ (_Nf @ _drive_f)
                    else:
                        _e_trk = q_ref - dx.qpos[arm_qpos]
                        _e_trk = _e_trk - (1.0 - ts_null_kp_scale) * (null_t.T @ (null_t @ _e_trk))
                        _uk_motion = _ff_ts + _Marm @ (ts_KP * _e_trk
                                                       + ts_KD * (qdot_ff[_abs] - dx.qvel[arm])) \
                            - ts_ccancel * _qfrc_c
                        # Tracking q_ref INTO the stiff stick without this rings → 17mm slam → eject → nan.
                        _vn_ct = jnp.dot(jacp_rock.T @ dx.qvel, n_geo)
                        _uk_motion = _uk_motion - jnp.where(
                            gap_geo < 0.01, jacp_rock[arm] @ (ts_pen_kd * _vn_ct * n_geo), 0.0)
                        # each task force; this branch didn't clamp anything → the wide-sample nan).
                        _uk_motion = jnp.clip(_uk_motion, -u_max[:nu_j], u_max[:nu_j])
                    # constructed trio must add it or the pressed tool DRAGS (velocity lag →
                    _Pt_fr = jnp.eye(3) - jnp.outer(n_geo, n_geo)
                    _railt = _Pt_fr @ rail_dir
                    _railt = _railt / jnp.maximum(jnp.linalg.norm(_railt), 1e-9)
                    _railt = _railt * jnp.sign(jnp.dot(_railt, rail_dir) + 1e-9)
                    _fn_cmd = press_force_des + jnp.where(press_control, U_k[idx][nu_j], 0.0)
                    _tau_fric = jacp_rock[arm] @ (ts_mu_ff * _fn_cmd * _railt)
                    _uk_motion = _uk_motion + jnp.where(gap_geo < 0.01, _tau_fric, 0.0)
                elif pd_track:
                    _uk_pd = U_k[idx][:nu_j]
                    if confine_motion:
                        _gt_pd = J_n_geom[arm] * vt_free_mask
                        _g2_pd = jnp.dot(_gt_pd, _gt_pd) + 1e-9
                        _proj_pd = _uk_pd - (jnp.dot(_uk_pd, _gt_pd) / _g2_pd) * _gt_pd
                        # the normal, so the PD must NEVER command it — otherwise the PD's
                        _uk_pd = (_proj_pd if force_press
                                  else jnp.where(gap_geo < 0.015, _proj_pd, _uk_pd))
                    q_ref = q0_arm + _uk_pd
                    _ff = jnp.zeros_like(_uk_pd)
                    if id_feedforward:
                        _ff = u_ff_id[idx] - tau_grav
                    elif pd_feedforward:
                        _ff = dx._impl.qM[arm][:, arm] @ qddot_ff[idx]
                    _uk_motion = (_ff + pd_kp_j * (q_ref - dx.qpos[arm_qpos])
                                  - pd_kd_j * dx.qvel[arm])
                    if pd_free_normal:
                        _gn_pd = J_n_geom[arm]
                        _gn2_pd = jnp.dot(_gn_pd, _gn_pd) + 1e-9
                        _uk_motion = _uk_motion - (jnp.dot(_uk_motion, _gn_pd) / _gn2_pd) * _gn_pd
                else:
                    _uk_motion = (Uk_apply_vt if virtual_target
                                  else (Uk_apply_em if (press_control and (confine_motion or contact_latch))
                                        else U_k[idx][:nu_j]))
                if rail_force:
                    u = jnp.clip(tau_grav + tau_rf, u_min[:nu_j], u_max[:nu_j])
                elif task_space_control:
                    u = jnp.clip(tau_grav + tau_task, u_min[:nu_j], u_max[:nu_j])
                else:
                    u = jnp.clip(
                        tau_grav + _uk_motion
                        + tau_press + tau_close + tau_brake + tau_press_amp + tau_rail_latch,
                        u_min[:nu_j], u_max[:nu_j])
                if __import__('os').environ.get('ROLLDBG'):
                    # fed into an eager mjx.step to reproduce the nan deterministically.
                    jax.lax.cond(
                        idx == 0,
                        lambda: (jax.debug.print("[rt0-u] u={u}", u=u), 0)[1],
                        lambda: 0)
                ctrl = dx.ctrl.at[:].set(u)
                dx = dx.replace(ctrl=ctrl)
                if locked_q.shape[0] > 0:
                    dx = dx.replace(
                        qpos=dx.qpos.at[locked_q].set(locked_vals),
                        qvel=dx.qvel.at[locked_dof].set(0.0),
                    )
                if stick_follow and has_stick:
                    abs_t = jnp.clip(stick_t0 + idx, 0, T_stick - 1)
                    stick_xyz = stick_traj_j[abs_t]
                    dx = dx.replace(
                        qpos=dx.qpos.at[stick_qposadr:stick_qposadr + 3].set(stick_xyz),
                        qvel=dx.qvel.at[stick_qposadr:stick_qposadr + 6].set(0.0),
                    )
                for _ in range(rollout_nsub):
                    dx = mjx.step(mx, dx)
                if __import__('os').environ.get('ROLLDBG'):
                    jax.lax.cond(
                        idx == 0,
                        lambda: (jax.debug.print(
                            "[post-step0] qpos_nan={a} qvel_nan={b} qacc_nan={c}",
                            a=jnp.isnan(dx.qpos).any(), b=jnp.isnan(dx.qvel).any(),
                            c=jnp.isnan(dx.qacc).any()), 0)[1],
                        lambda: 0)

                if contact_latch and latch_project and use_rock_jac and not rail_force \
                        and rock_geom_id >= 0 and stick_body_id >= 0:
                    _rp_p = dx.geom_xpos[rock_geom_id]; _sc_p = dx.xpos[stick_body_id]
                    _sa_p = dx.xmat[stick_body_id].reshape(3, 3)[:, 2]
                    _tp_p = jnp.clip(jnp.dot(_rp_p - _sc_p, _sa_p), -stick_half_len, stick_half_len)
                    _dv_p = _rp_p - (_sc_p + _tp_p * _sa_p); _dn_p = jnp.linalg.norm(_dv_p)
                    _n_p = _dv_p / jnp.maximum(_dn_p, 1e-9)
                    _gap_p = _dn_p - stick_radius - mx.geom_size[rock_geom_id][0]
                    _fdes_p = target_sched[idx] + (press_ema_val if (rail_walk and press_control) else 0.0)
                    _dz_p = jnp.maximum(_fdes_p, 0.0) / latch_k
                    _jp, _ = _mjx_support.jac(mx, dx, _rp_p, rock_body_geom)
                    _Ja_p = (_jp[arm] * vt_free_mask[:, None]).T
                    if rail_walk:
                        # (the min-norm 1e-6 J⁺ blow-up = the nan). Near-limit joints get
                        _qa_p = dx.qpos[arm_qpos]
                        _hlf_p = 0.5 * (q_lim_hi_arm - q_lim_lo_arm)
                        _mid_p = 0.5 * (q_lim_hi_arm + q_lim_lo_arm)
                        _prx_p = jnp.abs(_qa_p - _mid_p) / jnp.maximum(_hlf_p, 1e-6)
                        _prx_p = jnp.clip((_prx_p - (1.0 - rw_margin)) / jnp.maximum(rw_margin, 1e-6), 0.0, 1.0)
                        _Winv_p = 1.0 / (1.0 + rw_wlim * _prx_p * _prx_p)
                        _JJt_p = (_Ja_p * _Winv_p[None, :]) @ _Ja_p.T + (rw_lambda * rw_lambda) * jnp.eye(3)
                    else:
                        _Winv_p = jnp.ones(arm.shape[0])
                        _JJt_p = _Ja_p @ _Ja_p.T + 1e-6 * jnp.eye(3)
                    _railt_r = (jnp.eye(3) - jnp.outer(_n_p, _n_p)) @ rail_dir
                    _railt_r = _railt_r / jnp.maximum(jnp.linalg.norm(_railt_r), 1e-9)
                    _vrail_k = (jnp.maximum(latch_vrail + U_k[idx][pace_idx], 0.0) if rw_pace else latch_vrail)
                    _dx_n = -(_gap_p + _dz_p) * _n_p + _vrail_k * dt * _railt_r
                    _vx_p = _Ja_p @ dx.qvel[arm]
                    _vn_x = jnp.dot(_vx_p, _n_p) * _n_p
                    _dq_task = _Winv_p * (_Ja_p.T @ jnp.linalg.solve(_JJt_p, _dx_n))
                    if _rw_post_kp_py != 0.0:
                        _Jp_inv = _Ja_p.T @ jnp.linalg.solve(_JJt_p, jnp.eye(3))
                        _Nproj = jnp.eye(arm.shape[0]) - _Jp_inv @ _Ja_p
                        _dq_task = _dq_task + _Nproj @ (rw_post_kp * U_k[idx][:nu_j] - rw_post_kd * dx.qvel[arm]) * dt
                    _dq_p = jnp.where(latched > 0.5, _dq_task, 0.0)
                    _dv_q = jnp.where(latched > 0.5, -_Winv_p * (_Ja_p.T @ jnp.linalg.solve(_JJt_p, _vn_x)), 0.0)
                    dx = dx.replace(
                        qpos=dx.qpos.at[arm_qpos].add(_dq_p),
                        qvel=dx.qvel.at[arm].add(_dv_q))
                    if rail_walk:
                        dx = mjx.forward(mx, dx)

                if _ts_pin_n_py > 0:
                    dx = dx.replace(
                        qpos=dx.qpos.at[_ts_pin_qpos].set(_ts_pin_vals),
                        qvel=dx.qvel.at[_ts_pin_dof].set(0.0))
                u_feat = tau_feat_ema * u + (1.0 - tau_feat_ema) * prev_u_ema
                phi_h, traveled, v_arm_now, f_ema_now = _running_phi(
                    dx, u_feat, target_sched[idx], prev_f_ema, prev_v_arm,
                    dz_press=press_ema_val)
                rail_v = (traveled - prev_traveled) / dt
                if prog_vel_track:
                    pv_val = (rail_v - v_rail_target) ** 2
                elif prog_vel_sq:
                    pv_val = rail_v ** 2
                else:
                    pv_val = jnp.where(bound_pv,
                                       -jnp.minimum(rail_v, pv_cap), -rail_v)
                phi_h = phi_h.at[pv_idx].set(pv_val)
                if use_trav_track:
                    _abs_tr = jnp.clip(stick_t0 + idx, 0, T_trav - 1)
                    _dev = (traveled - traveled_ref_j[_abs_tr]) / jnp.maximum(rail_len, 1e-9)
                    phi_h = phi_h.at[trav_idx].set(_dev ** 2)
                ddq_kin = (v_arm_now - prev_v_arm) / dt
                phi_h = phi_h.at[ja_idx].set(jnp.linalg.norm(ddq_kin) / 100.0)
                # MUST be symmetric with human_base.get_traj_features (jtc_rate).
                if use_jtc:
                    phi_h = phi_h.at[jtc_idx].set(
                        jnp.linalg.norm(u - prev_u) / 50.0
                        * jnp.where(idx > 0, 1.0, 0.0))
                # no rock jac / pre-contact). LOW = braced (force through the structure). MUST mirror
                phi_h = phi_h.at[brace_idx].set(jnp.linalg.norm(J_n_geom[arm]))

                w_h = w_arr[idx]
                phi_plan = phi_h.at[trav_idx].multiply(trav_scale) if use_hnorm else phi_h
                step_cost = jnp.dot(w_h, phi_plan) * dt
                if use_crate:
                    _use_anchor = bool(getattr(self, 'boundary_anchor', False))
                    step_cost = step_cost + crw * jnp.sum((u - prev_u) ** 2) * (
                        1.0 if _use_anchor else jnp.where(idx > 0, 1.0, 0.0))
                if use_jlim:
                    _viol = (jnp.maximum(q_lim_lo - dx.qpos, 0.0)
                             + jnp.maximum(dx.qpos - q_lim_hi, 0.0))
                    step_cost = step_cost + jlim_w * jnp.sum(
                        q_lim_mask * _viol ** 2)

                return (dx, traveled, v_arm_now, N_now, f_ema_now, press_ema_val,
                        u, u_feat, latched, _pdev_next, cost_acc + step_cost,
                        phi_acc.at[idx].set(phi_h)), dx.qpos

            _fdt = dx0.qvel.dtype
            init = (dx0, jnp.zeros((), _fdt), dx0.qvel[arm],
                    jnp.zeros((), _fdt), jnp.zeros((), _fdt), jnp.zeros((), _fdt),
                    u_prev_exec.astype(_fdt),
                    jnp.zeros_like(dx0.qvel[arm]),
                    jnp.zeros((), _fdt),
                    jnp.zeros_like(dx0.qvel[arm]),
                    jnp.zeros((), _fdt),
                    jnp.zeros((self.H, self.nr_run), _fdt))
            (_, _, _, _, _, _, _, _, _, _, total_cost, phis), qpos_traj = jax.lax.scan(
                step_fn, init, jnp.arange(self.H))
            return total_cost, phis, qpos_traj

        def _sample_eps(sub, sigma_L, decay):
            """Per-sample control perturbation eps (K, H, nu).
            Discrete: independent per-step noise. Spline: noise on M control
            points mapped through the basis → smooth by construction.
            Fixed-sample mode: use precomputed deterministic z (cyclic shift
            by DIAL pass for diversity without randomness).
            """
            if use_fixed:
                shift = jax.random.randint(sub, (), 0, self.K)
                z = jnp.roll(fixed_z, shift, axis=0)
            elif use_spline:
                z = jax.random.normal(sub, (self.K, spline_M_val, self.n_ctrl))
            else:
                z = jax.random.normal(sub, (self.K, self.H, self.n_ctrl))

            if use_color:
                z = jnp.einsum('mn,knj->kmj', color_A, z)

            if use_spline:
                if sigma_L.ndim == 3:
                    eps_cp = jnp.einsum('mij,kmj->kmi', sigma_L, z) * spline_gain * decay
                else:
                    eps_cp = jnp.einsum('ij,kmj->kmi', sigma_L, z) * spline_gain * decay
                return jnp.einsum('hm,kmn->khn', B_spline, eps_cp)
            return jnp.einsum('ij,khj->khi', sigma_L, z) * decay

        sectioned = bool(getattr(self, 'sectioned_softmax', False))
        contact_gate = bool(getattr(self, 'contact_motion_gate', False))
        gate_floor = float(getattr(self, 'contact_gate_floor', 0.05))
        press_touch_eps = float(getattr(self, 'press_touch_eps', 1e-3))
        try:
            _press_idx = KEYS_RUN.index('press_force')
        except ValueError:
            _press_idx = -1
        if _press_idx < 0:
            sectioned = False

        def mppi_update(qpos0, qvel0, U, key, w_arr, target_sched, stick_t0,
                        sigma_L, u_prev_exec, t_press_start):
            """One DIAL iteration step. Returns (U_new, phis_chosen, costs)."""
            if self._anneal_on:
                sig_arr = jnp.asarray(self._anneal_sig, dtype=jnp.float32)
                lam_arr = jnp.asarray(self._anneal_lam, dtype=jnp.float32)
            else:
                sig_arr = jnp.asarray(
                    [self.dial_decay ** d for d in range(self.n_dial)], dtype=jnp.float32)
                lam_arr = jnp.full((self.n_dial,), self.lam, dtype=jnp.float32)

            def _dial_body(d, carry):
                U, key = carry
                _dec = sig_arr[d]
                _lam = lam_arr[d]
                key, _ = jax.random.split(key)
                sub = jax.random.fold_in(jax.random.PRNGKey(0), d)
                eps = _sample_eps(sub, sigma_L, _dec)
                costs, phis_all, _ = jax.vmap(
                    lambda e: rollout_k(qpos0, qvel0, U + e, w_arr, target_sched, stick_t0, u_prev_exec)
                )(eps)
                finite = jnp.isfinite(costs)
                c_safe = jnp.where(finite, costs,
                                    jnp.max(jnp.where(finite, costs, -jnp.inf)))
                if not sectioned:
                    c_min = jnp.min(c_safe)
                    weights = jnp.exp(-(c_safe - c_min) / _lam)
                    weights = weights / (jnp.sum(weights) + 1e-10)
                    U = U + jnp.einsum('k,khn->hn', weights, eps)
                    U = jnp.clip(U, u_min, u_max)
                    return (U, key)

                h_abs = stick_t0 + jnp.arange(self.H)
                is_press_h = h_abs >= t_press_start
                press_contrib = self.dt * jnp.sum(
                    w_arr[:, _press_idx][None, :] * phis_all[:, :, _press_idx],
                    axis=1)
                c_task = c_safe - press_contrib
                ct_min = jnp.min(c_task)
                w_task = jnp.exp(-(c_task - ct_min) / _lam)
                w_task = w_task / (jnp.sum(w_task) + 1e-10)
                touch_kh = (phis_all[:, :, _press_idx] > press_touch_eps)
                if contact_gate:
                    _pw = jnp.maximum(jnp.sum(is_press_h.astype(w_task.dtype)), 1.0)
                    press_touch_k = jnp.sum(
                        touch_kh.astype(w_task.dtype) * is_press_h[None, :],
                        axis=1) / _pw
                    g_k = press_touch_k + gate_floor
                    w_task = w_task * g_k
                    w_task = w_task / (jnp.sum(w_task) + 1e-10)
                cf_min = jnp.min(c_safe)
                w_full = jnp.exp(-(c_safe - cf_min) / _lam)
                num_kh = w_full[:, None] * touch_kh.astype(w_full.dtype)
                denom_h = jnp.sum(num_kh, axis=0)
                any_touch_h = denom_h > 0.0
                w_press_kh = jnp.where(
                    any_touch_h[None, :],
                    num_kh / (denom_h[None, :] + 1e-10),
                    w_task[:, None])
                w_kh = jnp.where(is_press_h[None, :], w_press_kh, w_task[:, None])
                U = U + jnp.einsum('kh,khn->hn', w_kh, eps)
                U = jnp.clip(U, u_min, u_max)
                return (U, key)

            U, key = jax.lax.fori_loop(0, self.n_dial, _dial_body, (U, key))
            return U, key

        if __import__('os').environ.get('TS_DUMP_SETUP'):
            _dp = __import__('os').environ.get('TS_DUMP_PATH', 'data/mppi_setup_dump.npz')
            def _g(name, default=None):
                v = getattr(self, name, default)
                return np.asarray(v) if v is not None else np.array([])
            def _gm(*names):
                for nm in names:
                    v = getattr(self, nm, None)
                    if v is not None:
                        return np.asarray(v)
                return np.array([])
            def _gf(name, dflt):
                return np.float32(getattr(self, name, dflt))
            np.savez(_dp,
                     p_start=_g('p_start'), rail_dir=_g('rail_dir'), rail_len=np.float32(getattr(self, 'rail_len', 0.0)),
                     stick_half_len=np.float32(getattr(self, 'stick_half_len', 0.45)),
                     stick_radius=np.float32(getattr(self, 'stick_radius', 0.02)),
                     init_qpos=_g('_dump_init_qpos'), init_qvel=_g('_dump_init_qvel'),
                     qddot_ff=_g('_qddot_ff'), qdot_ff=_g('_qdot_ff'),
                     ts_q_of_s=_g('ts_q_of_s'), ts_null_of_s=_g('ts_null_of_s'),
                     ts_s_nom=_g('ts_s_nom'), ts_s_grid=_g('ts_s_grid'), ts_lift_of_s=_g('ts_lift_of_s'),
                     arm_qposadr=_g('arm_qposadr'), arm_dofadr=_g('arm_dofadr'),
                     w_run=_g('w_run'), keys_run=np.array([str(k) for k in getattr(self, 'keys_run', [])]),
                     dt=np.float32(getattr(self, 'dt', 0.0)), H=np.int32(self.H), nu=np.int32(self.nu),
                     stick_traj=_gm('stick_traj', '_stick_traj', '_stick_traj_ctrl'),
                     stick_qposadr=np.int32(getattr(self, '_stick_qposadr', getattr(self, 'stick_qposadr', -1)) or -1),
                     K=np.int32(getattr(self, 'K', 128)), n_dial=np.int32(getattr(self, 'n_dial', 1)),
                     mppi_lambda=_gf('lambda_', getattr(self, 'mppi_lambda', 0.1)),
                     ts_pace_sigma=_gf('ts_pace_sigma', 2.0), ts_null_sigma=_gf('ts_null_sigma', 6.0),
                     ts_pace_scale=_gf('ts_pace_scale', 0.3), ts_pen_scale=_gf('ts_pen_scale', 0.0003),
                     press_force_des=_gf('press_force_des', 26.0), ts_proj_winv=_g('ts_proj_winv'),
                     vt_kp=_gf('vt_kp', 20000.0), force_max=_gf('force_max', 80.0), force_ema=_gf('force_ema', 1.0),
                     vel_feat_ema=_gf('vel_feat_ema', 1.0), press_ref=_gf('press_ref', 0.0),
                     press_emergent=np.int32(1 if getattr(self, 'press_emergent', False) else 0),
                     target_force=_gf('target_force', 26.0), rock_quat_ref=_g('rock_quat_ref'),
                     elbow_qposadr=_g('_elbow_qposadr'), elbow_ref=_g('_elbow_ref'),
                     w_run_keys=np.array([str(k) for k in getattr(self, 'w_run', {}).keys()]),
                     w_run_vals=np.array([float(v) for v in getattr(self, 'w_run', {}).values()]))
            print(f"[TS_DUMP_SETUP] wrote {_dp} — rail_len={getattr(self,'rail_len',0):.4f}m, "
                  f"init_qpos[{np.asarray(self._dump_init_qpos).shape}] — exiting before compile")
            raise SystemExit(0)

        print(f"[KinematicMPPI_MJX] JIT-compiling _jit_step "
              f"(K={self.K}, H={self.H}) — should appear ONCE per run")
        self._jit_step = jax.jit(mppi_update)

        def _batch_sample_phi(qpos0, qvel0, U, key,
                              w_arr, target_sched, stick_t0, sigma_L, u_prev_exec):
            eps = _sample_eps(key, sigma_L, 1.0)
            costs, phis_all, _ = jax.vmap(
                lambda e: rollout_k(qpos0, qvel0, U + e, w_arr, target_sched, stick_t0, u_prev_exec)
            )(eps)
            return phis_all, costs
        self._jit_sample_phi = jax.jit(_batch_sample_phi)

        def _batch_sample_qpos(qpos0, qvel0, U, key,
                               w_arr, target_sched, stick_t0, sigma_L, u_prev_exec):
            eps = _sample_eps(key, sigma_L, 1.0)
            _c, _p, qpos_all = jax.vmap(
                lambda e: rollout_k(qpos0, qvel0, U + e, w_arr, target_sched, stick_t0, u_prev_exec)
            )(eps)
            return qpos_all, _c
        self._jit_sample_qpos = jax.jit(_batch_sample_qpos)

        def _batch_sample_all(qpos0, qvel0, U, key,
                              w_arr, target_sched, stick_t0, sigma_L, u_prev_exec):
            eps = _sample_eps(key, sigma_L, 1.0)
            costs, phis_all, qpos_all = jax.vmap(
                lambda e: rollout_k(qpos0, qvel0, U + e, w_arr, target_sched, stick_t0, u_prev_exec)
            )(eps)
            return qpos_all, phis_all, costs
        self._jit_sample_all = jax.jit(_batch_sample_all)

    def sample_features(self, human_mppi, K=None, T=None, seed=0):
        """K independent stochastic rollouts at the current `self.U` + noise.

        Mirrors KinematicMPPI_CPU.sample_features. Returns
        ``phis : (K, T, nr_run)`` per-sample per-step features in
        ``KEYS_RUN`` order. Used by ``test_mppi_consistency.py`` to
        compare CPU and MJX backends head-to-head.

        Note: `K` is fixed at construction time because the JIT shape is
        baked in. If you pass a different K, the call returns the
        construction-time count with a warning.
        """
        jax = self.jax
        jnp = self.jnp
        if self._jit_step is None:
            self._build_jit()

        if K is not None and int(K) != int(self.K):
            print(f"[KinematicMPPI_MJX.sample_features] requested K={K} "
                  f"but JIT was compiled with K={self.K}; "
                  f"returning {self.K} samples.")
        T_eff = self.H if T is None else min(int(T), self.H)

        model = human_mppi.mj_model
        data  = human_mppi.mj_data
        mujoco.mj_resetData(model, data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                dof = model.jnt_dofadr[jid]
                if i < len(v0):
                    data.qvel[dof] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            data.qpos[qpos_id] = lock_val
            data.qvel[dof_id]  = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            data.qpos[adr:adr+3] = self.stick_traj[0]
            data.qvel[adr:adr+6] = 0.0
        mujoco.mj_forward(model, data)

        qpos0 = jnp.array(data.qpos.copy(), dtype=jnp.float32)
        qvel0 = jnp.array(data.qvel.copy(), dtype=jnp.float32)
        U_jax = jnp.array(self.U, dtype=jnp.float32)
        key   = jax.random.PRNGKey(int(seed))

        w_arr_uniform = np.array([self.w_run.get(k, 0.0) for k in KEYS_RUN],
                                  dtype=np.float32)
        w_arr_uniform = self._softmax_weights(w_arr_uniform)
        w_arr  = jnp.tile(jnp.array(w_arr_uniform), (self.H, 1))
        if self.target_force_profile is not None:
            _pf = np.asarray(self.target_force_profile, dtype=np.float32)
            target_sched = jnp.asarray(_pf[np.clip(np.arange(self.H), 0, len(_pf) - 1)], dtype=jnp.float32)
        else:
            target_sched = jnp.full((self.H,), jnp.float32(self.target_force))
        sigma_L      = jnp.asarray(self._sigma_L(0), dtype=jnp.float32)

        phis_all, costs = self._jit_sample_phi(
            qpos0, qvel0, U_jax, key, w_arr, target_sched,
            jnp.int32(0), sigma_L,
            jnp.zeros(self.nu, dtype=jnp.float32))
        phis_np = np.asarray(phis_all)
        if T_eff < self.H:
            phis_np = phis_np[:, :T_eff, :]
        return phis_np

    def sample_trajectories(self, human_mppi, seed=0, plan_iters=6):
        """K stochastic rollouts at a FROM-START plan + noise, returning each
        sample's FULL qpos trajectory (K, H, nq) and cost (K,) — for rendering
        the MPPI sample cloud in the MuJoCo viewer. plan_iters>0 first converges
        the nominal U from t=0 (so the samples explore the actual stroke, not
        noise around a shifted tail). Same state reset as sample_features."""
        jax = self.jax; jnp = self.jnp
        self._T_press_start = int(getattr(human_mppi, 'T_press_start', 0) or 0)
        if self._jit_step is None:
            self._build_jit()
        model = human_mppi.mj_model; data = human_mppi.mj_data
        mujoco.mj_resetData(model, data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                dof = model.jnt_dofadr[jid]
                if i < len(v0):
                    data.qvel[dof] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            data.qpos[qpos_id] = lock_val; data.qvel[dof_id] = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            data.qpos[adr:adr + 3] = self.stick_traj[0]; data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        qpos0 = jnp.array(data.qpos.copy(), dtype=jnp.float32)
        qvel0 = jnp.array(data.qvel.copy(), dtype=jnp.float32)
        self._dbg_qpos0 = np.asarray(data.qpos.copy())
        self._dbg_qvel0 = np.asarray(data.qvel.copy())
        U_jax = jnp.array(self.U, dtype=jnp.float32)
        key = jax.random.PRNGKey(int(seed))
        w_arr_uniform = np.array([self.w_run.get(k, 0.0) for k in KEYS_RUN], dtype=np.float32)
        w_arr_uniform = self._softmax_weights(w_arr_uniform)
        w_arr = jnp.tile(jnp.array(w_arr_uniform), (self.H, 1))
        if self.target_force_profile is not None:
            _pf = np.asarray(self.target_force_profile, dtype=np.float32)
            target_sched = jnp.asarray(_pf[np.clip(np.arange(self.H), 0, len(_pf) - 1)], dtype=jnp.float32)
        else:
            target_sched = jnp.full((self.H,), jnp.float32(self.target_force))
        sigma_L = jnp.asarray(self._sigma_L(0), dtype=jnp.float32)
        u_prev0 = jnp.zeros(self.nu, dtype=jnp.float32)
        U_plan = U_jax
        for _ in range(int(plan_iters)):
            U_plan, key = self._jit_step(
                qpos0, qvel0, U_plan, key, w_arr, target_sched,
                jnp.int32(0), sigma_L, u_prev0,
                jnp.int32(getattr(self, '_T_press_start', 0) or 0))
        self.U = np.asarray(U_plan)
        qpos_all, costs = self._jit_sample_qpos(
            qpos0, qvel0, U_plan, key, w_arr, target_sched,
            jnp.int32(0), sigma_L, u_prev0)
        return np.asarray(qpos_all), np.asarray(costs)

    def sample_all(self, human_mppi, seed=0, plan_iters=6):
        """K stochastic rollouts returning per-sample (qpos, phi, cost) from the
        SAME noise draw — so posture (qpos) and per-feature φ are EXACTLY paired
        (unlike sample_trajectories + sample_features, which draw different clouds).
        plan_iters>0 first converges the nominal U from t=0. Same reset as the others."""
        jax = self.jax; jnp = self.jnp
        self._T_press_start = int(getattr(human_mppi, 'T_press_start', 0) or 0)
        if self._jit_step is None:
            self._build_jit()
        model = human_mppi.mj_model; data = human_mppi.mj_data
        mujoco.mj_resetData(model, data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                dof = model.jnt_dofadr[jid]
                if i < len(v0):
                    data.qvel[dof] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            data.qpos[qpos_id] = lock_val; data.qvel[dof_id] = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            data.qpos[adr:adr + 3] = self.stick_traj[0]; data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(model, data)
        qpos0 = jnp.array(data.qpos.copy(), dtype=jnp.float32)
        qvel0 = jnp.array(data.qvel.copy(), dtype=jnp.float32)
        U_jax = jnp.array(self.U, dtype=jnp.float32)
        key = jax.random.PRNGKey(int(seed))
        w_arr_uniform = np.array([self.w_run.get(k, 0.0) for k in KEYS_RUN], dtype=np.float32)
        w_arr_uniform = self._softmax_weights(w_arr_uniform)
        w_arr = jnp.tile(jnp.array(w_arr_uniform), (self.H, 1))
        if self.target_force_profile is not None:
            _pf = np.asarray(self.target_force_profile, dtype=np.float32)
            target_sched = jnp.asarray(_pf[np.clip(np.arange(self.H), 0, len(_pf) - 1)], dtype=jnp.float32)
        else:
            target_sched = jnp.full((self.H,), jnp.float32(self.target_force))
        sigma_L = jnp.asarray(self._sigma_L(0), dtype=jnp.float32)
        u_prev0 = jnp.zeros(self.nu, dtype=jnp.float32)
        U_plan = U_jax
        for _ in range(int(plan_iters)):
            U_plan, key = self._jit_step(
                qpos0, qvel0, U_plan, key, w_arr, target_sched,
                jnp.int32(0), sigma_L, u_prev0,
                jnp.int32(getattr(self, '_T_press_start', 0) or 0))
        self.U = np.asarray(U_plan)
        qpos_all, phis_all, costs = self._jit_sample_all(
            qpos0, qvel0, U_plan, key, w_arr, target_sched,
            jnp.int32(0), sigma_L, u_prev0)
        return np.asarray(qpos_all), np.asarray(phis_all), np.asarray(costs)

    def _ensure_limit_L(self):
        """Precompute the per-rail-grid null-space sampling covariance Cholesky for
        LIMIT-AWARE posture exploration. At each grid point s:
            C(s) = nb(s)·diag(margin_j(s)²)·nb(s)ᵀ            (n_null × n_null)
        where margin_j = joint j's distance to its nearer POSITION limit at the reference
        posture q_of_s(s). Sampling c~N(0,C) → joint offset δq=nbᵀc has covariance
        P·diag(margin²)·P (P=null-space projector): joints with headroom (elbow) explore
        MORE, near-limit joints LESS, and δq stays tool-preserving. Trace-normalized to
        n_null·σ₀² so total exploration ENERGY equals the isotropic case (no scale change)."""
        if getattr(self, '_Lnull_grid', None) is not None:
            return
        if self._B_spline is None and self.spline_M is not None:
            self._B_spline = self._cubic_bspline_basis(self.H, self.spline_M)
        nb_grid = np.asarray(self.ts_null_of_s, dtype=np.float64)
        q_grid  = np.asarray(self.ts_q_of_s, dtype=np.float64)
        G, n_null, n_arm = nb_grid.shape
        aq  = np.asarray(self.arm_qposadr)
        lo  = np.asarray(self._q_lim_lo)[aq]; hi = np.asarray(self._q_lim_hi)[aq]
        msk = np.asarray(self._q_lim_mask)[aq].astype(bool)
        sig0 = float(getattr(self, 'ts_null_sigma', 0.08))
        Ls = np.zeros((G, n_null, n_null), np.float32)
        for g in range(G):
            qref = q_grid[g]
            m = np.minimum(qref - lo, hi - qref)
            m = np.where(msk, np.clip(m, 1e-3, None), np.nan)      # unlimited → nan → fill
            fill = np.nanmax(m) if np.isfinite(m).any() else 1.0
            m = np.where(np.isfinite(m), m, fill)
            nb = nb_grid[g]
            C = nb @ ((m ** 2)[:, None] * nb.T)
            tr = np.trace(C)
            if tr > 1e-12:
                C *= (n_null * sig0 ** 2) / tr
            C += 1e-8 * np.eye(n_null)
            Ls[g] = np.linalg.cholesky(C).astype(np.float32)
        self._Lnull_grid = Ls
        B = np.asarray(self._B_spline)
        hh = np.arange(B.shape[0])[:, None]
        self._knot_ctr = ((hh * B).sum(0) / np.maximum(B.sum(0), 1e-9)).astype(np.float64)
        print(f"[KinematicMPPI_MJX] TS_NULL_LIMIT_SCALE: precomputed per-grid null covariance "
              f"(G={G}, n_null={n_null}, σ₀={sig0}) — margin-weighted, tool-preserving, energy-normalized")

    def _build_knot_L(self, t):
        """Per-knot sampling covariance (M, n_ctrl, n_ctrl) for MPC step t: each spline knot
        is colored by C(s) at the rail arc-length s it covers (nominal-pace mapping). Starts
        from a copy of the isotropic self.L (keeps pace/press/discover slots) and overwrites
        ONLY the null block (slots 1..n_null). Rebuilt per replan on the host (~µs)."""
        self._ensure_limit_L()
        M = int(self.spline_M); nn = int(self.ts_n_null)
        snom = np.asarray(self.ts_s_nom); sgrid = np.asarray(self.ts_s_grid); Tn = len(snom)
        out = np.tile(np.asarray(self.L, dtype=np.float32), (M, 1, 1)).copy()
        for m in range(M):
            gframe = int(min(round(t + self._knot_ctr[m]), Tn - 1))
            gi = int(np.clip(np.searchsorted(sgrid, snom[gframe]), 0, len(sgrid) - 1))
            out[m, 1:1 + nn, 1:1 + nn] = self._Lnull_grid[gi][:nn, :nn]
        return out

    def _sigma_L(self, t=0):
        """Sampling covariance for the sampler at MPC step t: the per-knot limit-aware stack
        (M,nc,nc) when TS_NULL_LIMIT_SCALE is on AND spline mode is active, else the static
        isotropic self.L (nc,nc) — byte-identical to before when off."""
        if getattr(self, 'ts_null_limit_scale', False) and self.spline_M is not None:
            return self._build_knot_L(int(t))
        return self.L

    def solve(self, human_mppi, T=None, visualize=False):
        if self._jit_step is None:
            self._build_jit()

        jnp = self.jnp
        model = human_mppi.mj_model
        data  = human_mppi.mj_data
        T = T or human_mppi.T
        self._T_press_start = int(getattr(human_mppi, 'T_press_start', 0) or 0)
        self.reset()

        mujoco.mj_resetData(model, data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                dof = model.jnt_dofadr[jid]
                if i < len(v0):
                    data.qvel[dof] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            data.qpos[qpos_id] = lock_val
            data.qvel[dof_id] = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            data.qpos[adr:adr+3] = self.stick_traj[0]
            data.qvel[adr:adr+6] = 0.0
        mujoco.mj_forward(model, data)

        use_tv = (self._w_run_windows is not None
                  and len(self._w_run_windows) > 1)
        if use_tv:
            n_w = len(self._w_run_windows)
            window_size = max(1, T // n_w)
            w_windows_arr = np.array([
                [wd.get(k, 0.0) for k in KEYS_RUN]
                for wd in self._w_run_windows
            ], dtype=np.float32)
            w_windows_arr = self._softmax_weights(w_windows_arr)
            print(f"[KinematicMPPI_MJX.solve] TV weights active "
                  f"(n_w={n_w}, window_size={window_size}, T={T})")
        else:
            w_arr_uniform = np.array([self.w_run.get(k, 0.0) for k in KEYS_RUN],
                                      dtype=np.float32)
            w_arr_uniform = self._softmax_weights(w_arr_uniform)
            w_arr_horizon = np.tile(w_arr_uniform, (self.H, 1))

        xs, us, force_log = [], [], []
        self._chan_log = []
        self._u_off_prev = None
        q, v = human_mppi._extract_pin_state()
        xs.append(np.concatenate([q, v]))

        if use_tv:
            precomputed_w = np.zeros((T, self.H, self.nr_run), dtype=np.float32)
            for t_ in range(T):
                for h in range(self.H):
                    k_win = min((t_ + h) // window_size, n_w - 1)
                    precomputed_w[t_, h] = w_windows_arr[k_win]

        t_total = 0.0
        self._fc_prev_N = 0.0
        self._press_ema_prev = 0.0
        self._deploy_touched = 0.0
        self._ts_disc_s0 = None; self._ts_disc_lift = None
        print(f"[DBG rail] rail_len={self.rail_len:.4f}m  p_start={np.round(self.p_start,3)}  "
              f"p_end={np.round(self.p_start + self.rail_dir*self.rail_len,3)}")
        self._u_prev_exec = np.zeros(self.nu, dtype=np.float32)
        n_apply = max(1, min(int(getattr(self, 'n_apply', 1)), int(self.H)))
        for t in range(T):
            t0 = time.perf_counter()
            plan_idx = t % n_apply

            if self.target_force_profile is not None:
                idx = np.clip(np.arange(self.H) + t, 0,
                              len(self.target_force_profile) - 1)
                target_arr = np.asarray(
                    self.target_force_profile[idx], dtype=np.float32)
            else:
                target_arr = np.full(self.H, self.target_force, dtype=np.float32)
            T_ps = int(getattr(self, '_T_press_start', 0) or 0)
            if T_ps > 0:
                pre_press = (np.arange(self.H) + t) < T_ps
                target_arr = np.where(pre_press, -1.0, target_arr)
            target_sched = jnp.asarray(target_arr)

            if plan_idx == 0:
                if t > 0 and self.blend_steps > 0:
                    self._u_blend_from = self.U[n_apply - 1][:self.nu].copy()
                if t > 0 and n_apply > 1:
                    self.U = np.concatenate(
                        [self.U[n_apply:], np.repeat(self.U[-1:], n_apply, axis=0)],
                        axis=0)
                if (getattr(self, 'ts_carry_posture', False) and getattr(self, 'task_space', False)
                        and getattr(self, 'ts_null_of_s', None) is not None):
                    _nn_c = int(getattr(self, 'ts_n_null', 0))
                    if _nn_c > 0:
                        _aqc = self.arm_qposadr; _naq = len(_aqc)
                        _sc = float(np.clip(self.ts_s_nom[min(t, len(self.ts_s_nom) - 1)],
                                            float(np.min(self.ts_s_grid)), float(np.max(self.ts_s_grid))))
                        _qb_c = np.array([np.interp(_sc, self.ts_s_grid, self.ts_q_of_s[:, c]) for c in range(_naq)])
                        _nt_c = np.stack([[np.interp(_sc, self.ts_s_grid, self.ts_null_of_s[:, j, c])
                                           for c in range(_naq)] for j in range(_nn_c)])
                        _c_carry = _nt_c @ (np.asarray(data.qpos)[_aqc] - _qb_c)
                        self.U[:, 1:1 + _nn_c] = _c_carry[None, :].astype(self.U.dtype)
                _nbias_tab = getattr(self, 'ts_null_bias_of_s', None)
                _nbias_vec = getattr(self, 'ts_null_bias', None)
                if (_nbias_tab is not None or _nbias_vec is not None) and getattr(self, 'task_space', False):
                    _nnb = int(getattr(self, 'ts_n_null', 0))
                    if _nnb > 0:
                        if _nbias_tab is not None:
                            _sb = float(np.clip(self.ts_s_nom[min(t, len(self.ts_s_nom) - 1)],
                                                float(np.min(self.ts_s_grid)), float(np.max(self.ts_s_grid))))
                            _bv = np.array([np.interp(_sb, self.ts_s_grid, np.asarray(_nbias_tab)[:, j])
                                            for j in range(_nnb)], np.float64)
                        else:
                            _bv = np.zeros(_nnb, np.float64)
                            _m = min(len(_nbias_vec), _nnb)
                            _bv[:_m] = np.asarray(_nbias_vec, np.float64)[:_m]
                        self.U[:, 1:1 + _nnb] = _bv[None, :].astype(self.U.dtype)
                qpos0 = jnp.asarray(data.qpos, dtype=jnp.float32)
                qvel0 = jnp.asarray(data.qvel, dtype=jnp.float32)
                if getattr(self, 'pd_track', False):
                    self._pd_q0_arm = np.asarray(data.qpos)[self.arm_qposadr].copy()
                U_jax = jnp.asarray(self.U)
                if use_tv:
                    w_arr_horizon = precomputed_w[t]
                U_new, self._rng_key = self._jit_step(
                    qpos0, qvel0, U_jax, self._rng_key,
                    jnp.asarray(w_arr_horizon), target_sched,
                    jnp.int32(t),
                    jnp.asarray(self._sigma_L(t)),
                    jnp.asarray(self._u_prev_exec, dtype=jnp.float32),
                    jnp.int32(getattr(self, '_T_press_start', 0) or 0))
                self.U = np.asarray(U_new)

            tau_grav = data.qfrc_bias[self.arm_dofadr]
            tau_press = np.zeros_like(tau_grav)
            if self.force_control and data.ncon > 0:
                c0 = data.contact[0]
                efc_addr0 = int(c0.efc_address)
                if efc_addr0 >= 0:
                    N_now = (abs(float(data.efc_force[efc_addr0]))
                             if c0.dist < 0 else 0.0)
                    J_n = data.efc_J.reshape(data.nefc, self.nv)[efc_addr0]
                    N_des = abs(float(target_arr[0]))
                    dN = (N_now - self._fc_prev_N) / self.dt
                    tau_press_full = self.fc_sign * (
                        self.fc_kp * (N_des - N_now) - self.fc_kd * dN) * J_n
                    tau_press = tau_press_full[self.arm_dofadr]
                    self._fc_prev_N = N_now
            tau_close = np.zeros_like(tau_grav)
            tau_brake = np.zeros_like(tau_grav)
            if (self.force_control
                    and (self.fc_kp_close != 0.0 or self.fc_kv != 0.0)
                    and self.rock_geom_id >= 0 and self.stick_body_id >= 0):
                rp_c = data.geom_xpos[self.rock_geom_id]
                sc_c = data.xpos[self.stick_body_id]
                sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                tp_c = np.clip(np.dot(rp_c - sc_c, sa_c),
                               -self.stick_half_len, self.stick_half_len)
                nearest_c = sc_c + tp_c * sa_c
                dvec = rp_c - nearest_c
                dn = np.linalg.norm(dvec)
                n_geo = dvec / max(dn, 1e-9)
                gap_geo = dn - self.stick_radius - 0.02
                jacp = np.zeros((3, self.nv))
                mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                _gc_max = float(getattr(self, 'gap_close_max', 1.0))
                if self.fc_kp_close != 0.0 and 0.0 < gap_geo < _gc_max:
                    J_n_geom = n_geo @ jacp
                    tau_close = (self.fc_sign_close * self.fc_kp_close
                                 * gap_geo * J_n_geom)[self.arm_dofadr]
                if self.fc_kv != 0.0:
                    J_rail = self.rail_dir @ jacp
                    v_rail_now = (jacp @ data.qvel) @ self.rail_dir
                    tau_brake = (-self.fc_kv
                                 * (v_rail_now - self.target_rail_vel)
                                 * J_rail)[self.arm_dofadr]
            tau_press_amp = np.zeros_like(tau_grav)
            if ((getattr(self, 'force_press', False) or getattr(self, 'depth_impedance', False))
                    and self.press_control
                    and self.rock_geom_id >= 0 and self.stick_body_id >= 0):
                rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9)
                gap_geo_d = dn - self.stick_radius - 0.02
                jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                J_n_geom_d = (dvec / dn) @ jacp
                if getattr(self, 'force_press', False):
                    _fd = float(getattr(self, 'press_force_des', 26.0))
                    if self.press_control:
                        _fd = max(_fd + float(self.U[plan_idx][self.nu]), 0.0)
                    if getattr(self, 'ts_discover', False) and (1.0 - t / max(float(getattr(self, 'ts_n_app', 8.0)), 1.0)) > 0.0:
                        _fd = 0.0
                    tau_press_amp = (-_fd * J_n_geom_d)[self.arm_dofadr]
                else:
                    _dk = float(getattr(self, 'depth_kn', 3000.0)); _dt = float(getattr(self, 'depth_target', -0.0015))
                    tau_press_amp = (_dk * (_dt - gap_geo_d) * J_n_geom_d)[self.arm_dofadr]
            elif self.press_control:
                press_amp0 = float(self.U[plan_idx][self.nu])
                press_amp0 = (self.press_ema * press_amp0
                              + (1.0 - self.press_ema) * self._press_ema_prev)
                self._press_ema_prev = press_amp0
                if self.virtual_target and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp_c = data.geom_xpos[self.rock_geom_id]
                    sc_c = data.xpos[self.stick_body_id]
                    sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                    dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9)
                    n_geo = dvec / dn
                    jacp = np.zeros((3, self.nv))
                    mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                    J_n_geom = n_geo @ jacp
                    gap_vt = (float(data.contact[0].dist) if data.ncon > 0
                              else dn - self.stick_radius - 0.02)
                    vn_vt = float(J_n_geom @ data.qvel)
                    Fmag_vt = np.clip(self.vt_kp * (gap_vt + press_amp0) - self.vt_kd * vn_vt,
                                      -120.0, 120.0)
                    if gap_vt < 0.015:
                        tau_press_amp = (Fmag_vt * J_n_geom)[self.arm_dofadr]
                    self._vt_proj = (J_n_geom[self.arm_dofadr].copy(), bool(gap_vt < 0.015))
                    in_contact = False
                else:
                    in_contact = (data.ncon > 0 and int(data.contact[0].efc_address) >= 0
                                  and data.contact[0].dist < 0)
                if in_contact:
                    efc_addr_p = int(data.contact[0].efc_address)
                    J_n_p = data.efc_J.reshape(data.nefc, self.nv)[efc_addr_p]
                    tau_press_amp = (press_amp0 * J_n_p)[self.arm_dofadr]
                elif (not self.virtual_target) and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp_c = data.geom_xpos[self.rock_geom_id]
                    sc_c = data.xpos[self.stick_body_id]
                    sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = np.clip(np.dot(rp_c - sc_c, sa_c),
                                   -self.stick_half_len, self.stick_half_len)
                    dvec = rp_c - (sc_c + tp_c * sa_c)
                    n_geo = dvec / max(np.linalg.norm(dvec), 1e-9)
                    jacp = np.zeros((3, self.nv))
                    mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                    tau_press_amp = (press_amp0 * (n_geo @ jacp))[self.arm_dofadr]
            _latch_gnm = None
            _tau_rail_latch = np.zeros_like(tau_grav)
            _dbg_vrail = -999.0
            _dbg_ang = -1.0
            if getattr(self, 'contact_latch', False) and self.press_control \
                    and not getattr(self, 'rail_force', False) \
                    and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9)
                gap_geo_l = dn - self.stick_radius - 0.02
                jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                J_n_geom_l = (dvec / dn) @ jacp
                _touch_d = (gap_geo_l < float(getattr(self, 'latch_gap', 0.001))) or (
                    data.ncon > 0 and float(data.contact[0].dist) < 0)
                if _touch_d:
                    self._deploy_touched = 1.0
                if self._deploy_touched > 0.5:
                    _dk = float(getattr(self, 'depth_kn', 3000.0))
                    _dt = float(getattr(self, 'depth_target', -0.0015))
                    tau_press_amp = tau_press_amp + (_dk * (_dt - gap_geo_l) * J_n_geom_l)[self.arm_dofadr]
                    _latch_gnm = J_n_geom_l[self.arm_dofadr].copy()
                    n_geo_l = dvec / dn
                    railg = (np.eye(3) - np.outer(n_geo_l, n_geo_l)) @ self.rail_dir
                    railg = railg / max(np.linalg.norm(railg), 1e-9)
                    v_rail_now = float(np.dot(jacp @ np.asarray(data.qvel), railg))
                    _lvr = float(getattr(self, 'latch_vrail', self.target_rail_vel))
                    _lkd = float(getattr(self, 'latch_rail_kd', 40.0))
                    _lff = float(getattr(self, 'latch_rail_ff', 0.0))
                    _tau_rail_latch = jacp[:, self.arm_dofadr].T @ ((_lkd * (_lvr - v_rail_now) + _lff) * railg)
                    _dbg_vrail = v_rail_now
                    _dbg_ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(railg, self.rail_dir)), 0, 1))))
            U_apply = self.U[plan_idx][:self.nu]
            if self.virtual_target and self._vt_proj is not None and self._vt_proj[1]:
                _gnm = np.asarray(self._vt_proj[0]) * self._vt_free_mask
                _g2v = float(_gnm @ _gnm) + 1e-9
                U_apply = U_apply - (float(U_apply @ _gnm) / _g2v) * _gnm
            if _latch_gnm is not None:
                _gnl = _latch_gnm * self._vt_free_mask
                _g2l = float(_gnl @ _gnl) + 1e-9
                U_apply = U_apply - (float(U_apply @ _gnl) / _g2l) * _gnl
            if (self.blend_steps > 0 and t > 0 and plan_idx < self.blend_steps
                    and self._u_blend_from is not None):
                w = (plan_idx + 1) / (self.blend_steps + 1.0)
                U_apply = (1.0 - w) * self._u_blend_from + w * U_apply
            _du_max = float(getattr(self, 'exec_du_max', np.inf))
            _lp_a = float(getattr(self, 'exec_lp_alpha', 1.0))
            if np.isfinite(_du_max) or _lp_a < 1.0:
                if self._u_off_prev is None:
                    self._u_off_prev = U_apply.copy()
                U_off = _lp_a * U_apply + (1.0 - _lp_a) * self._u_off_prev
                U_apply = self._u_off_prev + np.clip(U_off - self._u_off_prev,
                                                     -_du_max, _du_max)
                self._u_off_prev = U_apply.copy()
            if getattr(self, 'task_space', False):
                _tt = min(t, len(self.ts_s_nom) - 1)
                _nn = int(self.ts_null_of_s.shape[1]); _na = int(self.ts_q_of_s.shape[1])
                _pace = float(self.U[plan_idx][0]); _ncoef = np.asarray(self.U[plan_idx][1:1 + _nn])
                if getattr(self, 'ts_pace_fwd', False):
                    _pace = max(_pace, 0.0)
                _s0d = 0.0
                if getattr(self, 'ts_discover', False):
                    if getattr(self, '_ts_disc_s0', None) is None:
                        self._ts_disc_s0 = float(np.clip(self.U[0][1 + _nn] * float(getattr(self, 'ts_s0_scale', 0.03)),
                                                         -float(getattr(self, 'ts_s0_max', 0.06)),
                                                         float(getattr(self, 'ts_s0_max', 0.06))))
                        self._ts_disc_lift = float(np.clip(self.U[0][2 + _nn] * float(getattr(self, 'ts_lift_scale', 0.01)),
                                                           0.0, float(getattr(self, 'ts_lift_max', 0.04))))
                    _s0d = self._ts_disc_s0
                _st = float(np.clip(self.ts_s_nom[_tt] + _s0d + _pace * float(getattr(self, 'ts_pace_scale', 0.03)),
                                    float(np.min(self.ts_s_grid)), float(np.max(self.ts_s_grid))))
                _qbase = np.array([np.interp(_st, self.ts_s_grid, self.ts_q_of_s[:, c]) for c in range(_na)])
                _nullt = np.stack([[np.interp(_st, self.ts_s_grid, self.ts_null_of_s[:, j, c])
                                    for c in range(_na)] for j in range(_nn)])
                _qref_ema = float(getattr(self, 'ts_qref_ema', 1.0))
                if t == 0:
                    self._ts_qref_prev = None
                    self._ts_pdev_prev = None
                _warm_host = (getattr(self, 'ts_qref_warm', False)
                              and (getattr(self, 'ts_reanchor_exact', False)
                                   or getattr(self, 'ts_posture_sample', False)))
                if _warm_host:
                    if getattr(self, '_ts_fk_data', None) is None:
                        self._ts_fk_data = mujoco.MjData(model)
                    _fkd = self._ts_fk_data; _aq = self.arm_qposadr
                    _lo = self._q_lim_lo[_aq]; _hi = self._q_lim_hi[_aq]
                    if getattr(self, 'ts_posture_sample', False):
                        q_ref = np.clip(data.qpos[_aq] + _ncoef @ _nullt, _lo, _hi)
                    else:
                        q_ref = _qbase + _ncoef @ _nullt
                    if getattr(self, '_ts_pdev_prev', None) is None:
                        self._ts_pdev_prev = (data.qpos[_aq] - _qbase).copy()
                    _ema = _qref_ema if _qref_ema < 1.0 else 1.0
                    _dev_seed = _ema * (q_ref - _qbase) + (1.0 - _ema) * self._ts_pdev_prev
                    q_ref = np.clip(_qbase + _dev_seed, _lo, _hi)
                    _Wi = np.asarray(getattr(self, 'ts_proj_winv', np.ones(_na)), dtype=float)
                    _rbid = self.rock_body_id
                    def _fk_jac(_qarm):
                        _fkd.qpos[:] = data.qpos; _fkd.qpos[_aq] = _qarm
                        mujoco.mj_kinematics(model, _fkd); mujoco.mj_comPos(model, _fkd)
                        _p = _fkd.geom_xpos[self.rock_geom_id].copy()
                        _J = np.zeros((3, self.nv)); mujoco.mj_jac(model, _fkd, _J, None, _p, _rbid)
                        return _p, _J[:, self.arm_dofadr].T
                    _p_tgt, _ = _fk_jac(_qbase); _q_lin = q_ref.copy()
                    for _ in range(int(getattr(self, 'ts_reanchor_iters', 2))):
                        _p_now, _Ja = _fk_jac(q_ref)
                        _JaW = _Wi[:, None] * _Ja
                        _dq = np.clip(_JaW @ np.linalg.inv(_Ja.T @ _JaW + 1e-4 * np.eye(3)) @ (_p_now - _p_tgt),
                                      -0.2, 0.2)
                        q_ref = np.clip(q_ref - _dq, _lo, _hi)
                    if not np.all(np.isfinite(q_ref)):
                        q_ref = _q_lin
                    self._ts_pdev_prev = (q_ref - _qbase).copy()
                    if getattr(self, 'ts_discover', False):
                        _appf_d = max(0.0, 1.0 - t / max(float(getattr(self, 'ts_n_app', 8.0)), 1.0))
                        if _appf_d > 0.0 and getattr(self, 'ts_lift_of_s', None) is not None:
                            _liftt = np.array([np.interp(_st, self.ts_s_grid, self.ts_lift_of_s[:, c])
                                               for c in range(_na)])
                            q_ref = q_ref + (self._ts_disc_lift * _appf_d) * _liftt
                else:
                    q_ref = _qbase + _ncoef @ _nullt
                    if getattr(self, 'ts_reanchor_exact', False):
                        if getattr(self, '_ts_fk_data', None) is None:
                            self._ts_fk_data = mujoco.MjData(model)
                        _fkd = self._ts_fk_data; _aq = self.arm_qposadr
                        _reJd = np.stack([[np.interp(_st, self.ts_s_grid, self.ts_reanchor_JpT[:, c, d])
                                           for d in range(3)] for c in range(_na)])
                        def _fk_rock(_qarm):
                            _fkd.qpos[:] = data.qpos; _fkd.qpos[_aq] = _qarm
                            mujoco.mj_kinematics(model, _fkd)
                            return _fkd.geom_xpos[self.rock_geom_id].copy()
                        _p_tgt = _fk_rock(_qbase); _q_lin = q_ref.copy()
                        for _ in range(int(getattr(self, 'ts_reanchor_iters', 2))):
                            _dq = np.clip(_reJd @ (_fk_rock(q_ref) - _p_tgt), -0.2, 0.2)
                            q_ref = np.clip(q_ref - _dq, self._q_lim_lo[_aq], self._q_lim_hi[_aq])
                        if not np.all(np.isfinite(q_ref)):
                            q_ref = _q_lin
                    elif getattr(self, 'ts_reanchor', False):
                        _reBd = np.stack([[[np.interp(_st, self.ts_s_grid, self.ts_reanchor_B[:, d, i, j])
                                            for j in range(_nn)] for i in range(_nn)] for d in range(3)])
                        _driftd = 0.5 * np.einsum('i,j,dij->d', _ncoef, _ncoef, _reBd)
                        _reJd = np.stack([[np.interp(_st, self.ts_s_grid, self.ts_reanchor_JpT[:, c, d])
                                           for d in range(3)] for c in range(_na)])
                        q_ref = q_ref - _reJd @ _driftd
                    if getattr(self, 'ts_discover', False):
                        _appf_d = max(0.0, 1.0 - t / max(float(getattr(self, 'ts_n_app', 8.0)), 1.0))
                        if _appf_d > 0.0 and getattr(self, 'ts_lift_of_s', None) is not None:
                            _liftt = np.array([np.interp(_st, self.ts_s_grid, self.ts_lift_of_s[:, c]) for c in range(_na)])
                            q_ref = q_ref + (self._ts_disc_lift * _appf_d) * _liftt
                    if _qref_ema < 1.0:
                        _qdev = q_ref - _qbase
                        if getattr(self, '_ts_qref_prev', None) is not None:
                            _qdev = _qref_ema * _qdev + (1.0 - _qref_ema) * self._ts_qref_prev
                        self._ts_qref_prev = _qdev.copy()
                        q_ref = _qbase + _qdev
                _Mf_d = np.zeros((self.nv, self.nv)); mujoco.mj_fullM(model, _Mf_d, data.qM)
                _Marm_d = _Mf_d[np.ix_(self.arm_dofadr, self.arm_dofadr)]
                _qdd_d = self._qddot_ff[min(t, self._qddot_ff.shape[0] - 1)]
                _qd_d = self._qdot_ff[min(t, self._qdot_ff.shape[0] - 1)]
                _c_d = np.asarray(data.qfrc_constraint)[self.arm_dofadr]
                _ffrac = float(getattr(self, 'ts_ff_frac', 0.5))
                _ffcap_d = _ffrac * np.asarray(self.u_max)[:self.nu]
                _ff_d = np.clip(_Marm_d @ _qdd_d, -_ffcap_d, _ffcap_d)
                _nks = float(getattr(self, 'ts_null_kp_scale', 1.0))
                if getattr(self, 'ts_task_pd', False) and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                    sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                    dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9); n_geo = dvec / dn
                    _gap_geo = dn - self.stick_radius - 0.02
                    jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                    _Pt = np.eye(3) - np.outer(n_geo, n_geo)
                    _railt = _Pt @ self.rail_dir; _railt = _railt / max(np.linalg.norm(_railt), 1e-9)
                    _railt = _railt * np.sign(np.dot(_railt, self.rail_dir) + 1e-9)
                    _latd = np.cross(n_geo, _railt); _latd = _latd / max(np.linalg.norm(_latd), 1e-9)
                    _Ja_d = jacp[:, self.arm_dofadr]
                    _vtool = jacp @ np.asarray(data.qvel)
                    _vref = _Ja_d @ _qd_d
                    _frail_s = float(np.clip(float(getattr(self, "ts_kd_railf", 40.0)) * (
                        float(np.dot(_vref, _railt)) - float(np.dot(_vtool, _railt))), -60.0, 60.0))
                    _Frail = _frail_s * _railt
                    _latoff = float(np.dot(rp_c - np.asarray(self.p_start), _latd))
                    _Flat = (float(getattr(self, 'ts_kp_lat', 2500.0)) * (-_latoff)
                             - float(getattr(self, 'ts_kd_lat', 60.0)) * float(np.dot(_vtool, _latd))) * _latd
                    _pen_des = float(np.clip(float(getattr(self, 'ts_pen_des', 0.003))
                                     + (float(self.U[plan_idx][self.nu]) * float(getattr(self, 'ts_pen_scale', 0.0003))
                                        if self.press_control else 0.0),
                                     0.0, float(getattr(self, 'ts_pen_max', 0.02))))
                    _vn = float(np.dot(_vtool, n_geo))
                    _spring = float(getattr(self, 'ts_pen_kp', 8000.0)) * (-_pen_des - _gap_geo)
                    _damp = -(float(getattr(self, 'ts_pen_kd', 120.0)) * _vn)
                    _dfrac = float(getattr(self, 'ts_pen_damp_frac', 0.0))
                    if _dfrac > 0.0:
                        _dcap = _dfrac * abs(_spring)
                        _damp = float(np.clip(_damp, -_dcap, _dcap))
                    _fnmax_d = float(getattr(self, 'ts_fnmax', 200.0))
                    _coef = float(np.clip(_spring + _damp, -_fnmax_d, _fnmax_d))
                    _Fn = _coef * n_geo
                    U_apply = (_ff_d + _Ja_d.T @ (_Frail + _Flat + _Fn)
                               + _Marm_d @ (self.ts_KD * (_qd_d - np.asarray(data.qvel)[self.arm_dofadr]))
                               - float(getattr(self, 'ts_ccancel', 1.0)) * _c_d).astype(U_apply.dtype)
                    _nkp = np.asarray(getattr(self, 'ts_null_kp', 0.0), dtype=float)
                    if np.any(_nkp > 0.0):
                        _e_null_d = _nullt.T @ (_nullt @ (q_ref - np.asarray(data.qpos)[self.arm_qposadr]))
                        U_apply = (U_apply + _Marm_d @ (_nkp * _e_null_d)).astype(U_apply.dtype)
                else:
                    _etrk = q_ref - np.asarray(data.qpos)[self.arm_qposadr]
                    if _nks != 1.0:
                        _etrk = _etrk - (1.0 - _nks) * (_nullt.T @ (_nullt @ _etrk))
                    U_apply = (_ff_d + _Marm_d @ (self.ts_KP * _etrk
                                                  + self.ts_KD * (_qd_d - np.asarray(data.qvel)[self.arm_dofadr]))
                               - float(getattr(self, 'ts_ccancel', 1.0)) * _c_d).astype(U_apply.dtype)
            elif getattr(self, 'pd_track', False):
                if (getattr(self, 'confine_motion', False)
                        and self.rock_geom_id >= 0 and self.stick_body_id >= 0):
                    rp_c = data.geom_xpos[self.rock_geom_id]
                    sc_c = data.xpos[self.stick_body_id]
                    sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = np.clip(np.dot(rp_c - sc_c, sa_c),
                                   -self.stick_half_len, self.stick_half_len)
                    dvec = rp_c - (sc_c + tp_c * sa_c)
                    dn = max(np.linalg.norm(dvec), 1e-9)
                    if getattr(self, 'force_press', False) or (dn - self.stick_radius - 0.02 < 0.015):
                        jacp = np.zeros((3, self.nv))
                        mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                        _gt = ((dvec / dn) @ jacp)[self.arm_dofadr] * self._vt_free_mask
                        _g2 = float(_gt @ _gt) + 1e-9
                        U_apply = U_apply - (float(U_apply @ _gt) / _g2) * _gt
                _ff = 0.0
                if getattr(self, 'id_feedforward', False) and getattr(self, '_u_feedforward_id', None) is not None:
                    _uid = np.asarray(self._u_feedforward_id)
                    _ff = _uid[min(t, _uid.shape[0] - 1), :self.nu] - tau_grav
                elif getattr(self, 'pd_feedforward', False) and getattr(self, '_qddot_ff', None) is not None:
                    _qff = self._qddot_ff[min(t, self._qddot_ff.shape[0] - 1)]
                    _Mfull = np.zeros((self.nv, self.nv))
                    mujoco.mj_fullM(model, _Mfull, data.qM)
                    _ff = _Mfull[np.ix_(self.arm_dofadr, self.arm_dofadr)] @ _qff
                q_ref = self._pd_q0_arm + U_apply
                U_apply = (_ff
                           + self._pd_kp * (q_ref - np.asarray(data.qpos)[self.arm_qposadr])
                           - self._pd_kd * np.asarray(data.qvel)[self.arm_dofadr]
                           ).astype(U_apply.dtype)
                if getattr(self, 'pd_free_normal', False) and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                    sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                    dvec = rp_c - (sc_c + tp_c * sa_c); n_geo = dvec / max(np.linalg.norm(dvec), 1e-9)
                    jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                    _gn = (n_geo @ jacp)[self.arm_dofadr]; _gn2 = float(_gn @ _gn) + 1e-9
                    U_apply = (U_apply - (float(U_apply @ _gn) / _gn2) * _gn).astype(U_apply.dtype)
            tau_task = np.zeros_like(tau_grav)
            if (getattr(self, 'task_space_control', False)
                    and self.rock_geom_id >= 0 and self.stick_body_id >= 0):
                rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9); n_geo = dvec / dn
                jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                Ja = jacp[:, self.arm_dofadr]
                v_ball = jacp @ np.asarray(data.qvel)
                Pt = np.eye(3) - np.outer(n_geo, n_geo)
                _railg = Pt @ sa_c; _railg = _railg / max(np.linalg.norm(_railg), 1e-9)
                _railg = _railg * np.sign(np.dot(_railg, self.rail_dir) + 1e-9)
                _vrail = float(getattr(self, 'ts_vrail', 0.46))
                _vr_ramp_steps = float(getattr(self, 'ts_vrail_ramp', 0.0))
                if _vr_ramp_steps > 0.5:
                    _vrail = _vrail * float(np.clip(t / max(_vr_ramp_steps, 1.0), 0.0, 1.0))
                _latd = np.cross(n_geo, _railg); _latd = _latd / max(np.linalg.norm(_latd), 1e-9)
                _lat_off = float(np.dot(rp_c - np.asarray(self.p_start), _latd))
                _F_rail = float(getattr(self, 'ts_kd_rail', 40.0)) * (_vrail - float(np.dot(v_ball, _railg))) * _railg
                _F_lat = (float(getattr(self, 'ts_kp', 600.0)) * (-_lat_off)
                          - float(getattr(self, 'ts_kd', 60.0)) * float(np.dot(v_ball, _latd))) * _latd
                F_t = _F_rail + _F_lat
                if bool(getattr(self, 'ts_impedance_normal', False)):
                    _gap_geo = dn - self.stick_radius - 0.02
                    _pen_des = float(np.clip(float(getattr(self, 'ts_pen_des', 0.003))
                                    + (float(self.U[plan_idx][self.nu]) * float(getattr(self, 'ts_pen_scale', 0.0003))
                                       if self.press_control else 0.0),
                                    0.0, float(getattr(self, 'ts_pen_max', 0.02))))
                    _vn = float(np.dot(v_ball, n_geo))
                    _coef = (float(getattr(self, 'ts_pen_kp', 8000.0)) * (-_pen_des - _gap_geo)
                             - float(getattr(self, 'ts_pen_kd', 120.0)) * _vn)
                    _coef = float(np.clip(_coef, -float(getattr(self, 'ts_fnmax', 200.0)),
                                          float(getattr(self, 'ts_fnmax', 200.0))))
                    F_n = _coef * n_geo
                else:
                    _fp = np.asarray(getattr(self, 'ts_fdes_profile', [float(getattr(self, 'ts_fdes', 30.0))]))
                    _fdes_t = float(_fp[min(t, len(_fp) - 1)])
                    if self.press_control:
                        _fdes_t += float(self.U[plan_idx][self.nu])
                    F_n = -max(_fdes_t, float(getattr(self, 'ts_fdes_floor', 15.0))) * n_geo
                tau_task = Ja.T @ (F_t + F_n)
                if __import__('os').environ.get('IMPDBG') and (t % 10 == 0):
                    _gg = dn - self.stick_radius - 0.02
                    _kp_dbg = float(getattr(self, 'ts_pen_kp', 8000.0))
                    print(f"[impdbg t={t:3d}] gap_geo={_gg:+.4f} pen_des={_pen_des:+.4f} vn={_vn:+.3f} "
                          f"coef={_coef:+.1f} (kp*(−pen−gap)−kd*vn, kp={_kp_dbg:.0f}) |F_n|={np.linalg.norm(F_n):6.1f} "
                          f"|F_lat|={np.linalg.norm(_F_lat):5.1f} |tau_task|={np.linalg.norm(tau_task):6.1f} "
                          f"|Ja^T·n|={np.linalg.norm(Ja.T@n_geo):.3f} rock_stick_dn={dn:.4f} stick_R={self.stick_radius:.4f}")
                _tp_kp = float(getattr(self, 'ts_post_kp', 0.0))
                if _tp_kp != 0.0:
                    Nproj_h = np.eye(Ja.shape[1]) - np.linalg.pinv(Ja) @ Ja
                    tau_task = tau_task + Nproj_h @ (
                        _tp_kp * np.asarray(self.U[plan_idx][:self.nu])
                        - float(getattr(self, 'ts_post_kd', 6.0)) * np.asarray(data.qvel)[self.arm_dofadr])
            tau_rf = np.zeros_like(tau_grav)
            if getattr(self, 'rail_force', False) \
                    and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                rp_c = data.geom_xpos[self.rock_geom_id]; sc_c = data.xpos[self.stick_body_id]
                sa_c = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                tp_c = np.clip(np.dot(rp_c - sc_c, sa_c), -self.stick_half_len, self.stick_half_len)
                dvec = rp_c - (sc_c + tp_c * sa_c); dn = max(np.linalg.norm(dvec), 1e-9)
                n_geo = dvec / dn
                jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, jacp, None, rp_c, self.rock_body_id)
                Ja_rf = jacp[:, self.arm_dofadr]
                vball_rf = jacp @ np.asarray(data.qvel)
                Pt_rf = np.eye(3) - np.outer(n_geo, n_geo)
                railg_rf = Pt_rf @ sa_c
                railg_rf = railg_rf / max(np.linalg.norm(railg_rf), 1e-9)
                railg_rf = railg_rf * np.sign(np.dot(railg_rf, self.rail_dir) + 1e-9)
                latd_rf = np.cross(n_geo, railg_rf); latd_rf = latd_rf / max(np.linalg.norm(latd_rf), 1e-9)
                _sdot = float(np.dot(vball_rf, railg_rf)); _vn = float(np.dot(vball_rf, n_geo)); _vl = float(np.dot(vball_rf, latd_rf))
                _latoff = float(np.dot(rp_c - np.asarray(self.p_start), latd_rf))
                _rrad = float(model.geom_size[self.rock_geom_id][0])
                _truegap = dn - self.stick_radius - _rrad
                _vtgt = float(getattr(self, 'rf_vbase', self.target_rail_vel))
                if getattr(self, 'rail_walk_pace', False):
                    _vtgt += float(self.U[plan_idx][self._pace_idx])
                _Fpress = float(self.U[plan_idx][self.nu]) if self.press_control else 0.0
                _rf_kv = float(getattr(self, 'rf_kv', 400.0)); _rf_fmax = float(getattr(self, 'rf_fmax', 120.0))
                _rf_kpn = float(getattr(self, 'rf_kpn', 8000.0)); _rf_kdn = float(getattr(self, 'rf_kdn', 120.0))
                _rf_fnmax = float(getattr(self, 'rf_fnmax', 200.0))
                _rf_near = float(getattr(self, 'rf_near', 0.015)); _rf_fapp = float(getattr(self, 'rf_fapp', 25.0))
                _rf_kplat = float(getattr(self, 'rf_kplat', 3000.0)); _rf_kdlat = float(getattr(self, 'rf_kdlat', 40.0))
                _rf_pen = float(getattr(self, 'rf_pen', 0.003))
                _Ft = np.clip(_rf_kv * (_vtgt - _sdot), -_rf_fmax, _rf_fmax)
                _farcap = _rf_fnmax if _truegap < _rf_near else _rf_fapp
                _Fn = np.clip(_rf_kpn * (_truegap + _rf_pen) - _rf_kdn * _vn + _Fpress, 0.0, _farcap)
                _Fl = -_rf_kplat * _latoff - _rf_kdlat * _vl
                F_rf = _Ft * railg_rf + _Fn * (-n_geo) + _Fl * latd_rf
                tau_rf = Ja_rf.T @ F_rf
                _rf_postkp = float(getattr(self, 'rail_walk_post_kp', 0.0))
                if _rf_postkp != 0.0:
                    Nproj_rf = np.eye(Ja_rf.shape[1]) - np.linalg.pinv(Ja_rf) @ Ja_rf
                    tau_rf = tau_rf + Nproj_rf @ (
                        _rf_postkp * np.asarray(self.U[plan_idx][:self.nu])
                        - float(getattr(self, 'rail_walk_post_kd', 4.0)) * np.asarray(data.qvel)[self.arm_dofadr])
            if getattr(self, 'rail_force', False):
                u = np.clip(tau_grav + tau_rf, self.u_min[:self.nu], self.u_max[:self.nu])
            elif getattr(self, 'task_space_control', False):
                u = np.clip(tau_grav + tau_task, self.u_min[:self.nu], self.u_max[:self.nu])
            else:
                u = np.clip(
                    tau_grav + U_apply + tau_press + tau_close
                    + tau_brake + tau_press_amp + _tau_rail_latch,
                    self.u_min[:self.nu], self.u_max[:self.nu])
            if t == 0 and (getattr(self, '_nanchk', False) or __import__('os').environ.get('NANCHK')):
                for _nm in ['tau_grav', 'U_apply', 'tau_press', 'tau_close', 'tau_brake',
                            'tau_press_amp', '_tau_rail_latch', 'tau_task', 'tau_rf']:
                    _v = locals().get(_nm, None)
                    if _v is None:
                        continue
                    _a = np.asarray(_v, dtype=float)
                    if np.isnan(_a).any():
                        print(f"[NANCHK t0] {_nm} HAS NAN at idx {list(np.where(np.isnan(_a.ravel()))[0][:5])}")
                    else:
                        print(f"[NANCHK t0] {_nm} ok (|.|={np.linalg.norm(_a):.3g})")
            data.ctrl[:] = u
            us.append(u.copy())
            if getattr(self, 'debug_channels', False):
                _qref_c = locals().get('q_ref', None)
                self._chan_log.append((
                    np.asarray(tau_grav[:self.nu], dtype=float).copy(),
                    np.asarray(U_apply[:self.nu], dtype=float).copy(),
                    np.asarray(tau_press_amp[:self.nu], dtype=float).copy(),
                    int(data.ncon),
                    (np.asarray(_qref_c, dtype=float).copy() if _qref_c is not None else None),
                ))
            self._u_prev_exec = u[:self.nu].astype(np.float32).copy()
            if self.stick_follow and self.has_stick:
                adr = self.stick_qposadr
                sidx = min(t, self.stick_traj.shape[0] - 1)
                data.qpos[adr:adr + 3] = self.stick_traj[sidx]
                data.qvel[adr:adr + 6] = 0.0
            _site_pre = (data.site_xpos[self.contact_site_id].copy()
                         if self.contact_site_id >= 0 else np.zeros(3))
            _pre_r = float(np.dot(_site_pre, self.rail_dir))
            self._dbg_between = _pre_r - getattr(self, '_end_r_prev', _pre_r)
            _jr = np.zeros((3, self.nv))
            if self.contact_site_id >= 0:
                mujoco.mj_jacSite(model, data, _jr, None, self.contact_site_id)
            self._dbg_vpre = float(np.dot(_jr @ np.asarray(data.qvel), self.rail_dir))
            _mid_pre_r = float(np.dot(data.site_xpos[self.contact_site_id], self.rail_dir))
            for _ in range(int(getattr(self, '_rollout_nsub', 1))):
                mujoco.mj_step(model, data)
            self._dbg_step_dr = float(np.dot(data.site_xpos[self.contact_site_id], self.rail_dir)) - _mid_pre_r
            self._dbg_vpost = float(np.dot(_jr @ np.asarray(data.qvel), self.rail_dir))
            if getattr(self, 'contact_latch', False) and getattr(self, 'latch_project', False) \
                    and self._deploy_touched > 0.5 and self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                _rp = data.geom_xpos[self.rock_geom_id]; _sc = data.xpos[self.stick_body_id]
                _sa = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                _tp = np.clip(np.dot(_rp - _sc, _sa), -self.stick_half_len, self.stick_half_len)
                _dv = _rp - (_sc + _tp * _sa); _dn = max(np.linalg.norm(_dv), 1e-9); _n = _dv / _dn
                _rrad = float(model.geom_size[self.rock_geom_id][0])
                _gap = _dn - self.stick_radius - _rrad
                _ftgt = (float(self.target_force_profile[min(t, len(self.target_force_profile) - 1)])
                         if self.target_force_profile is not None else float(self.target_force))
                if getattr(self, 'rail_walk', False) and self.press_control:
                    _ftgt = _ftgt + float(self.U[plan_idx][self.nu])
                _dz = max(_ftgt, 0.0) / float(getattr(self, 'latch_k', 4000.0))
                _jacp = np.zeros((3, self.nv)); mujoco.mj_jac(model, data, _jacp, None, _rp, self.rock_body_id)
                _Ja = _jacp[:, self.arm_dofadr] * self._vt_free_mask[None, :]
                if getattr(self, 'rail_walk', False):
                    _qa = np.asarray(data.qpos)[self.arm_qposadr]
                    _lo = self._q_lim_lo[self.arm_qposadr]; _hi = self._q_lim_hi[self.arm_qposadr]
                    _hlf = 0.5 * (_hi - _lo); _mid = 0.5 * (_hi + _lo)
                    _mrg = max(float(getattr(self, 'rail_walk_margin', 0.15)), 1e-6)
                    _prx = np.clip((np.abs(_qa - _mid) / np.maximum(_hlf, 1e-6) - (1.0 - _mrg)) / _mrg, 0.0, 1.0)
                    _Winv = 1.0 / (1.0 + float(getattr(self, 'rail_walk_wlim', 4.0)) * _prx * _prx)
                    # DEPLOY damping is tiny (host MuJoCo-C never nans, unlike the GPU rollout):
                    _lam = float(getattr(self, 'rail_walk_deploy_lambda', 1e-3))
                    _JJt = (_Ja * _Winv[None, :]) @ _Ja.T + (_lam * _lam) * np.eye(3)
                else:
                    _Winv = np.ones(len(self.arm_dofadr))
                    _JJt = _Ja @ _Ja.T + 1e-6 * np.eye(3)
                _railt_d = (np.eye(3) - np.outer(_n, _n)) @ self.rail_dir
                _railt_d = _railt_d / max(np.linalg.norm(_railt_d), 1e-9)
                _v_kin = float(getattr(self, 'latch_vrail', self.target_rail_vel))
                if getattr(self, 'rail_walk', False) and self.rail_walk_pace:
                    _v_kin = max(_v_kin + float(self.U[plan_idx][self._pace_idx]), 0.0)
                _dxn = -(_gap + _dz) * _n + _v_kin * self.dt * _railt_d
                _vx = _Ja @ np.asarray(data.qvel)[self.arm_dofadr]
                _vnx = float(np.dot(_vx, _n)) * _n
                _proj_pre_r = float(np.dot(data.site_xpos[self.contact_site_id], self.rail_dir))
                _dq_task = _Winv * (_Ja.T @ np.linalg.solve(_JJt, _dxn))
                if getattr(self, 'rail_walk', False) and float(getattr(self, 'rail_walk_post_kp', 0.0)) != 0.0:
                    _Jpi = _Ja.T @ np.linalg.solve(_JJt, np.eye(3))
                    _Np = np.eye(len(self.arm_dofadr)) - _Jpi @ _Ja
                    _dq_task = _dq_task + _Np @ (float(getattr(self, 'rail_walk_post_kp', 0.0)) * np.asarray(self.U[plan_idx][:self.nu])
                                                 - float(getattr(self, 'rail_walk_post_kd', 4.0)) * np.asarray(data.qvel)[self.arm_dofadr]) * self.dt
                data.qpos[self.arm_qposadr] += _dq_task
                data.qvel[self.arm_dofadr] += -_Winv * (_Ja.T @ np.linalg.solve(_JJt, _vnx))
                mujoco.mj_forward(model, data)
                self._dbg_proj_dr = float(np.dot(data.site_xpos[self.contact_site_id], self.rail_dir)) - _proj_pre_r
            else:
                self._dbg_proj_dr = 0.0
            _pin_q = getattr(self, 'ts_pin_qpos', None)
            if _pin_q is not None and len(_pin_q) > 0:
                data.qpos[np.asarray(_pin_q)] = np.asarray(self.ts_pin_vals)
                data.qvel[np.asarray(self.ts_pin_dof)] = 0.0
                mujoco.mj_forward(model, data)
            self._end_r_prev = float(np.dot(data.site_xpos[self.contact_site_id], self.rail_dir)) \
                if self.contact_site_id >= 0 else 0.0
            # press_force feature (mirror mppi_cpu.py:578-587). MUST store the
            f_step = 0.0
            for ci in range(data.ncon):
                c = data.contact[ci]
                pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
                if pair == {self.rock_body_id, self.stick_body_id}:
                    fb = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, fb)
                    f_step = max(f_step, abs(fb[0]))
            force_log.append(f_step)
            q, v = human_mppi._extract_pin_state()
            xs.append(np.concatenate([q, v]))

            self.U = np.concatenate([self.U[1:], np.zeros((1, self.n_ctrl),
                                                            dtype=np.float32)],
                                     axis=0)

            dt_step = time.perf_counter() - t0
            t_total += dt_step

            if (t % 10 == 0 or getattr(self, 'log_every_step', False)) and getattr(self, 'verbose_solve', True):
                if self.contact_site_id >= 0:
                    p_site = data.site_xpos[self.contact_site_id]
                else:
                    p_site = np.zeros(3)
                progress_pct = (
                    np.dot(p_site - self.p_start, self.rail_dir)
                    / self.rail_len * 100
                    if self.rail_len > 1e-6 else 0.0)
                gap = 0.0
                if self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp = data.geom_xpos[self.rock_geom_id]
                    sc = data.xpos[self.stick_body_id]
                    sa = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp = np.clip(np.dot(rp - sc, sa),
                                 -self.stick_half_len, self.stick_half_len)
                    nearest = sc + tp * sa
                    gap = (np.linalg.norm(rp - nearest)
                           - self.stick_radius - 0.02)
                rs_normals = []
                rs_efc = []
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    pair = {model.geom_bodyid[c.geom1],
                            model.geom_bodyid[c.geom2]}
                    if pair == {self.rock_body_id, self.stick_body_id}:
                        fb = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, fb)
                        rs_normals.append(float(fb[0]))
                        if c.efc_address >= 0:
                            rs_efc.append(float(
                                data.efc_force[c.efc_address]))
                demo_SUM = sum(max(0.0, n) for n in rs_normals)
                CPU_MAX  = max((abs(n) for n in rs_normals), default=0.0)
                slot0_efc = None
                if data.ncon > 0 and data.contact[0].efc_address >= 0:
                    slot0_efc = float(
                        data.efc_force[data.contact[0].efc_address])
                demo_rec = None
                if (hasattr(human_mppi, '_demo_forces')
                        and human_mppi._demo_forces is not None
                        and t < len(human_mppi._demo_forces)):
                    demo_rec = float(human_mppi._demo_forces[t])
                _dbg_rs = ""
                if getattr(self, 'debug_stick', False) and self.rock_geom_id >= 0:
                    _rp = data.geom_xpos[self.rock_geom_id]
                    _sc = data.xpos[self.stick_body_id]
                    _sa = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    _tp = np.clip(np.dot(_rp - _sc, _sa), -self.stick_half_len, self.stick_half_len)
                    _near = _sc + _tp * _sa
                    _dbg_rs = (f"  ROCK=[{_rp[0]:.3f},{_rp[1]:.3f},{_rp[2]:.3f}] "
                               f"STK_NEAR=[{_near[0]:.3f},{_near[1]:.3f},{_near[2]:.3f}] "
                               f"stkbase=[{_sc[0]:.3f},{_sc[1]:.3f},{_sc[2]:.3f}]")
                print(f"  [t={t:4d}]  progress={progress_pct:.1f}%  "
                      f"|u|={np.linalg.norm(u):.1f}  gap={gap:.4f}  "
                      f"rs={len(rs_normals)}  normals={[round(n,2) for n in rs_normals]}  "
                      f"efc@rs={[round(e,2) for e in rs_efc]}  | "
                      f"SUM={demo_SUM:.2f}  MAX={CPU_MAX:.2f}  "
                      f"slot0_efc={slot0_efc}  demo_rec={demo_rec}  "
                      f"step={dt_step:.2f}s{_dbg_rs}"
                      f" | RAIL touched={self._deploy_touched:.0f} vr={_dbg_vrail:.3f} "
                      f"|tau_rail|={np.linalg.norm(_tau_rail_latch):.2f} ang={_dbg_ang:.0f}deg"
                      f" | vpre={getattr(self,'_dbg_vpre',0):.3f} vpost={getattr(self,'_dbg_vpost',0):.3f}"
                      f" | mm: between={getattr(self,'_dbg_between',0)*1000:.2f} step={getattr(self,'_dbg_step_dr',0)*1000:.2f} "
                      f"proj={getattr(self,'_dbg_proj_dr',0)*1000:.2f} net={float(np.dot(data.site_xpos[self.contact_site_id]-_site_pre, self.rail_dir))*1000:.2f}")

        if getattr(self, 'verbose_solve', True):
            print(f"\n[KinematicMPPI_MJX.solve] done — {T} steps, "
                  f"{t_total:.1f}s total ({t_total/T:.2f}s/step)")
        self._last_forces = np.array(force_log)
        human_mppi._press_forces = self._last_forces
        self._solve_seq = getattr(self, '_solve_seq', 0) + 1
        return np.array(xs), np.array(us)

    def install_as_solver(self, human_mppi):
        """Same semantics as KinematicMPPI_CPU.install_as_solver — MUST
        patch human_mppi.update_solver_weights{,_tv} so the IRL's weight
        updates actually reach kin.w_run. Without that hook, every solve()
        runs at the init weights and the IRL is stuck."""
        kin = self
        T = human_mppi.T
        kin.KEYS_RUN = KEYS_RUN

        class _ProblemShim:
            def __init__(self):
                self.T = T

        class _KinSolverShim:
            def __init__(self):
                self.problem = _ProblemShim()
                self.termination_tolerance = 1e-4
                self.with_callbacks = False
                self.xs = []
                self.us = []

            @staticmethod
            def _weights_key():
                """Hashable snapshot of the weights the next kin.solve() would use
                (the shim discards xs_init/us_init and kin.solve() resets state +
                uses solve-invariant sampling keys, so weights fully determine the
                result on a deterministic challenger)."""
                if getattr(kin, '_w_run_windows', None):
                    return tuple(tuple(sorted(w.items())) for w in kin._w_run_windows)
                return tuple(sorted(kin.w_run.items()))

            def solve(self, xs_init=None, us_init=None, maxiter=1000,
                      isFeasible=False, init_reg=None):
                _memo_on = os.environ.get('MPPI_SOLVE_MEMO', '0').strip().lower() \
                    not in ('0', '', 'false', 'off', 'no')
                if _memo_on:
                    _key = self._weights_key()
                    _m = getattr(self, '_solve_memo', None)
                    if (_m is not None and _m[0] == _key
                            and getattr(kin, '_solve_seq', 0) == _m[1]):
                        kin.U = _m[2].copy()
                        print("[KinematicMPPI_MJX] solve memo HIT — same weights as "
                              "previous solve, reusing its (deterministic) result "
                              "[skipped 1 full deploy]")
                        return True
                xs, us = kin.solve(human_mppi, T=T, visualize=False)
                self.xs = list(xs)
                self.us = list(us)
                human_mppi._last_xs = xs
                human_mppi._last_us = us
                if _memo_on:
                    self._solve_memo = (_key, getattr(kin, '_solve_seq', 0),
                                        kin.U.copy())
                return True

            def reset(self):
                """Delegate to the underlying kin's reset (zeros self.U,
                resets current_t and RNG key). Required so the IRL outer
                loop (MO_IRL._compute_dw) can request a cold-start MPPI
                solve via `solver.reset()` between gradient evaluations.
                Without this, hasattr(solver, 'reset') returns False on
                the shim and the reset call silently no-ops → MPPI keeps
                warmstarting between IRL iters → basin-hopping at K≥512."""
                kin.reset()

        human_mppi.solver   = _KinSolverShim()
        human_mppi._kin_mppi = kin

        _orig_update    = human_mppi.update_solver_weights
        _orig_update_tv = human_mppi.update_solver_weights_tv

        def _patched_update(w_run_new, w_term_new):
            _orig_update(w_run_new, w_term_new)
            kin_w = {k: w_run_new.get(k, 0.0) for k in kin.KEYS_RUN
                     if k in w_run_new}
            if kin_w:
                kin.set_weights({**kin.w_run, **kin_w})

        def _patched_update_tv(w_run_windows, w_term_windows):
            _orig_update_tv(w_run_windows, w_term_windows)
            keys_run = human_mppi.keys_run
            n_w = len(w_run_windows)
            kin_windows = []
            for k in range(n_w):
                w_full = dict(zip(keys_run, w_run_windows[k]))
                kin_w  = {key: w_full.get(key, 0.0) for key in kin.KEYS_RUN}
                kin_windows.append(kin_w)
            kin.set_tv_weights(kin_windows)

        human_mppi.update_solver_weights    = _patched_update
        human_mppi.update_solver_weights_tv = _patched_update_tv
        print(f"[KinematicMPPI_MJX] Installed as solver on HumanMPPI "
              f"(features: {KEYS_RUN})")

    def set_tv_weights(self, w_run_windows):
        """Store time-varying weight dicts (one per window). For --n_w 1 the
        single dict is used as uniform weights in solve()."""
        self._w_run_windows = list(w_run_windows) if w_run_windows else None
        if self._w_run_windows:
            self.set_weights(self._w_run_windows[0])

    def update_solver_weights(self, w_run, w_term=None):
        """Mirrors KinematicMPPI_CPU.update_solver_weights for the IRL hook."""
        self.set_weights(w_run)

    def update_solver_weights_tv(self, w_run_windows, w_term_windows=None):
        """Windowed/basis weight schedule. Mirrors CPU API. TV-aware
        rollout now wired in solve() (2026-05-25 fix) — `use_tv` branch
        builds per-horizon w_arr from these windows correctly. Previously
        only window 0's weights were used uniformly across the horizon,
        silently nullifying IRL's TV recovery work."""
        self._w_run_windows = w_run_windows
        if w_run_windows:
            keys = human_mppi_keys_run_or_default(self)
            self.set_weights(dict(zip(keys, w_run_windows[0])))

def human_mppi_keys_run_or_default(kin):
    """Helper for the legacy update_solver_weights_tv path (called without
    a human_mppi reference) — returns the kin's KEYS_RUN."""
    return KEYS_RUN