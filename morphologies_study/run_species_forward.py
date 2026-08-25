"""
run_species_forward.py
======================
EXPERIMENT A -- forward cost-transfer across upper-limb morphologies.

Take a cost function w* recovered from HUMAN demonstrations (a weights.json from
the MO-IRL pipeline) and hold it FIXED while re-solving the contact-rich
sharpening OCP on each species' body plan (human / chimp / H.naledi /
Neanderthal URDFs in human_model/urdf/). Same reward, same task target, four
morphologies -> behaviour diverges only because of segment geometry, humeral
torsion, mass-inertia, and ROM. This is the trajectory-level version of the
static reach/effort table in the paper.

The task geometry (q0_full, q_traj warmstart, contact target) is taken from one
HUMAN mocap cycle and shared across species, so the ONLY thing that changes
between runs is the URDF.

What reaches the CSQP solver, and what does not (verified empirically in
human_base.py / human_crocoddyl.py -- see analysis/species_experiments/README):
  * segment geometry (bone lengths via joint origins)        -> YES (kinematics)
  * joint orientation offsets (torsion / cranial tilt rpy)   -> YES (kinematics)
  * joint POSITION + VELOCITY limits (ROM)                   -> YES, via
        keep_species_limits=True (state bounds x_lb/x_ub). Without it
        _setup_limits() overwrites them with the reference human's.
  * mass-inertia (thorax / upper-arm / forearm)              -> YES, via
        keep_urdf_inertias=True on the GENERATED per-taxon URDFs. buildReducedModel
        merges each locked link's inertia into its nearest moving arm parent, so
        the species' segment masses survive onto the moving joints (verified:
        human arm 1.80/1.28/0.45 kg vs naledi 1.00/0.61/0.23 kg, matching Table
        tab:mass). Use --generated so this path is taken.
  * actuation EFFORT limits                                  -> SOFT only. Not hard
        OCP torque ceilings; they bite indirectly through the Tau feature cost.
        Contact force is therefore set by --target_force, not by a strength cap.
So this run transfers GEOMETRY + TORSION + ROM + MASS-INERTIA (4/5); effort is
the remaining channel, present only as a soft cost.

Usage (unified_env):
    conda run -n unified_env python run_species_forward.py \
        --weights analysis/moirl/felix_dl/csqp_irl__nw1__windowed/weights.json \
        --geom_subject S3 --task down_long --cycle 5 \
        --species human chimp homo_naledi homo_neanderthal \
        --outdir analysis/species_experiments
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

from run_csqp_identifiability import load_geometry, REPO

URDF_DIR = REPO / "human_model" / "urdf"
GEN_URDF_DIR = REPO / "human_model" / "urdf" / "generated"
DATE = "27_02"

def load_wstar(weights_path):
    """Read recovered cost weights. Two sources, auto-detected by extension:
      - pipeline weights.json        -> (w_run dict, None)            [constant]
      - population_recovery.npz      -> (w_run time-avg dict, basis)  [basis W(t)]
    `basis` (None for constant) = {'theta': (K, nr), 'K': K, 'keys': npz_keys}
    so run_one can rebuild the smooth Gaussian W(t) at EACH species' own horizon
    (the basis is defined in normalized time, so it re-evaluates at any T)."""
    if str(weights_path).endswith(".npz"):
        d = np.load(weights_path, allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        w_avg = np.asarray(d["w_hat"], float)
        w_run = {k: float(v) for k, v in zip(keys, w_avg)}
        mode = str(d["weight_mode"]) if "weight_mode" in d.files else "single"
        basis = None
        if mode == "basis":
            n_w = int(d["n_w"]) if "n_w" in d.files else 1
            K = int(d["K"]) if "K" in d.files else n_w
            full = (np.asarray(d["w_hat_full"], float) if "w_hat_full" in d.files
                    else w_avg)
            theta = full[:K * len(keys)].reshape(K, len(keys))
            basis = {"theta": theta, "K": K, "keys": keys}
            print(f"[wstar] BASIS W(t): K={K}, {len(keys)} features "
                  f"(time-varying transfer)")
        return w_run, basis
    with open(weights_path) as fh:
        blob = json.load(fh)
    w = blob.get("weights_run", blob.get("w_run"))
    if w is None:
        raise SystemExit(f"no weights_run/w_run in {weights_path}")
    return {k: float(v) for k, v in w.items()}, None

def reconstruct_Wt(basis, run_keys, T):
    """Smooth Gaussian-basis W(t) at T nodes, columns ordered by run_keys.
    Mirrors IRL._build_basis: centers evenly spaced in normalized time, row-
    normalized, sigma = 1/(K-1). Returns (T, len(run_keys)); features not in the
    recovered set get 0."""
    theta, K, npz_keys = basis["theta"], basis["K"], basis["keys"]
    tfrac = np.linspace(0.0, 1.0, T)
    centers = np.linspace(0.0, 1.0, K)
    sigma = max(1.0 / max(K - 1, 1), 1e-6)
    B_raw = np.exp(-((tfrac[:, None] - centers[None, :]) ** 2) / (2.0 * sigma * sigma))
    B = B_raw / (B_raw.sum(axis=1, keepdims=True) + 1e-12)
    W = B @ theta
    out = np.zeros((T, len(run_keys)))
    for j, k in enumerate(run_keys):
        if k in npz_keys:
            out[:, j] = W[:, npz_keys.index(k)]
    return out

EFFORT_LIMITS = False
EFFORT_LIMIT_SCALE = 1.0
FORCE_STRICT_SLACK = 0.0
FORCE_MAX = 47.0
FMAX_HUMAN = 112.0
PRESS_CAPACITY_W = 0.0
FMAX_MAP = {}
_ARM_LINKS = ("clavicle", "shoulder_Y", "elbow_Y", "wrist_X")
# the tool must be FREE to fall short (else every reach gap collapses to 0).
HARD_RAIL = True
HARD_RAIL_TOL = 0.012
                        # 0.012 blows up divergent bodies -> use a looser window (e.g. 0.05)
RAIL_FROM_CONTACT = True
RAIL_LAT_W = 0.0
STICK_SHIFT_MAX = -1.0
EMERGENT_FRICTION = True
ARMATURE = 0.0
MAX_ITER = 1000
RETARGET = False
FREE_POSTURE = False
THORAX_OUT_OF_IK = False
MATCH_POSTURE = 0.0
MATCH_FIRST_ONLY = True
MATCH_TOOL_POSE = False
DEMO_TOOL_ROT = None
CONTACT_WINDOW = None
NEUTRAL_WARMSTART = False
APPROACH_HEIGHT = 0.05
RAIL_LENGTH = 0.0
START_TOOL_ORI = 0.0
OWN_TASK = False
TERM_POS_W = 0.0
TERM_POS_SLACK = 0.02
PROJECT_CONTACT = False
FRIC_NORMAL_ONLY = False
FRIC_DAMPING = 1.0
FRIC_CAP = 0.0
KEEP_SPECIES_LIMITS = True
LOCK_THORAX = True
LOCK_THORAX_TOL = 1e-3
FORCE_CONTINUATION = False
FORCE_SCHEDULE = (0.4, 0.7, 1.0)
SELF_COLLISION = False
SELF_COLLISION_MARGIN = 0.10

START_POSTURE_MAP = {}
ROCK_RATE_W = 0.0
ROCK_ORI_SCALE = 1.0
WRIST_HOLD_DEG = 0.0
ROCK_ORI_BOUND = 0.0
ANCHOR_START = False
START_DEMO_GRIP = 0.0
DEMO_GRIP_R = None
START_ROCK_PERP = 0.0
STATION_SHIFT = None
STICK_RADIAL = None
STICK_RADIUS = 0.0

def build_species(species, geom_subject, task, q0_full, q_traj, dt, w_run,
                  stick=None, neutral_q0=None, target_force=60.0):
    """HumanCrocoddyl on a species URDF, human-recovered w*, masses preserved.

    stick      : optional (p_start, p_end) world coords -> pin the rail to that
                 absolute world location (shared across species).
    neutral_q0 : optional reduced joint vector. When given, the model STARTS from
                 this common neutral arm posture (body at the default stance)
                 instead of the mocap pose, and the stick is anchored at the tool
                 contact point of that posture, so every species begins from the
                 same neutral stance with the rock already on the stick.
    """
    from final_models import HumanCrocoddyl
    urdf = str(URDF_DIR / f"{species}.urdf")
    if not os.path.exists(urdf):
        raise FileNotFoundError(urdf)
    w_eff = dict(w_run)
    _ros = float(globals().get('ROCK_ORI_SCALE', 1.0))
    if _ros != 1.0 and 'rock_ori' in w_eff:
        w_eff['rock_ori'] = float(w_eff['rock_ori']) * _ros
    if PRESS_CAPACITY_W > 0.0:
        w_eff['press_capacity'] = PRESS_CAPACITY_W
    fmax = float(FMAX_MAP.get(species, FORCE_MAX)) if FMAX_MAP else float(FORCE_MAX)
    args = dict(
        subject_id=geom_subject.lower(), date=DATE, task=task,
        urdf_path=urdf, keep_urdf_inertias=True,
        keep_species_limits=globals().get("KEEP_SPECIES_LIMITS", True),
        mu=0.3, target_force=float(target_force), contact="automatic", dt=dt,
        q0_full=q0_full, q0=q_traj[0], q_traj=q_traj, T=len(q_traj) - 1,
        force_strict_slack_frac=FORCE_STRICT_SLACK, w_run=w_eff,
        w_term=({'term_pos': TERM_POS_W} if TERM_POS_W > 0.0 else {}),
        rock_ori_bound=globals().get('ROCK_ORI_BOUND', 0.0),
        rock_rate_w=globals().get('ROCK_RATE_W', 0.0),
        wrist_hold_deg=globals().get('WRIST_HOLD_DEG', 0.0),
        press_normal_dual=True, press_friction_dual=True, press_in_effort=True,
        friction_dual_normal_only=globals().get('FRIC_NORMAL_ONLY', False),
        friction_dual_damping=globals().get('FRIC_DAMPING', 1.0),
        friction_dual_cap=globals().get('FRIC_CAP', 0.0),
        term_pos_slack=TERM_POS_SLACK,
        hard_rail=HARD_RAIL, hard_rail_tol=HARD_RAIL_TOL, windowed_rail=HARD_RAIL,
        rail_from_contact=RAIL_FROM_CONTACT,
        lock_thorax=globals().get('LOCK_THORAX', True),
        thorax_static=OWN_TASK,
        lock_thorax_tol=globals().get('LOCK_THORAX_TOL', 1e-3),
        progress_vel_target_mode=True, target_rail_vel=-1.0,
        own_task=OWN_TASK,
        # the tool is pinned to the surface from t=0 and each body must already be in place
        **({'contact_window': list(CONTACT_WINDOW)} if CONTACT_WINDOW else {}),
        # posture (e.g. naledi at its close-in own station) fights it -> blows up. Match it.
        stick_static=False,
        force_max=fmax,
        solver_type="CSQP", contact_aware_cost=True,
        effort_limits=EFFORT_LIMITS, effort_limit_scale=EFFORT_LIMIT_SCALE,
        armature=ARMATURE, max_iter=MAX_ITER,
        self_collision=SELF_COLLISION, self_collision_margin=SELF_COLLISION_MARGIN,
    )
    if neutral_q0 is not None:
        args.update(q0_full=None, q0=np.asarray(neutral_q0, dtype=float))
    if stick is not None:
        ps, pe = stick
        args.update(absolute_stick=True,
                    p_start=np.asarray(ps).tolist(), p_end=np.asarray(pe).tolist())
    if STICK_RADIAL is not None:
        args.update(stick_radial=np.asarray(STICK_RADIAL, float).tolist())
    if STICK_RADIUS > 0.0:
        args.update(stick_radius=float(STICK_RADIUS))
    return HumanCrocoddyl(args=args)

def neutral_pose(geom_subject, task, cycle):
    """A reference arm posture = mean of the human mocap cycle (a central,
    sensible sharpening stance). Returned as the reduced joint vector."""
    _, q_traj, _ = load_geometry(geom_subject, task, cycle)
    return np.mean(q_traj, axis=0)

def common_target(geom_subject, task, cycle, w_run, ref_q0):
    """The shared world working point + rail: where the HUMAN's reference pose
    puts the tool. All species will IK their tool to this same point."""
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    h = build_species("human", geom_subject, task, q0_full, q_traj, dt, w_run,
                      neutral_q0=ref_q0)
    m, d = h.pin_model, h.pin_model.createData()
    pin.framesForwardKinematics(m, d, np.asarray(h.q0, float))
    pin.updateFramePlacements(m, d)
    target = d.oMf[h.contact_frame_id].translation.copy()
    rail = h.p_end_world - h.p_start_world
    return target, rail

def species_ik_q0(species, geom_subject, task, cycle, w_run, target, ref_q0):
    """Build the species at the reference pose, IK its tool contact frame to the
    shared `target`, clamped to that body's joint limits. Returns the start q0 so
    every species begins at the SAME working point with a morphology-appropriate
    posture. Also returns the residual reach error (m)."""
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    h = build_species(species, geom_subject, task, q0_full, q_traj, dt, w_run,
                      neutral_q0=ref_q0)
    m, d = h.pin_model, h.pin_model.createData()
    fid = h.contact_frame_id
    qmin, qmax = np.asarray(m.lowerPositionLimit), np.asarray(m.upperPositionLimit)
    q = np.asarray(h.q0, float).copy()
    err = 1.0
    for _ in range(400):
        pin.framesForwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        e = d.oMf[fid].translation - np.asarray(target, float)
        err = float(np.linalg.norm(e))
        if err < 1e-5:
            break
        J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        dq = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
        q = pin.integrate(m, q, dq)
        q = np.minimum(np.maximum(q, qmin), qmax)
    return q, err

def reference_stick(geom_subject, task, cycle, w_run):
    """Build the human model with its natural rail and return (p_start, p_end)
    world coords, to share as the fixed stick across all species.

    Also returns the geometry subject's stick AXIS DIRECTION, which --share_stick_frame
    hands to every other body so the workpiece is shared in orientation and not only in
    position."""
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    h = build_species("human", geom_subject, task, q0_full, q_traj, dt, w_run)
    return (h.p_start_world.copy(), h.p_end_world.copy(),
            np.asarray(getattr(h, "stick_radial", None), float).copy()
            if getattr(h, "stick_radial", None) is not None else None)

def _reach_geom(human):
    """Shoulder world position + max arm reach (shoulder->elbow->wrist->tool segment
    sum) for a built body, at its start pose."""
    m = human.pin_model; d = m.createData()
    pin.framesForwardKinematics(m, d, np.asarray(human.q0, float))
    pin.updateFramePlacements(m, d)
    sh  = d.oMi[m.getJointId("right_shoulder_Z")].translation.copy()
    elb = d.oMi[m.getJointId("right_elbow_Z")].translation.copy()
    wri = d.oMi[m.getJointId("right_wrist_Z")].translation.copy()
    tool = d.oMf[human.contact_frame_id].translation.copy()
    L = float(np.linalg.norm(elb - sh) + np.linalg.norm(wri - elb) + np.linalg.norm(tool - wri))
    return sh, L

def _ik_tool_to(human, target, q_seed, iters=800, posture_gain=0.6):
    """Damped-least-squares IK bringing THIS body's contact frame to `target`,
    clamped to the body's own joint limits, with a NULL-SPACE bias that keeps the
    redundant arm near the natural seed posture (q_seed). Without the null-space
    term the tool-position IK resolves the redundancy INTO the body (arm inside the
    trunk); the bias keeps a natural, arm-out posture while still reaching."""
    m = human.pin_model; d = m.createData(); fid = human.contact_frame_id
    qmin = np.asarray(m.lowerPositionLimit); qmax = np.asarray(m.upperPositionLimit)
    q = np.asarray(q_seed, float).copy(); q_ref = q.copy(); nv = m.nv; I3 = np.eye(3); err = 1.0
    for _ in range(iters):
        pin.framesForwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
        e = d.oMf[fid].translation - np.asarray(target, float); err = float(np.linalg.norm(e))
        if err < 1e-5:
            break
        J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * I3, I3)
        dq_task = -Jp @ e
        dq_post = (np.eye(nv) - Jp @ J) @ (posture_gain * pin.difference(m, q, q_ref))
        q = pin.integrate(m, q, dq_task + dq_post)
        q = np.minimum(np.maximum(q, qmin), qmax)
    return q, err

def _thorax_frame(human):
    """World placement of the (locked) thorax/trunk frame -- a stance-STABLE anchor
    (the shoulder joint origin, by contrast, swings ~25cm with the clavicle)."""
    m = human.pin_model; d = m.createData()
    pin.framesForwardKinematics(m, d, np.asarray(human.q0, float)); pin.updateFramePlacements(m, d)
    for nm in ("thorax", "middle_thorax", "thoracic", "torso", "abdomen", "middle_abdomen"):
        if m.existFrame(nm):
            T = d.oMf[m.getFrameId(nm)]; return T.translation.copy(), T.rotation.copy()
    T = d.oMi[1]; return T.translation.copy(), T.rotation.copy()

def _reapply_node_overrides(human, species, verbose=False):
    """Re-assert the per-node constant-weight overrides on the CURRENT solver.

    Must be called after ANY rebuild (update_solver_weights_tv, or
    set_force_target_profile, which rebuilds via create_solver) or these silently
    revert: press_capacity is zeroed by the TV weight update (the basis has no
    column for it), and rail_lat falls back to its recovered value."""
    if PRESS_CAPACITY_W > 0.0:
        n = 0
        for rm in human.solver.problem.runningModels:
            cd = rm.differential.costs.costs.todict()
            if 'press_capacity' in cd:
                cd['press_capacity'].weight = PRESS_CAPACITY_W; n += 1
        if verbose:
            _fmax = float(FMAX_MAP.get(species, FORCE_MAX)) if FMAX_MAP else float(FORCE_MAX)
            print(f"  [press-capacity] w={PRESS_CAPACITY_W} on {n} nodes, Fmax={_fmax:.0f} N")
    _rrw = float(globals().get('ROCK_RATE_W', 0.0))
    if _rrw > 0.0:
        n = 0
        for rm in human.solver.problem.runningModels:
            cd = rm.differential.costs.costs.todict()
            if 'rock_rate' in cd:
                cd['rock_rate'].weight = _rrw; n += 1
        if verbose:
            print(f"  [rock-rate] w={_rrw:g} re-asserted on {n} nodes")
    if RAIL_LAT_W > 0.0:
        n = 0
        for rm in human.solver.problem.runningModels:
            cd = rm.differential.costs.costs.todict()
            if 'rail_lat' in cd:
                cd['rail_lat'].weight = RAIL_LAT_W; n += 1
        if verbose:
            print(f"  [rail_lat] soft rail imposed w={RAIL_LAT_W} on {n} nodes")

def species_stick_shift(sp, geom_subject, task, cycle, w_run, stick, shift_max):
    """Bounded free-hand shift of the SHARED stick for one body.

    The workpiece is held in the other hand, which can feed it up/down its own shaft
    while scraping but cannot re-station it. We therefore take the body's neutral-pose
    tool position, measure how far it sits from the shared stick's start ALONG THE STICK
    AXIS, and grant at most `shift_max` of that. Lateral error is never granted -- the
    body must reach the human's scrape line sideways on its own, so a genuine reach
    failure is still a failure. Returns a world translation (3,) to apply to the stick."""
    q0f, qtraj, dt = load_geometry(geom_subject, task, cycle)
    ref_q0 = np.mean(qtraj, axis=0)
    hb = build_species(sp, geom_subject, task, q0f, qtraj, dt, w_run, neutral_q0=ref_q0)
    m = hb.pin_model; d = m.createData()
    pin.framesForwardKinematics(m, d, np.asarray(hb.q0, float)); pin.updateFramePlacements(m, d)
    tool = d.oMf[hb.contact_frame_id].translation.copy()
    ax = np.asarray(stick[1]) - np.asarray(stick[0])
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    want = float(np.dot(tool - np.asarray(stick[0]), ax))
    grant = float(np.clip(want, -shift_max, shift_max))
    print(f"[stick_shift] {sp:28s} wants {want*100:+6.1f} cm along shaft, "
          f"granted {grant*100:+6.1f} cm"
          f"{'  (CLAMPED)' if abs(want) > shift_max else ''}")
    return grant * ax

def own_station_sticks(species_list, geom_subject, task, cycle, w_run):
    """Per-body ALLOMETRIC workpiece (paper 'Task and workpiece placement'): place
    each body's stick at the same THORAX-relative location as the human's demonstrated
    contact, scaled to the fraction of ITS OWN reach, and start each body IK'd onto its
    OWN stick in a morphology-appropriate posture. Anchored to the locked thorax (not
    the swinging shoulder) and built at the NEUTRAL stance that run_one solves at, so
    the hand actually lands on the stick. Returns {sp: (stick=(p_start,p_end), start_q0)}."""
    def _tool_at(human):
        m = human.pin_model; d = m.createData()
        pin.framesForwardKinematics(m, d, np.asarray(human.q0, float)); pin.updateFramePlacements(m, d)
        return d.oMf[human.contact_frame_id].translation.copy()

    def _stick_axis(human):
        """Direction + length of the human's PHYSICAL stick (its stick_visual geom),
        so the per-body stick keeps the demo's (horizontal) orientation rather than
        being drawn along the vertical down-stroke path."""
        vm = human.visual_model; m = human.pin_model; d = m.createData(); gd = vm.createData()
        pin.forwardKinematics(m, d, np.asarray(human.q0, float))
        pin.updateGeometryPlacements(m, d, vm, gd)
        for i, g in enumerate(vm.geometryObjects):
            if "stick" in g.name.lower():
                ax = np.asarray(gd.oMg[i].rotation)[:, 2]
                try:
                    L = 2.0 * float(g.geometry.halfLength)
                except Exception:
                    L = 0.20
                return ax / (np.linalg.norm(ax) + 1e-9), L
        return np.array([1.0, 0.0, 0.0]), 0.20

    q0f, qtraj, dt = load_geometry(geom_subject, task, cycle)
    ref_q0 = np.mean(qtraj, axis=0)
    global RAIL_FROM_CONTACT
    _saved_rfc = RAIL_FROM_CONTACT
    RAIL_FROM_CONTACT = True
    h = build_species("human", geom_subject, task, q0f, qtraj, dt, w_run)
    RAIL_FROM_CONTACT = _saved_rfc
    _, L_h = _reach_geom(h)
    contact_h, end_h = h.p_start_world.copy(), h.p_end_world.copy()
    rail_demo = end_h - contact_h
    stick_ax, stick_len = _stick_axis(h)
    print(f"[own_station] stick axis {np.round(stick_ax,2)} len {stick_len:.2f} m; "
          f"demo rail vec {np.round(rail_demo,3)} len {np.linalg.norm(rail_demo):.3f} m")
    hn = build_species("human", geom_subject, task, q0f, qtraj, dt, w_run, neutral_q0=ref_q0)
    tool_n_h = _tool_at(hn)
    offset = contact_h - tool_n_h
    print(f"[own_station] human reach L={L_h:.3f} m; neutral-hand->contact offset {np.round(offset,3)}")
    out = {}
    for sp in species_list:
        try:
            hb = build_species(sp, geom_subject, task, q0f, qtraj, dt, w_run, neutral_q0=ref_q0)
        except Exception as e:
            print(f"[own_station] {sp} skip: {e}"); continue
        _, L_b = _reach_geom(hb)
        scale = L_b / (L_h + 1e-9)
        contact_b = _tool_at(hb) + offset * scale
        if STICK_SHIFT_MAX >= 0.0:
            # Everything off-axis is dropped: the body must reach the human's line laterally.
            d = contact_b - contact_h
            d_ax = float(np.dot(d, stick_ax)) * stick_ax
            n = float(np.linalg.norm(d_ax))
            if n > STICK_SHIFT_MAX:
                d_ax = d_ax * (STICK_SHIFT_MAX / n)
            contact_b = contact_h + d_ax
            print(f"[own_station] {sp:26s} shift: full={np.linalg.norm(d)*100:.1f}cm "
                  f"-> along-shaft {np.linalg.norm(d_ax)*100:.1f}cm "
                  f"(cap {STICK_SHIFT_MAX*100:.0f}cm{', CLAMPED' if n > STICK_SHIFT_MAX else ''})")
        q_start, ik_err = _ik_tool_to(hb, contact_b, ref_q0)
        # contact_b (rail start) and must slide the full demo vector down its OWN rail --
        stick = (contact_b, contact_b + rail_demo)
        out[sp] = (stick, q_start)
        print(f"[own_station] {sp:26s} L={L_b:.3f}  disp-from-human={np.linalg.norm(contact_b-contact_h):.3f} m  "
              f"IK-resid={ik_err*1000:.1f} mm")
    return out

def demonstrated_tool_path(geom_subject, task, cycle, w_run):
    """World-space path of the HUMAN tool/contact frame over the demonstrated
    stroke (forward kinematics of the human model on the mocap q_traj). Used as the
    IK-retarget target so every body seeds its tool on the demonstrated path rather
    than pasting the human's joint angles onto a differently-proportioned skeleton."""
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    h = build_species("human", geom_subject, task, q0_full, q_traj, dt, w_run)
    m, d = h.pin_model, h.pin_model.createData()
    fid = h.contact_frame_id
    path, _rot = [], []
    for q in q_traj:
        pin.framesForwardKinematics(m, d, np.asarray(q, float))
        pin.updateFramePlacements(m, d)
        path.append(d.oMf[fid].translation.copy())
        _rot.append(d.oMf[fid].rotation.copy())
    global DEMO_TOOL_ROT
    DEMO_TOOL_ROT = np.asarray(_rot)
    return np.asarray(path)

def ik_retarget_traj(human, tool_path, contact_start=0):
    """Per-frame differential IK: bring THIS body's tool/contact frame onto the
    demonstrated tool_path[t], starting each frame from the mocap pose so non-tool
    DOFs stay near the reference posture and the rock DOF (which does not move the
    tool frame) is left for the solver. Returns ((T,nq) seed, max residual mm).
    For the human this is a no-op (its q_traj already realises tool_path)."""
    m, d = human.pin_model, human.pin_model.createData()
    fid = human.contact_frame_id
    qmin = np.asarray(m.lowerPositionLimit); qmax = np.asarray(m.upperPositionLimit)
    q_traj = np.asarray(human.q_traj, float)
    _lo, _hi = np.asarray(m.lowerPositionLimit, float), np.asarray(m.upperPositionLimit, float)
    _q_neutral = np.clip(0.5 * (_lo + _hi), _lo, _hi)
    if m.existJointName('middle_thoracic_X'):
        _ti = int(m.joints[m.getJointId('middle_thoracic_X')].idx_q)
        _q_neutral[_ti] = float(np.asarray(q_traj, float)[0, _ti])
    _thx_iv = None
    if THORAX_OUT_OF_IK and m.existJointName('middle_thoracic_X'):
        _thx_iv = int(m.joints[m.getJointId('middle_thoracic_X')].idx_v)
    seed, max_res = [], 0.0
    _prev = None
    for t in range(len(q_traj)):
        target = np.asarray(tool_path[t], float)
        if NEUTRAL_WARMSTART:
            q = (_prev.copy() if _prev is not None else _q_neutral.copy())
        else:
            q = (_prev.copy() if (FREE_POSTURE and _prev is not None) else q_traj[t].copy())
        err = 1.0
        for _ in range(150):
            pin.framesForwardKinematics(m, d, q)
            pin.updateFramePlacements(m, d)
            if MATCH_TOOL_POSE and DEMO_TOOL_ROT is not None:
                Md = pin.SE3(DEMO_TOOL_ROT[min(t, len(DEMO_TOOL_ROT) - 1)],
                             np.asarray(target, float))
                e6 = pin.log6(d.oMf[fid].actInv(Md)).vector
                err = float(np.linalg.norm(e6[:3]))
                if np.linalg.norm(e6) < 1e-4:
                    break
            else:
                e = d.oMf[fid].translation - target
                err = float(np.linalg.norm(e))
                if err < 1e-4:
                    break
            if MATCH_TOOL_POSE and DEMO_TOOL_ROT is not None:
                J6 = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)
                dq_task = J6.T @ np.linalg.solve(J6 @ J6.T + 1e-6 * np.eye(6), e6)
                J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            else:
                J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                dq_task = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e)
            if MATCH_POSTURE > 0.0 and (t == 0 or not MATCH_FIRST_ONLY):
                Jp_ = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), np.eye(3))
                dq_post = ((np.eye(m.nv) - Jp_ @ J)
                           @ (MATCH_POSTURE * pin.difference(m, q, q_traj[t])))
            else:
                dq_post = 0.0
            if _thx_iv is not None:
                J[:, _thx_iv] = 0.0
            q = pin.integrate(m, q, dq_task + dq_post)
            q = np.minimum(np.maximum(q, qmin), qmax)
        seed.append(q.copy()); _prev = q
        max_res = max(max_res, err)
    seed = np.asarray(seed)
    if contact_start > 0:
        # APPROACH PHASE. Do NOT hold the tool on the stick before contact: those frames
        # test never fires, and for five of seven taxa the stick passes BETWEEN the wrist and
        q_land = seed[contact_start].copy()
        pin.framesForwardKinematics(m, d, q_land); pin.updateFramePlacements(m, d)
        p_tool = d.oMf[fid].translation.copy()
        _ax = human.p_end_world - human.p_start_world
        _ax = _ax / (np.linalg.norm(_ax) + 1e-12)
        # problem and must not be solved by twisting the start posture.
        _shj = 'right_shoulder_Y'
        _sh = (d.oMi[m.getJointId(_shj)].translation.copy() if m.existJointName(_shj)
               else p_tool)
        _out = _sh - human.p_start_world
        _out = _out - np.dot(_out, _ax) * _ax
        _out = _out / (np.linalg.norm(_out) + 1e-12)
        if ANCHOR_START:
            _on_axis = human.p_start_world
        else:
            _on_axis = human.p_start_world + float(
                np.dot(p_tool - human.p_start_world, _ax)) * _ax
        p_land = _on_axis + 0.02 * _out
        p_air  = _on_axis + (0.02 + APPROACH_HEIGHT) * _out

        _Rd = (DEMO_TOOL_ROT[min(contact_start, len(DEMO_TOOL_ROT)-1)]
               if (START_TOOL_ORI > 0.0 and DEMO_TOOL_ROT is not None) else None)

        _a_perp = None
        if START_ROCK_PERP > 0.0:
            _a_perp = np.asarray(human.R_surface, float)[:, 2].copy()

        def _ik_to(target, q_init, ori_gain=None):
            _gain = START_TOOL_ORI if ori_gain is None else ori_gain
            _q = np.asarray(q_init, float).copy()
            for _ in range(400):
                pin.framesForwardKinematics(m, d, _q); pin.updateFramePlacements(m, d)
                e_a = d.oMf[fid].translation - target
                Jp = pin.computeFrameJacobian(m, d, _q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                if _thx_iv is not None:
                    Jp[:, _thx_iv] = 0.0
                dq = -Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6*np.eye(3), e_a)
                if _Rd is not None or _a_perp is not None:
                    Rc = np.asarray(d.oMf[fid].rotation)
                    a_cur = Rc[:, 2]
                    a_ref = (_a_perp if _a_perp is not None else np.asarray(_Rd)[:, 2])
                    _cx = np.cross(a_cur, a_ref)
                    _dt = float(np.clip(np.dot(a_cur, a_ref), -1.0, 1.0))
                    _ln = np.linalg.norm(_cx)
                    w_err = (Rc.T @ (_cx / _ln * np.arctan2(_ln, _dt))) if _ln > 1e-9 \
                            else np.zeros(3)
                    Jw = pin.computeFrameJacobian(m, d, _q, fid, pin.LOCAL)[3:, :]
                    if _thx_iv is not None:
                        Jw[:, _thx_iv] = 0.0
                    dq_w = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-4*np.eye(3), w_err)
                    Jpinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-6*np.eye(3), np.eye(3))
                    dq = dq + (np.eye(m.nv) - Jpinv @ Jp) @ (_gain * dq_w)
                if np.linalg.norm(e_a) < 1e-4 and np.linalg.norm(dq) < 1e-6:
                    break
                _q = pin.integrate(m, _q, dq)
                _q = np.minimum(np.maximum(_q, qmin), qmax)
            pin.framesForwardKinematics(m, d, _q); pin.updateFramePlacements(m, d)
            return _q, float(np.linalg.norm(d.oMf[fid].translation - target))

        def _pinned(_q, tol=1e-3):
            """How many joints the IK drove onto a limit -- the signature of an
            'impossible' posture, which orientation targets can easily produce."""
            return int(np.sum((np.asarray(_q) <= qmin + tol) | (np.asarray(_q) >= qmax - tol)))

        def _ik_safe(target, q_init, gain):
            """Orientation-aimed IK that BACKS OFF rather than contorting the arm.
            An attitude target competes with position in the null space, and when the body
            cannot honour both the solver walks joints onto their stops. Halve the gain
            until the position residual and the pinned-joint count are acceptable."""
            base_q, base_r = _ik_to(target, q_init, ori_gain=0.0)
            base_p = _pinned(base_q)
            _tol = max(5e-3, base_r + 1e-3)
            _dbg = os.environ.get("ROCK_DBG") == "1"
            if _dbg:
                print(f"    [rock-dbg] base residual {base_r*1000:.2f} mm, "
                      f"{base_p} pinned, accept if residual < {_tol*1000:.2f} mm")
            g = float(gain)
            while g > 1e-3:
                _q, _r = _ik_to(target, q_init, ori_gain=g)
                if _dbg:
                    pin.framesForwardKinematics(m, d, _q); pin.updateFramePlacements(m, d)
                    _o = np.degrees(np.arccos(np.clip(
                        float(np.asarray(d.oMf[fid].rotation)[:, 2] @ _a_perp), -1.0, 1.0)))
                    print(f"    [rock-dbg] gain {g:6.4f}: residual {_r*1000:8.2f} mm, "
                          f"{_pinned(_q)} pinned, {_o:5.1f} deg off -> "
                          f"{'ACCEPT' if (_r < _tol and _pinned(_q) <= base_p) else 'reject'}")
                if _r < _tol and _pinned(_q) <= base_p:
                    return _q, _r, g
                g *= 0.5
            return base_q, base_r, 0.0

        if START_DEMO_GRIP > 0.0:
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location(
                "_bsp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "build_start_postures.py"))
            _bsp = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_bsp)
            _P = np.column_stack(_bsp.rock_axes(human))
            _Rdes = DEMO_GRIP_R @ _P.T
            _ow = START_DEMO_GRIP

            def _ik6(target, q_init, iters=600):
                _q = np.asarray(q_init, float).copy()
                for _ in range(iters):
                    pin.framesForwardKinematics(m, d, _q); pin.updateFramePlacements(m, d)
                    _Rc = np.asarray(d.oMf[fid].rotation)
                    e = np.concatenate([d.oMf[fid].translation - target,
                                        _ow * (-pin.log3(_Rc.T @ _Rdes))])
                    _J = np.vstack([
                        pin.computeFrameJacobian(m, d, _q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :],
                        _ow * pin.computeFrameJacobian(m, d, _q, fid, pin.LOCAL)[3:, :]])
                    if _thx_iv is not None:
                        _J[:, _thx_iv] = 0.0
                    dq = -_J.T @ np.linalg.solve(_J @ _J.T + 1e-6*np.eye(6), e)
                    if np.linalg.norm(dq) < 1e-9:
                        break
                    _q = np.minimum(np.maximum(pin.integrate(m, _q, dq), qmin), qmax)
                pin.framesForwardKinematics(m, d, _q); pin.updateFramePlacements(m, d)
                return _q, float(np.linalg.norm(d.oMf[fid].translation - target))

            q_land, _rl = _ik6(p_land, q_land)
            _Rc = np.asarray(d.oMf[fid].rotation)
            _lon = np.degrees(np.arccos(np.clip(abs(float(
                (_Rc @ _bsp.rock_axes(human)[0]) @ _bsp.across_dir(
                    np.asarray(human.R_surface, float)[:, 2], _ax))), -1.0, 1.0)))
            print(f"  [grip] demonstrated attitude at the start: long axis {_lon:.0f} deg "
                  f"from across the stick, tool residual {_rl*1000:.1f} mm")
            seed[contact_start] = q_land.copy()
            q_air, _ra = _ik6(p_air, q_land)
        elif _a_perp is not None:
            q_land, _rl, _g1 = _ik_safe(p_land, q_land, START_ROCK_PERP)
            _Rc = np.asarray(d.oMf[fid].rotation)
            _off = np.degrees(np.arccos(np.clip(float(_Rc[:, 2] @ _a_perp), -1.0, 1.0)))
            # model's q0, which the approach phase never touches, so without this the
            human._R_rock_ref = np.asarray(d.oMf[human.tool_frame_id].rotation).copy()
            print(f"  [rock] start attitude aimed perpendicular to the stick: "
                  f"{_off:.0f} deg off (gain {_g1:.2f}, {_pinned(q_land)} joints at a limit)")
            seed[contact_start] = q_land.copy()
            q_air,  _ra, _ = _ik_safe(p_air, q_land, START_ROCK_PERP)
        else:
            q_land, _rl = _ik_to(p_land, q_land)
            seed[contact_start] = q_land.copy()
            q_air,  _ra = _ik_to(p_air,  q_land)
        print(f"  [approach] stick-built: lands ON the surface ({_rl*1000:.1f} mm), "
              f"hovers {APPROACH_HEIGHT*100:.0f} cm clear on the arm side ({_ra*1000:.1f} mm)")
        for t in range(contact_start):
            a = 0.5 * (1.0 - np.cos(np.pi * t / max(contact_start, 1)))
            seed[t] = pin.interpolate(m, q_air, q_land, float(a))
    return seed, max_res * 1000.0

def contact_forces(human):
    """Per-step 3D contact reaction at 'rail_sliding' on the current solution."""
    human.solver.problem.calc(human.solver.xs, human.solver.us)
    f = []
    for d in human.solver.problem.runningDatas:
        try:
            cd = d.differential.multibody.contacts.contacts["rail_sliding"]
            f.append(np.asarray(cd.f.linear).copy())
        except (KeyError, AttributeError):
            f.append(np.zeros(3))
    return np.asarray(f)

def segment_masses(human):
    """Total mass + a few named link masses from the reduced pin model."""
    m = human.pin_model
    out = {"total_mass": float(sum(I.mass for I in m.inertias))}
    return out

def arm_mass(human):
    """Upper-limb mass (clavicle+upper-arm+forearm+hand) from the reduced model --
    the strength proxy for press capacity. The _Z/_X shoulder/elbow decomposition
    joints are massless; thorax + rock are excluded."""
    m = human.pin_model
    return float(sum(m.inertias[i].mass for i in range(1, m.njoints)
                     if any(a in m.names[i] for a in _ARM_LINKS)))

def build_fmax_map(species_list, geom_subject, task, cycle, w_run):
    """Per-species press capacity Fmax = FMAX_HUMAN * (arm_mass/arm_mass_human)^(2/3)
    (isometric strength scaling: max force ~ muscle PCSA ~ mass^(2/3))."""
    q0f, qt, dt = load_geometry(geom_subject, task, cycle)
    am = {sp: arm_mass(build_species(sp, geom_subject, task, q0f, qt, dt, w_run))
          for sp in species_list}
    ref = am.get("human") or float(np.mean(list(am.values())))
    fmax = {sp: FMAX_HUMAN * (am[sp] / ref) ** (2.0 / 3.0) for sp in species_list}
    return fmax, am

def run_one(species, geom_subject, task, cycle, w_run, keys, outdir, stick=None,
            neutral_q0=None, basis=None, target_force=60.0, hold_steps=0,
            tool_path=None):
    print(f"\n{'#'*64}\n# {species}  ({task} / {geom_subject} cyc{cycle})\n{'#'*64}", flush=True)
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    if hold_steps:
        # horizon, so the body must HOLD the press longer. Cost is untouched; the
        q_traj = np.vstack([q_traj, np.repeat(q_traj[-1:], hold_steps, axis=0)])
        print(f"  [hold] +{hold_steps} frames -> T={len(q_traj)-1}", flush=True)
    human = build_species(species, geom_subject, task, q0_full, q_traj, dt, w_run,
                          stick=stick, neutral_q0=neutral_q0, target_force=target_force)
    human.q_traj = q_traj
    run_keys = list(human.keys_run)

    nv = human.nv
    if neutral_q0 is not None:
        x0 = np.concatenate([np.asarray(human.q0, float), np.zeros(nv)])
        xs_init = [x0.copy() for _ in range(len(q_traj))]
    elif RETARGET and tool_path is not None:
        _cs = int(round(CONTACT_WINDOW[0] * (len(q_traj) - 1))) if CONTACT_WINDOW else 0
        q_seed, res_mm = ik_retarget_traj(human, tool_path, contact_start=_cs)
        if _cs:
            print(f"  [approach] frames 0-{_cs} free flight: arm eases from a relaxed "
                  f"posture onto the stick (tool NOT pinned to the path before contact)")
        print(f"  [retarget] tool-path IK warmstart, max residual {res_mm:.1f} mm")
        if PROJECT_CONTACT:
            dq_seed = np.vstack([np.zeros((1, nv)), np.diff(q_seed, axis=0) / max(dt, 1e-9)])
            try:
                q_seed, dq_seed, _prep = human.project_kinematics_to_surface(q_seed, dq_seed)
                print(f"  [project] warmstart projected onto the contact surface "
                      f"({_prep if not isinstance(_prep, np.ndarray) else 'ok'})")
            except Exception as _e:
                print(f"  [project] SKIPPED: {type(_e).__name__}: {_e}")
        xs_init = [np.concatenate([q_seed[i], np.zeros(nv)]) for i in range(len(q_traj))]
    else:
        xs_init = [np.concatenate([q_traj[i], np.zeros(nv)]) for i in range(len(q_traj))]

    _pentry = START_POSTURE_MAP.get(species)
    if START_POSTURE_MAP and _pentry is None:
        print(f"  [start] WARNING no start posture for {species} in the posture file; "
              f"this body keeps its retargeted warmstart")
    if _pentry is not None:
        _q0s = np.asarray(_pentry['q'], float)
        if len(_q0s) != human.nq:
            print(f"  [start] WARNING posture for {species} has {len(_q0s)} dof but the "
                  f"model has {human.nq}; NOT applied")
        else:
            xs_init[0] = np.concatenate([_q0s, np.zeros(nv)])
            human.q0 = _q0s.copy()
            human.x0[:human.nq] = _q0s
            import pinocchio as _pin
            _dchk = human.pin_model.createData()
            _pin.framesForwardKinematics(human.pin_model, _dchk, _q0s)
            _pin.updateFramePlacements(human.pin_model, _dchk)
            _tip = _dchk.oMf[human.contact_frame_id].translation
            print(f"  [start] whole-chain posture applied at node 0; tool "
                  f"{np.linalg.norm(_tip - human.p_start_world)*1000:.1f} mm from the rail start")
    us_init = [np.zeros(human.nu) for _ in range(len(q_traj) - 1)]

    if basis is not None:
        W_node = reconstruct_Wt(basis, run_keys, human.T)
        _ros = float(globals().get('ROCK_ORI_SCALE', 1.0))
        if _ros != 1.0 and 'rock_ori' in run_keys:
            W_node[:, run_keys.index('rock_ori')] *= _ros
            print(f"  [rock] rock_ori column of W(t) scaled x{_ros:g}")
        human.update_solver_weights_tv(W_node, None)
        w_vec = W_node.mean(axis=0)
        print(f"  applied time-varying W(t) ({W_node.shape[0]} nodes)")
    else:
        w_use = {k: float(w_run.get(k, 0.0)) for k in run_keys}
        human.update_solver_weights(w_use, {})
        w_vec = np.array([w_use[k] for k in run_keys], float)
    _reapply_node_overrides(human, species, verbose=True)
    if FORCE_CONTINUATION:
        T = human.T
        prev_xs, prev_us = list(xs_init), list(us_init)
        for frac in FORCE_SCHEDULE:
            human.set_force_target_profile(np.full(T, frac * target_force))
            if basis is not None:
                human.update_solver_weights_tv(W_node, None)
            else:
                human.update_solver_weights(w_use, {})
            human.solve(xs_init=list(prev_xs), us_init=list(prev_us),
                        use_given_warmstart=True)
            prev_xs = [np.asarray(x) for x in human.solver.xs]
            prev_us = [np.asarray(u) for u in human.solver.us]
            print(f"  [force-cont] target {frac*target_force:.0f}N -> "
                  f"KKT {float(getattr(human.solver,'KKT',np.nan)):.3g}")
        xs, us = np.asarray(prev_xs), np.asarray(prev_us)
    else:
        if EMERGENT_FRICTION:
            # Friction must follow the REAL contact reaction, not the commanded target.
            human.set_force_target_profile(np.full(human.T, float(target_force)))
            if basis is not None:
                human.update_solver_weights_tv(W_node, None)
            else:
                human.update_solver_weights(w_use, {})
            _reapply_node_overrides(human, species)
            print(f"  [friction] emergent: per-node actuation, f_n from the contact dual "
                  f"(seed {float(target_force):.0f} N)")
        xs, us = human.solve(xs_init=list(xs_init), us_init=list(us_init),
                             use_given_warmstart=(RETARGET and tool_path is not None))
        xs, us = np.asarray(xs), np.asarray(us)
    f_c = contact_forces(human)

    Phi, *_ = human.get_traj_features(xs, us)
    Phi = np.asarray(Phi[:len(run_keys)], float)
    cost_contrib = w_vec * Phi

    import pinocchio as pin
    dd = human.pin_model.createData()
    pin.framesForwardKinematics(human.pin_model, dd, human.q0)
    pin.updateFramePlacements(human.pin_model, dd)
    hand0 = dd.oMf[human.contact_frame_id].translation.copy()
    reach_gap = float(np.linalg.norm(hand0 - human.p_start_world))

    feasible = bool(getattr(human.solver, "isFeasible", False))
    _sv = human.solver
    _dec = {k: float(getattr(_sv, k, float("nan"))) for k in ("dfeas", "gfeas", "hfeas", "feasNorm")}
    print(f"  [feas] dfeas(dyn gaps)={_dec['dfeas']:.3e}  gfeas(ineq)={_dec['gfeas']:.3e}  "
          f"hfeas(eq)={_dec['hfeas']:.3e}  feasNorm={_dec['feasNorm']:.3e}", flush=True)
    kkt = float(getattr(human.solver, "KKT", np.nan))
    fmag = np.linalg.norm(f_c, axis=1)
    sp_dir = os.path.join(outdir, f"{species}__{task}")
    os.makedirs(sp_dir, exist_ok=True)
    np.savez(os.path.join(sp_dir, "forward.npz"),
             keys=np.array(run_keys), w=w_vec, phi=Phi, cost_contrib=cost_contrib,
             xs=xs, us=us, f_contact=f_c, dt=dt, nq=human.nq, nv=human.nv,
             feasible=feasible, kkt=kkt, total_mass=segment_masses(human)["total_mass"],
             target_force=float(human.target_force),
             p_start=human.p_start_world, p_end=human.p_end_world,
             hand0=hand0, reach_gap=reach_gap,
             neutral_start=bool(neutral_q0 is not None),
             tv_weights=bool(basis is not None),
             q0_used=np.asarray(human.q0, float))

    print(f"  feasible={feasible}  KKT={kkt:.2e}  total_mass={segment_masses(human)['total_mass']:.2f} kg"
          f"  reach_gap={reach_gap:.3f} m")
    print(f"  mean|F_contact|={np.mean(fmag):.1f} N  peak={np.max(fmag):.1f} N")
    print(f"  saved -> {sp_dir}/forward.npz", flush=True)
    return dict(species=species, task=task, feasible=feasible, kkt=kkt,
                total_mass=segment_masses(human)["total_mass"], reach_gap=reach_gap,
                mean_force=float(np.mean(fmag)), peak_force=float(np.max(fmag)),
                phi=dict(zip(run_keys, Phi.tolist())),
                cost=dict(zip(run_keys, cost_contrib.tolist())))

TASK_DEFAULTS = {
    "down_long": {
        "weights": "data/weights/recovered_down_long.npz",
        "geom_subject": "S2", "cycle": 20, "target_force": 40.0,
        "force_strict_slack_frac": 0.15, "hard_rail_tol": 0.012,
        "fixed_stick": True, "share_stick_frame": True, "stick_radius": 0.02,
        "stick_shift_max": 0.05,
        "contact_window": [0.15, 1.0],
        "start_demo_grip": 0.15,
        "rock_rate_w": 0.0,
        "free_posture": True, "own_task": True, "project_contact": True,
        "retarget": True, "generated": True, "max_iter": 60,
    },
    "up_long": {
        "weights": "data/weights/recovered_up_long.npz",
        "geom_subject": "S3", "cycle": 7, "target_force": 30.0,
        "force_strict_slack_frac": 0.15, "hard_rail_tol": 0.012,
        "fixed_stick": True, "share_stick_frame": True, "stick_radius": 0.02,
        "stick_shift_vec": "0,6,4",
        "anchor_start": True,
        "start_posture_json": "data/start_postures_up_long.json",
        "rock_rate_w": 50.0,
        "free_posture": True, "own_task": True, "project_contact": True,
        "retarget": True, "generated": True, "max_iter": 60,
    },
}

def _apply_task_defaults(ap, args):
    """Fill unset options from TASK_DEFAULTS[args.task]."""
    table = TASK_DEFAULTS.get(args.task)
    if not table:
        return
    supplied = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    applied = []
    for key, value in table.items():
        if key not in supplied and getattr(args, key, None) in (None, "", ap.get_default(key)):
            setattr(args, key, value)
            applied.append(f"{key}={value}")
    if applied:
        print(f"[recipe] {args.task} defaults applied -> " + "  ".join(applied))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="recovered cost: pipeline weights.json (constant) OR a "
                         "population_recovery.npz (basis W(t), applied time-varying)")
    ap.add_argument("--target_force", type=float, default=60.0,
                    help="press target (N) for the transfer OCP (e.g. 50 = mean "
                         "of the measured profile)")
    ap.add_argument("--force_strict_slack_frac", type=float, default=0.15,
                    help="HARD force constraint: bound the contact dual to "
                         "target_force*(1+-frac) so the press FOLLOWS the commanded force "
                         "(0=off/soft, mirrors the human 'supplied force' B runs; try 0.15)")
    ap.add_argument("--press_capacity_w", type=float, default=0.0,
                    help="FORCE IN THE COST: weight on the (F-Fmax)^2 capacity term so the "
                         "press is pulled toward each body's Fmax (0=off -> emergent force)")
    ap.add_argument("--fmax_human", type=float, default=112.0,
                    help="human-species reference press capacity Fmax (N); other species "
                         "scale by (arm_mass/arm_mass_human)^(2/3)")
    ap.add_argument("--geom_subject", default="S3")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--species", nargs="+",
                    default=["human", "chimp", "homo_naledi", "homo_neanderthal"])
    ap.add_argument("--outdir", default="analysis/species_experiments")
    ap.add_argument("--start_tool_ori", type=float, default=0.0,
                    help="null-space gain aligning the tool's POINTING AXIS with the "
                         "demonstration at the hover and landing poses, so every body presents "
                         "the rock to the work the way the human does. Spin about that axis is "
                         "left free (a scrape does not care). 0 = attitude free (default). "
                         "Position is unaffected: the term lives in the position task's null space.")
    ap.add_argument("--rail_length", type=float, default=0.0,
                    help="override the stick length (m), keeping its start and direction. "
                         "0 = inherit the PCA fit through the chosen human cycle, whose length "
                         "varies 0.5-22.9 cm across cycles of the same nominal stroke.")
    ap.add_argument("--approach_height", type=float, default=0.05,
                    help="how far above the contact point (m, along the surface normal) every "
                         "body's TOOL starts the approach. Shared for all taxa, so the only "
                         "difference is how each arm gets down to the stick.")
    ap.add_argument("--neutral_warmstart", action="store_true",
                    help="styleless warmstart for EVERY body: IK the tool onto the "
                         "demonstrated path from each taxon's own mid-range posture instead "
                         "of from the human mocap. Makes the seven comparable -- with the "
                         "default the human is handed its own demonstration exactly (0.00 deg) "
                         "and the others are not.")
    ap.add_argument("--contact_window", type=float, nargs=2, default=None,
                    metavar=("START", "END"),
                    help="stroke fractions bounding the CONTACT phase, e.g. 0.25 1.0 -> the "
                         "first quarter is free flight (approach: no contact, no rail, no "
                         "press) and contact runs from 25%% to the end. Default: contact at "
                         "every node, which leaves no approach and demands the tool already "
                         "be on the surface at t=0.")
    ap.add_argument("--match_tool_pose", action="store_true",
                    help="retarget the tool's full 6-DOF POSE (position AND orientation) onto "
                         "the demonstration, so every body holds the rock against the stick the "
                         "same way. Default matches position only, leaving the tool's attitude "
                         "to whatever each body's redundancy picks. The arm is free otherwise "
                         "(6 of 9 DOF task-bound, 3 redundant).")
    ap.add_argument("--match_posture", type=float, default=0.0,
                    help="null-space gain pulling each body toward the HUMAN's joint posture "
                         "at every frame, while the tool still meets the demonstrated path "
                         "exactly. 0 = free redundancy (tool placed, posture arbitrary); "
                         "0.2-0.6 = start like the human, deviating only where this body's "
                         "own joint limits force it to. Applied to the FIRST FRAME only by "
                         "default (--match_all_frames to apply throughout): held across every "
                         "frame at a strong gain the correction leaks out of the null space "
                         "and fights the reach (chimp warmstart residual 0.1 -> 38 mm).")
    ap.add_argument("--match_all_frames", action="store_true",
                    help="apply --match_posture at every frame, not just the first.")
    ap.add_argument("--own_task", action=argparse.BooleanOptionalAction, default=True,
                    help="Share the TASK, not the human's TRAJECTORY. The transfer normally "
                         "hands every taxon the human's recorded q_traj, which then fixes its "
                         "trunk motion (thorax tracked to 1e-3 rad), its contact schedule, its "
                         "per-node contact normals and its seeded initial velocity -- so a "
                         "non-human body is asked to scrape the stick WHILE reproducing a human "
                         "stroke, and the problem is over-determined before it starts. This "
                         "frees all of those, keeping only what the task genuinely shares: the "
                         "same stick, the same cost weights, the same commanded force.")
    ap.add_argument("--term_pos_w", type=float, default=0.0,
                    help="soft terminal cost pulling the tool to the rail END. 0 = off "
                         "(only the hard +-term_pos_slack box, which bodies that stop short "
                         "simply violate). Try 5-50.")
    ap.add_argument("--term_pos_slack", type=float, default=0.02,
                    help="half-width (m) of the hard terminal position box.")
    ap.add_argument("--project_contact", action=argparse.BooleanOptionalAction, default=True,
                    help="project the warmstart onto the contact surface and zero the normal "
                         "velocity before solving -- the transfer-side equivalent of the "
                         "demonstration projection the recovery already applies.")
    ap.add_argument("--fric_fix", action="store_true",
                    help="use the NORMAL component of the contact dual for friction (not the "
                         "full-vector norm), damp the fixed point, and cap it at 5x the target.")
    ap.add_argument("--const_friction", action="store_true",
                    help="LEGACY: drive friction from the constant --target_force instead "
                         "of the solved contact dual. Default is emergent friction, which "
                         "matches the recovery; the constant form applies mu*target_force "
                         "to every body regardless of what it actually presses.")
    ap.add_argument("--stick_shift_max", type=float, default=-1.0,
                    help="With --own_station: cap the per-body workpiece displacement (m) and "
                         "restrict it to the stick's OWN axis, modelling the free hand sliding "
                         "the stick up/down while scraping. <0 = off (unbounded allometric "
                         "re-stationing). Typical 0.05-0.10. The required shift is reported "
                         "per taxon and is a physical readout in place of a reach gap.")
    ap.add_argument("--generated", action=argparse.BooleanOptionalAction, default=True,
                    help="use the generated 7-taxa URDFs in human_model/urdf/generated/ "
                         "(bonobo, A.sediba, A.prometheus + regenerated human/chimp/"
                         "naledi/neanderthal) instead of the old flat set")
    ap.add_argument("--urdf_dir", default=None,
                    help="explicit URDF directory (overrides --generated)")
    ap.add_argument("--fixed_stick", action=argparse.BooleanOptionalAction, default=True,
                    help="pin ONE shared world stick (the human's) for all species")
    ap.add_argument("--stick_shift_vec", default="",
                    help="translate the shared stick for THIS species, as "
                         "'along,out,across' in cm, expressed in the stick's own frame "
                         "(along its axis / along the contact normal / perpendicular to "
                         "both). Models the worker moving to a comfortable station while "
                         "the task geometry stays identical. Reach is NOT the binding "
                         "constraint -- all seven cover the shared rail -- so this buys "
                         "dexterity at the worst point of the stroke, not access.")
    ap.add_argument("--share_stick_frame", action=argparse.BooleanOptionalAction, default=True,
                    help="also share the stick's AXIS DIRECTION, not just the rail. "
                         "--fixed_stick pins p_start/p_end only; the axis offset direction "
                         "is re-derived from each body's own q0, so the contact normal "
                         "differed by 13-106 deg between taxa and pointed DOWNWARD for "
                         "H. naledi (hand pressing into the ground). This takes the "
                         "geometry subject's direction for every body.")
    ap.add_argument("--anchor_start", action="store_true",
                    help="begin contact at the RAIL START (s=0) instead of wherever the "
                         "retargeted tool projects onto the axis. Without it the stroke "
                         "starts at -2 cm on the down-stroke and +8 cm on the up, so the "
                         "two strokes cover 20 cm and 9 cm of the same 20 cm rail for "
                         "reasons that have nothing to do with the body.")
    ap.add_argument("--start_demo_grip", type=float, default=0.0,
                    help="start each body holding the rock at the DEMONSTRATED attitude "
                         "(0 = off, try 0.35 = the orientation weight in the joint 6D solve). "
                         "The demo grip is oblique -- 39 deg from across the stick, 73 from "
                         "the normal -- so imposing a squared-up attitude instead looks wrong "
                         "and is unreachable for the apes. Matched on the cobble's own "
                         "principal axes so the stone lies identically for every body.")
    ap.add_argument("--start_rock_perp", type=float, default=0.0,
                    help="aim the rock's pointing axis at the stick's contact NORMAL at the "
                         "start (0 = off, try 0.6). rock_ori is the heaviest recovered "
                         "weight and references an FK snapshot at q0, so the start attitude "
                         "is what the cost holds all stroke; this gives every body an "
                         "upright press without copying the human's grip. The gain backs "
                         "off automatically if it would drive joints onto their limits.")
    ap.add_argument("--stick_radius", type=float, default=0.02,
                    help="override the stick radius (m). The axis is placed one radius "
                         "from the rail, so the demonstrated contact path lies ON the "
                         "surface; the model shipped a 0.045 m offset against a 0.02 m "
                         "radius, leaving contact 25 mm off the stick. 0 = leave as-is.")
    ap.add_argument("--neutral_start", action="store_true",
                    help="start every species from a common neutral stance with "
                         "the rock already on the stick")
    ap.add_argument("--own_station", action="store_true",
                    help="place each body's stick at ITS OWN allometric station "
                         "(same shoulder-relative dir + fraction of its own reach as the "
                         "human's contact) and start it there morphology-appropriately, so "
                         "each scrapes its own comfortable line (paper Task-placement).")
    ap.add_argument("--hold_steps", type=int, default=0,
                    help="SUSTAINED-PRESS sweep: append N held-contact frames so "
                         "the body must hold the press longer (cost unchanged)")
    ap.add_argument("--effort_limits", action="store_true",
                    help="enforce per-taxon URDF effort ceilings as HARD torque box "
                         "constraints (makes effort a real morphology channel; "
                         "default OFF = soft Tau cost only)")
    ap.add_argument("--effort_limit_scale", type=float, default=1.0,
                    help="uniform multiplier on the effort ceilings (sensitivity sweep)")
    ap.add_argument("--armature", type=float, default=0.0,
                    help="uniform joint armature (reflected inertia, kg.m^2) applied to "
                         "ALL taxa to damp contact-onset ringing; 0 = off")
    ap.add_argument("--human_rom", action="store_true",
                    help="build every taxon with the HUMAN joint limits. The per-taxon ROM caps "
                         "enter the OCP as HARD state bounds at every node; a body pinned against "
                         "them has no slack left for the dynamics. Diagnostic only.")
    ap.add_argument("--free_thorax", action="store_true",
                    help="do not constrain the thoracic joint to the human demo path. The "
                         "transfer normally makes every taxon TRACK the human trunk per node, "
                         "which a differently-proportioned trunk may not be able to hold.")
    ap.add_argument("--thorax_tol", type=float, default=1e-3,
                    help="tolerance (rad) on the per-node thorax tracking constraint.")
    ap.add_argument("--free_posture", action=argparse.BooleanOptionalAction, default=True,
                    help="warmstart each taxon in its own redundancy branch (chain the "
                         "retarget IK frame-to-frame) rather than re-seeding every frame "
                         "from the human mocap posture. Same tool path and same cost.")
    ap.add_argument("--retarget", action=argparse.BooleanOptionalAction, default=True,
                    help="IK-retarget warmstart: seed every body's tool on the "
                         "demonstrated world path (per-frame IK) instead of the human's "
                         "joint angles; fixes divergent-morphology ringing")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--no_hard_rail", action="store_true",
                    help="turn OFF the hard windowed rail (default ON, matches recovery). "
                         "Use for the REACH analysis with a shared world stick, so the tool "
                         "is FREE to fall short instead of being pinned (reach gaps -> 0).")
    ap.add_argument("--hard_rail_tol", type=float, default=0.012,
                    help="lateral window (m) the tool may deviate from the scrape line under "
                         "the hard rail; 0.012 blows up divergent bodies, ~0.05 keeps the "
                         "scrape on-line while converging + still allowing reach gaps")
    ap.add_argument("--rail_lat_w", type=float, default=0.0,
                    help="impose a SOFT rail: override the recovered rail_lat weight so the "
                         "tool hugs the scrape line (looks like scraping) WITHOUT a hard "
                         "constraint -> converges + reach gaps survive. Use with --no_hard_rail. "
                         "Try ~0.4. Fixes the up-stroke wander (recovered rail_lat=0 there).")
    ap.add_argument("--force_continuation", action="store_true",
                    help="ramp the press target 0->target over a few chained solves "
                         "so a press-heavy cost builds contact force without stalling")
    ap.add_argument("--start_posture_json", default="",
                    help="use the precomputed whole-chain start posture from this file "
                         "(analysis/build_start_postures.py output) instead of re-solving it "
                         "in the run. The standalone solve poses the TOOL -- position and "
                         "attitude together -- and lets the whole chain adapt, reaching the "
                         "target to 0.0 mm for every body; the run's own 6D solve lands "
                         "10-22 deg off because position outweighs orientation there.")
    ap.add_argument("--rock_rate_w", type=float, default=50.0,
                    help="cost on the tool's ANGULAR VELOCITY, so the stone may not turn "
                         "while it travels. Unlike rock_ori (deviation from one snapshot, "
                         "paid down node by node) this penalises the turning itself, leaving "
                         "the slow drift the demonstration shows. Try 1-50.")
    ap.add_argument("--rock_ori_scale", type=float, default=1.0,
                    help="multiply the recovered rock_ori weight by this factor for the "
                         "transfer. The hard tool-frame bound (--rock_ori_bound) is returned "
                         "VIOLATED by this solver, and holding the wrist instead just moves "
                         "the rotation into the forearm; scaling the cost is the lever the "
                         "solver actually responds to. 1 = unchanged.")
    ap.add_argument("--wrist_hold_deg", type=float, default=0.0,
                    help="bound the wrist joints within this many degrees of their start "
                         "value, holding the rock's attitude through the joint-limit "
                         "channel (which this solver respects) rather than via a tool-frame "
                         "rotation constraint (which it returns violated). Try 8-15.")
    ap.add_argument("--rock_ori_bound", type=float, default=0.0,
                    help="HARD bound (deg) on how far the rock may rotate from its start "
                         "attitude. 0 = off, only the soft rock_ori cost, which the solver "
                         "trades away node by node: 30 deg of roll for the small hominins "
                         "where the demonstration turns the stone 11.9 deg. Try ~12.")
    ap.add_argument("--self_collision", action="store_true",
                    help="keep the forearm out of the trunk (lateral clearance "
                         "constraint); stops low-torsion elbows routing through the body")
    ap.add_argument("--self_collision_margin", type=float, default=0.10,
                    help="forearm-to-trunk lateral margin in metres (default 0.10)")
    args = ap.parse_args()
    _apply_task_defaults(ap, args)
    os.makedirs(args.outdir, exist_ok=True)

    global EFFORT_LIMITS, EFFORT_LIMIT_SCALE, ARMATURE, RETARGET, FORCE_CONTINUATION
    global FORCE_STRICT_SLACK, HARD_RAIL, HARD_RAIL_TOL, RAIL_FROM_CONTACT, STICK_SHIFT_MAX
    global EMERGENT_FRICTION, FREE_POSTURE, LOCK_THORAX, LOCK_THORAX_TOL, KEEP_SPECIES_LIMITS
    global FRIC_NORMAL_ONLY, FRIC_DAMPING, FRIC_CAP, PROJECT_CONTACT, TERM_POS_W, TERM_POS_SLACK, OWN_TASK, THORAX_OUT_OF_IK, MATCH_POSTURE, MATCH_FIRST_ONLY, MATCH_TOOL_POSE, CONTACT_WINDOW, NEUTRAL_WARMSTART, APPROACH_HEIGHT, RAIL_LENGTH, START_TOOL_ORI
    HARD_RAIL = not args.no_hard_rail
    HARD_RAIL_TOL = args.hard_rail_tol
    print(f"[hard_rail] {'ON (tool on scrape line, tol=%.3fm)'%HARD_RAIL_TOL if HARD_RAIL else 'OFF (tool free -> reach gaps)'}")
    RAIL_FROM_CONTACT = HARD_RAIL and not args.own_station
    if HARD_RAIL:
        print(f"[rail_from_contact] {'ON (rail re-fit from demo contact path)' if RAIL_FROM_CONTACT else 'OFF (own_station absolute per-body stick IS the rail)'}")
    OWN_TASK = args.own_task
    THORAX_OUT_OF_IK = OWN_TASK
    START_TOOL_ORI = args.start_tool_ori
    if START_TOOL_ORI > 0: print(f'[tool] start orientation matched to the demo (gain {START_TOOL_ORI})')
    RAIL_LENGTH = args.rail_length
    APPROACH_HEIGHT = args.approach_height
    NEUTRAL_WARMSTART = args.neutral_warmstart
    if NEUTRAL_WARMSTART: print('[warmstart] styleless mid-range start for every body')
    CONTACT_WINDOW = tuple(args.contact_window) if args.contact_window else None
    if CONTACT_WINDOW:
        print(f'[contact] approach {0:.0%}-{CONTACT_WINDOW[0]:.0%} free, contact {CONTACT_WINDOW[0]:.0%}-{CONTACT_WINDOW[1]:.0%}')
    MATCH_TOOL_POSE = args.match_tool_pose
    if MATCH_TOOL_POSE: print('[tool] retargeting the full 6-DOF tool pose (position + orientation)')
    MATCH_POSTURE = args.match_posture
    MATCH_FIRST_ONLY = not args.match_all_frames
    if MATCH_POSTURE > 0:
        print(f'[posture] matching the human posture, gain {MATCH_POSTURE}, '
              f"{'FIRST FRAME only' if MATCH_FIRST_ONLY else 'every frame'}")
    if OWN_TASK: print('[own_task] sharing the TASK, not the human trajectory: '
                       'STATIC thorax (same for all), own contact schedule, own contact frames')
    TERM_POS_W = args.term_pos_w
    TERM_POS_SLACK = args.term_pos_slack
    if TERM_POS_W > 0: print(f'[term] soft endpoint pull w={TERM_POS_W} toward the rail end')
    PROJECT_CONTACT = args.project_contact
    global STICK_RADIAL, STICK_RADIUS, STATION_SHIFT
    STICK_RADIUS = float(args.stick_radius)
    global START_ROCK_PERP
    START_ROCK_PERP = float(args.start_rock_perp)
    global START_DEMO_GRIP, DEMO_GRIP_R
    START_DEMO_GRIP = float(args.start_demo_grip)
    global ROCK_ORI_BOUND
    ROCK_ORI_BOUND = float(args.rock_ori_bound)
    global WRIST_HOLD_DEG
    WRIST_HOLD_DEG = float(args.wrist_hold_deg)
    global ROCK_ORI_SCALE
    ROCK_ORI_SCALE = float(args.rock_ori_scale)
    global ROCK_RATE_W
    ROCK_RATE_W = float(args.rock_rate_w)
    if ROCK_RATE_W > 0: print(f'[rock] angular-rate cost on the tool, w={ROCK_RATE_W:g}')
    global START_POSTURE_MAP
    if args.start_posture_json:
        with open(args.start_posture_json) as _fh:
            _sp = json.load(_fh)
        # taxa share joint topology, so the len(q)==nq guard downstream never fires
        START_POSTURE_MAP = dict(_sp.get('postures', {}))
        print(f"[start] whole-chain start postures loaded from {args.start_posture_json} "
              f"({len(START_POSTURE_MAP)} bodies)")
    if ROCK_ORI_SCALE != 1.0:
        print(f'[rock] rock_ori weight scaled x{ROCK_ORI_SCALE:g} for the transfer')
    global ANCHOR_START
    ANCHOR_START = bool(args.anchor_start)
    if STICK_RADIUS > 0:
        print(f"[stick] radius {STICK_RADIUS*100:.1f} cm; axis placed one radius from the "
              f"rail so the contact path lies on the surface")
    if PROJECT_CONTACT: print('[project] contact-surface projection of the warmstart: ON')
    EMERGENT_FRICTION = not args.const_friction
    if args.fric_fix:
        FRIC_NORMAL_ONLY, FRIC_DAMPING = True, 0.4
        FRIC_CAP = 5.0 * max(float(args.target_force), 1.0)
        print(f'[friction] FIX: normal-component only, damping 0.4, cap {FRIC_CAP:.0f} N')
    print(f"[friction] {'EMERGENT (from the contact dual)' if EMERGENT_FRICTION else 'CONSTANT mu*target_force (legacy)'}")
    STICK_SHIFT_MAX = args.stick_shift_max
    if STICK_SHIFT_MAX >= 0.0:
        if not (args.own_station or args.fixed_stick):
            print("[stick_shift] WARNING: --stick_shift_max needs --fixed_stick "
                  "(preferred) or --own_station; ignored")
        else:
            _where = "per-body allometric stick" if args.own_station else "shared world stick"
            print(f"[stick_shift] bounded station ON ({_where}): free hand may slide the "
                  f"stick along its own axis by at most {STICK_SHIFT_MAX*100:.0f} cm")
    EFFORT_LIMITS = args.effort_limits
    EFFORT_LIMIT_SCALE = args.effort_limit_scale
    FORCE_STRICT_SLACK = args.force_strict_slack_frac
    if FORCE_STRICT_SLACK > 0.0:
        print(f"[force-strict] HARD force constraint ON: press pinned to "
              f"target*(1+-{FORCE_STRICT_SLACK}) -> follows the recorded force")
    ARMATURE = args.armature
    global MAX_ITER; MAX_ITER = args.max_iter
    RETARGET = args.retarget
    FREE_POSTURE = args.free_posture
    KEEP_SPECIES_LIMITS = not args.human_rom
    if args.human_rom: print('[rom] per-taxon ROM caps DISABLED (human limits) -- diagnostic')
    LOCK_THORAX = not args.free_thorax
    LOCK_THORAX_TOL = args.thorax_tol
    print(f"[thorax] {'FREE' if not LOCK_THORAX else f'tracks demo path, tol {LOCK_THORAX_TOL} rad'}")
    if FREE_POSTURE: print('[posture] FREE: each taxon warmstarts in its own branch')
    FORCE_CONTINUATION = args.force_continuation
    if FORCE_CONTINUATION:
        print("[force-cont] press-target continuation ON")
    global SELF_COLLISION, SELF_COLLISION_MARGIN
    SELF_COLLISION = args.self_collision
    SELF_COLLISION_MARGIN = args.self_collision_margin
    if SELF_COLLISION:
        print(f"[self-collision] forearm-trunk lateral margin {SELF_COLLISION_MARGIN} m")
    if EFFORT_LIMITS:
        print(f"[effort] HARD per-taxon torque ceilings ON (scale={EFFORT_LIMIT_SCALE})")
    if ARMATURE > 0.0:
        print(f"[armature] uniform reflected inertia {ARMATURE} kg.m^2 on all taxa")
    if RETARGET:
        print("[retarget] IK tool-path warmstart ON for all taxa")

    global URDF_DIR
    if args.urdf_dir is not None:
        URDF_DIR = Path(args.urdf_dir)
    elif args.generated:
        URDF_DIR = GEN_URDF_DIR
    print(f"[urdf] using {URDF_DIR}")

    w_run, basis = load_wstar(args.weights)
    keys = sorted(w_run.keys())
    print(f"[wstar] from {args.weights}  (target_force={args.target_force} N)")
    print(f"[wstar] {json.dumps({k: round(w_run[k], 4) for k in keys}, indent=0)}")

    global PRESS_CAPACITY_W, FMAX_HUMAN, FMAX_MAP
    PRESS_CAPACITY_W = args.press_capacity_w
    FMAX_HUMAN = args.fmax_human
    if PRESS_CAPACITY_W > 0.0:
        FMAX_MAP, _am = build_fmax_map(args.species, args.geom_subject, args.task,
                                       args.cycle, w_run)
        print(f"[press-capacity] force IN THE COST: w={PRESS_CAPACITY_W}, "
              f"Fmax_human={FMAX_HUMAN} N")
        for sp in args.species:
            print(f"    {sp:28s} arm={_am[sp]:.2f}kg  Fmax={FMAX_MAP[sp]:.0f}N")

    stick = None
    if args.fixed_stick:
        stick_ps, stick_pe, _radial = reference_stick(args.geom_subject, args.task,
                                                      args.cycle, w_run)
        stick = (stick_ps, stick_pe)
        if args.stick_shift_vec:
            _c = [float(v) / 100.0 for v in args.stick_shift_vec.split(",")]
            _axd = (stick_pe - stick_ps) / (np.linalg.norm(stick_pe - stick_ps) + 1e-12)
            _nrm = -np.asarray(_radial, float) if _radial is not None else np.array([0., 0., 1.])
            _nrm = _nrm - float(np.dot(_nrm, _axd)) * _axd
            _nrm = _nrm / (np.linalg.norm(_nrm) + 1e-12)
            _acr = np.cross(_nrm, _axd)
            _shift = _c[0] * _axd + _c[1] * _nrm + _c[2] * _acr
            stick_ps = stick_ps + _shift; stick_pe = stick_pe + _shift
            stick = (stick_ps, stick_pe)
            STATION_SHIFT = _shift
            print(f"[station] stick translated by {np.round(_shift*100,1)} cm "
                  f"(along/out/across = {args.stick_shift_vec})")
        if args.share_stick_frame and _radial is not None:
            STICK_RADIAL = _radial
            print(f"[stick] axis direction SHARED across bodies: {np.round(_radial, 4)} "
                  f"(otherwise re-derived per body from its own q0)")
        if RAIL_LENGTH > 0:
            _a = np.asarray(stick[1], float) - np.asarray(stick[0], float)
            _a = _a / (np.linalg.norm(_a) + 1e-12)
            stick = (np.asarray(stick[0], float), np.asarray(stick[0], float) + RAIL_LENGTH * _a)
            print(f"[stick] rail length set to {RAIL_LENGTH*100:.0f} cm "
                  f"(demo cycles range 0.5-22.9 cm, mean 9.8)")
        print(f"[fixed_stick] shared world stick p_start={np.round(stick[0], 4)} "
              f"p_end={np.round(stick[1], 4)}")
        if START_DEMO_GRIP > 0.0:
            import importlib.util as _ilu
            _s2 = _ilu.spec_from_file_location("_bsp0", os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "build_start_postures.py"))
            _b0 = _ilu.module_from_spec(_s2); _s2.loader.exec_module(_b0)
            _qg0, _qgt, _dtg = load_geometry(args.geom_subject, args.task, args.cycle)
            _href = build_species("human", args.geom_subject, args.task, _qg0, _qgt,
                                  _dtg, w_run, stick=stick)
            DEMO_GRIP_R = _b0.demo_contact_rotation(_href, _qgt, CONTACT_WINDOW[0]
                                                    if CONTACT_WINDOW else 0.15)
            print(f"[grip] demonstrated rock attitude taken from the human at "
                  f"{100*(CONTACT_WINDOW[0] if CONTACT_WINDOW else 0.15):.0f}% of the stroke")
    ref_q0 = None
    if args.neutral_start:
        ref_q0 = neutral_pose(args.geom_subject, args.task, args.cycle)
        target, rail_vec = common_target(args.geom_subject, args.task, args.cycle,
                                         w_run, ref_q0)
        stick = (target, target + rail_vec)
        print(f"[neutral_start] shared tool target {np.round(target, 3)} -> each "
              f"species IK'd to start there, in contact")

    own = None
    if args.own_station:
        own = own_station_sticks(args.species, args.geom_subject, args.task, args.cycle, w_run)

    tool_path = None
    if RETARGET and not args.own_station:
        tool_path = demonstrated_tool_path(args.geom_subject, args.task, args.cycle, w_run)
        print(f"[retarget] demonstrated tool path: {len(tool_path)} frames")

    rows = []
    for sp in args.species:
        try:
            neutral_q0 = None; sp_stick = stick; sp_tool_path = tool_path
            if STATION_SHIFT is not None and tool_path is not None:
                # The rail moved; the DEMONSTRATED PATH the retarget IKs onto must move with
                sp_tool_path = [np.asarray(p, float) + STATION_SHIFT for p in tool_path]
            if args.own_station and own is not None and sp in own:
                sp_stick, neutral_q0 = own[sp]
            elif (STICK_SHIFT_MAX >= 0.0 and stick is not None):
                sh = species_stick_shift(sp, args.geom_subject, args.task, args.cycle,
                                         w_run, stick, STICK_SHIFT_MAX)
                sp_stick = (stick[0] + sh, stick[1] + sh)
                # The warmstart tool path must travel WITH the workpiece. Shifting the stick
                sp_tool_path = (None if tool_path is None
                                else [np.asarray(p, float) + sh for p in tool_path])
            elif args.neutral_start:
                neutral_q0, ik_err = species_ik_q0(sp, args.geom_subject, args.task,
                                                   args.cycle, w_run, target, ref_q0)
                print(f"[{sp}] IK to shared target: residual {ik_err*1000:.1f} mm")
            rows.append(run_one(sp, args.geom_subject, args.task, args.cycle,
                                w_run, keys, args.outdir, stick=sp_stick,
                                neutral_q0=neutral_q0, basis=basis,
                                target_force=args.target_force,
                                hold_steps=args.hold_steps, tool_path=sp_tool_path))
        except Exception as e:
            print(f"  !! {sp} FAILED: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(species=sp, task=args.task, feasible=False, kkt=float("nan"),
                             error=f"{type(e).__name__}: {e}"))

    summ = os.path.join(args.outdir, f"summary__{args.task}.json")
    with open(summ, "w") as fh:
        json.dump(dict(weights=args.weights, geom_subject=args.geom_subject,
                       task=args.task, cycle=args.cycle, rows=rows), fh, indent=2)
    print(f"\n=== summary -> {summ} ===")
    print(f"{'species':18s}{'feasible':>9s}{'mass':>8s}{'gap_m':>8s}{'mean_F':>9s}{'peak_F':>9s}")
    for r in rows:
        if "error" in r:
            print(f"{r['species']:18s}{'ERR':>9s}  {r['error']}")
        else:
            print(f"{r['species']:18s}{str(r['feasible']):>9s}{r['total_mass']:8.2f}"
                  f"{r.get('reach_gap', float('nan')):8.3f}{r['mean_force']:9.1f}{r['peak_force']:9.1f}")

if __name__ == "__main__":
    main()
