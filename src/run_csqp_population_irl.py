"""
run_csqp_population_irl.py
==========================
PROPER population (multi-subject) IRL on the CSQP human model, using our own
MO_IRL — each subject solved on its OWN body (URDF/morphology), one SHARED weight
vector. This is the model-side driver; the multi-subject support lives in the
gated `models=[...]` path of IRL.py / MO_IRL.py / IRL_utils.py.

Unlike run_moirl_batch's `all_dl` (all demos through ONE reference URDF), here
each subject's effort features are computed on THAT subject's body, so morphology
is not conflated with preference.

Synthetic protocol (known ground truth):
  1. Build N per-subject HumanCrocoddyl (own URDF), all RESAMPLED to a common
     horizon T (per-timestep feature averaging needs aligned timesteps).
  2. Pick a shared w*  ->  solve the OCP at w* on EACH body  ->  per-subject demos.
  3. Run MO_IRL with models=[m_0..], xs_opt=[demo_0..] (one shared w). Report
     cos(w*, w_hat) over the biomech features + per-subject q_norm.

Usage (unified_env):
    conda run -n unified_env python run_csqp_population_irl.py \
        --subjects S2,S3,S1 --task down_long --cycle 5 \
        --dominant Tau --max_iter 12 --line_search_steps 8 \
        --outdir toy_irl_study/csqp_population
"""
from repo_paths import REPO
import argparse
from csqp_cli import build_parser
from csqp_replay import replay
from csqp_replay import replay
from demo_prep import (consistent_us, cosine, load_avg_force_profile,
                       load_measured_force, resample_q,
                       select_representative_cycles, tv_wstar_windows)
import json
import os
import numpy as np
import pandas as pd

from run_csqp_identifiability import (load_geometry, build_model, ENG_SPLIT, DATE,
                                      TASK_DIR, REPO, SLICE_START, SLICE_END)
from run_csqp_synthetic_irl import contact_forces

BIOMECH = ["Tau", "JA", "JV", "JTC", "Geo",
           "Eng_thoracic", "Eng_clavicle", "Eng_shoulder", "Eng_elbow", "Eng_wrist",
           "press_force"]
TAU_SPLIT = ["Tau_thoracic", "Tau_clavicle", "Tau_shoulder", "Tau_elbow", "Tau_wrist"]
_ENG_GROUPS = ("thoracic", "clavicle", "shoulder", "elbow", "wrist")
_PROXIMAL_TAU = ("Tau_shoulder", "Tau_clavicle", "Tau_thoracic")

def tausplit_config(mode):
    """Per-segment-Tau experiment config: returns
    (biomech_keys, build_eng_weights, build_drop, wstar_bio)."""
    from run_csqp_identifiability import TAU_SPLIT_STAR
    eng = {f"Eng_{g}": 0.01 for g in _ENG_GROUPS}
    tau = (dict(TAU_SPLIT_STAR) if mode == "full"
           else {k: v for k, v in TAU_SPLIT_STAR.items() if k in _PROXIMAL_TAU})
    biomech = ["JA", "JV", "JTC", "Geo", "press_force", "press_capacity"] + list(eng) + list(tau)
    wstar_bio = {**{k: 0.01 for k in ("JA", "JV", "JTC", "Geo")}, **eng, **tau,
                 "press_force": 0.5, "press_capacity": 0.5}
    return biomech, {**eng, **tau}, ("Tau",), wstar_bio

# Task terms define WHAT the motion must achieve (reach + stay on rail), not the
# (masked). Leaving rail_lat/rock_ori learnable lets them blow up; progress_vel
TASK_INIT = {"progress_vel": 1.0, "rail_lat": 1e-6, "rock_ori": 1e-6}
TASK_FEATURES = list(TASK_INIT)

def main():
    ap = build_parser()
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.rock_ori_seed is not None:
        TASK_INIT["rock_ori"] = args.rock_ori_seed

    contact_windows = {}
    if args.contact_windows:
        with open(args.contact_windows) as _f:
            contact_windows = json.load(_f)
        print(f"[pop] contact windows from {args.contact_windows}: {contact_windows}")

    force_scales = {}
    if args.force_scale:
        with open(args.force_scale) as _f:
            force_scales = json.load(_f)
        print(f"[pop] force scales from {args.force_scale}: {force_scales}")
    subjects = [s.strip() for s in args.subjects.split(",")]

    from MO_IRL import MO_IRL
    from run_moirl_batch import _build_csqp_mask

    if args.tau_split != "none":
        biomech, build_eng, build_drop, wstar_bio = tausplit_config(args.tau_split)
        print(f"[pop] tau_split={args.tau_split}: biomech={biomech}")
    else:
        biomech, build_eng, build_drop, wstar_bio = BIOMECH, ENG_SPLIT, (), None
    if args.no_press_feature:
        build_drop = tuple(build_drop) + ("press_force", "press_capacity")
        print("[pop] --no_press_feature: dropped press_force/press_capacity from the "
              "cost (force enters only via press_in_actuation)")
    elif args.no_capacity and "press_capacity" not in build_drop:
        build_drop = tuple(build_drop) + ("press_capacity",)
        print("[pop] --no_capacity: dropped press_capacity; force feature = press_force (f_n^2) only")
    elif args.no_press_force and "press_force" not in build_drop:
        build_drop = tuple(build_drop) + ("press_force",)
        print("[pop] --no_press_force: dropped press_force; force feature = press_capacity "
              "((f-Fmax)^2) ONLY -- capacity/threshold traded against effort (Tau/Eng)")
    if args.force_track_max:
        if "press_peak" not in biomech:
            biomech = list(biomech) + ["press_peak"]
        for _pk_drop in ("press_force", "press_capacity"):
            if _pk_drop not in build_drop:
                build_drop = tuple(build_drop) + (_pk_drop,)
        print("[pop] --force_track_max: sole force feature = press_peak (F-F_peak)^2 at peak "
              "node; dropped press_force/press_capacity")

    if wstar_bio is not None:
        wstar = dict(wstar_bio)
    else:
        wstar = {k: (args.low_val if args.dominant else 0.01) for k in biomech}
        if args.dominant:
            wstar[args.dominant] = args.dominant_val
        wstar.setdefault("press_force", 0.1)
    if args.force_track_max:
        wstar.setdefault("press_peak", 0.1)
    wstar.update({"progress_vel": 1.0, "rail_lat": 1e-6, "rock_ori": 1e-6})

    tasks = [t.strip() for t in args.task.split(",")]
    if args.representative:
        _ro = int(args.rep_offset)
        pairs = [(s, tk, c) for tk in tasks for s in subjects
                 for c in select_representative_cycles(s, tk, _ro + args.n_cycles)[_ro:]]
    else:
        pairs = [(s, tk, c) for tk in tasks for s in subjects
                 for c in range(args.cycle, args.cycle + args.n_cycles)]
    print(f"[pop] {len(pairs)} demos over tasks {tasks}: {pairs}")
    _proj = None
    _proj_subj_aware = False
    if args.demos_from_npz:
        _proj = {}
        for _p in str(args.demos_from_npz).split(","):
            _p = _p.strip()
            if not _p:
                continue
            _pd = np.load(_p, allow_pickle=True)
            _pc = [int(x) for x in _pd["pair_cycles"]]
            _ps = ([str(x) for x in _pd["pair_subjects"]]
                   if "pair_subjects" in _pd.files else [None] * len(_pc))
            for i, c in enumerate(_pc):
                _xu = (np.asarray(_pd["proj_xs"][i], float), np.asarray(_pd["proj_us"][i], float))
                if _ps[i] is not None:
                    _proj_subj_aware = True
                    _proj[(_ps[i], c)] = _xu
                else:
                    _proj[c] = _xu

    def _projkey(s, c):
        return (s, int(c)) if _proj_subj_aware else int(c)

    if _proj is not None:
        pairs = [(s, tk, c) for (s, tk, c) in pairs if _projkey(s, c) in _proj]
        print(f"[pop] demos_from_npz: {len(pairs)} projected-feasible pairs "
              f"(subject-aware={_proj_subj_aware}): {pairs}")
    if args.drop_cycles:
        _drop = {int(x) for x in str(args.drop_cycles).split(",") if x.strip()}
        pairs = [(s, tk, c) for (s, tk, c) in pairs if int(c) not in _drop]
        print(f"[pop] dropped cycles {_drop} -> {len(pairs)} demos: {pairs}")
    geoms = {}
    for (s, tk, c) in pairs:
        q0_full, q_traj, dt = load_geometry(s, tk, c)
        geoms[(s, tk, c)] = (q0_full, q_traj, dt)
    T_target = min(len(q_traj) - 1 for (_, q_traj, _) in geoms.values())
    print(f"[pop] common horizon T = {T_target}")

    _fmax_map = {}
    if args.force_max_map:
        for _kv in str(args.force_max_map).split(","):
            if ":" in _kv:
                _k, _v = _kv.split(":"); _fmax_map[_k.strip()] = float(_v)
        print(f"[pop] per-subject Fmax: {_fmax_map} (else global {args.force_max})")

    models, demos_xs, demos_us, keys = [], [], [], None
    tv_ground_truth = None
    for (s, tk, c) in pairs:
        q0_full, q_traj, dt = geoms[(s, tk, c)]
        q_rs = resample_q(q_traj, T_target)
        if args.demo_smooth_q and len(q_rs) > args.demo_smooth_q:
            from scipy.signal import savgol_filter
            _w = args.demo_smooth_q + (1 - args.demo_smooth_q % 2)
            q_rs = savgol_filter(q_rs, _w, min(3, _w - 1), axis=0)
        dt_rs = dt * (len(q_traj) - 1) / T_target
        cw = contact_windows.get(f"{s}/{tk}")
        _fm_s = _fmax_map.get(s, float(args.force_max))
        _tf_s = _fmax_map.get(s, float(args.target_force))
        human, _ = build_model(s, tk, q0_full, q_rs, dt_rs, build_eng,
                               drop=build_drop, stick_static=args.static_stick,
                               target_force=_tf_s, contact_window=cw,
                               hard_rail=args.hard_rail, hard_rail_tol=args.hard_rail_tol,
                               windowed_rail=args.windowed_rail,
                               rail_from_contact=args.rail_from_contact,
                               contact_normal_slack=args.contact_normal_slack,
                               contact_mask_from_force=args.contact_mask_from_force,
                               contact_mask_thresh=args.contact_mask_thresh,
                               two_cost_force=args.two_cost_force,
                               force_track_max=args.force_track_max,
                               force_max=_fm_s,
                               force_strict_slack_frac=args.force_strict_slack_frac,
                               term_pos_slack=args.term_pos_slack,
                               press_in_actuation=args.press_in_actuation,
                               press_normal_dual=args.press_normal_dual,
                               press_friction_dual=args.press_friction_dual,
                               press_in_effort=args.press_in_effort,
                               progress_vel_target_mode=args.progress_vel_target_mode,
                               target_rail_vel=args.target_rail_vel,
                               eps_abs=args.eps_abs, max_qp_iters=args.max_qp_iters,
                               termination_tolerance=args.termination_tolerance,
                               lock_thorax=args.lock_thorax,
                               lock_thorax_tol=args.lock_thorax_tol,
                               effort_limits=args.effort_limits,
                               effort_limit_scale=args.effort_limit_scale,
                               contact_normal_track=args.contact_normal_track)
        human.q_traj = q_rs
        human.args['warmstart_press'] = float(args.warmstart_press)
        human.args['warmstart_gate_contact'] = bool(args.warmstart_gate_contact)
        keys = list(human.keys_run)
        if _proj is not None and _projkey(s, c) in _proj:
            xs_d, us_d = _proj[_projkey(s, c)]
            if args.demo_edge_hold > 0:
                us_d = np.asarray(us_d, float).copy()
                k = int(args.demo_edge_hold)
                if len(us_d) > 2 * k:
                    us_d[:k] = us_d[k]
                    us_d[-k:] = us_d[-k - 1]
            if args.avg_force:
                _tp = np.asarray(load_avg_force_profile(s, tk, T_target, fill=args.force_ref_fill), float)
                if getattr(args, 'flatten_force', False):
                    _tp = np.full_like(_tp, float(args.target_force))
                if float(args.force_strict_slack_frac) > 0.0:
                    human.set_force_target_profile(np.asarray(_tp[:human.T], float))
                else:
                    human.target_force_profile = _tp
            else:
                human.target_force_profile = np.full(T_target, float(args.target_force))
            print(f"[demo] {s}/cyc{c}: PROJECTED-FEASIBLE demo from npz (T={len(xs_d)-1}) "
                  f"F_target={float(np.mean(getattr(human,'target_force_profile',[0]))):.1f}N", flush=True)
            models.append(human)
            demos_xs.append(np.asarray(xs_d, float)); demos_us.append(np.asarray(us_d, float))
            continue
        if args.reuse_demo and args.replay_npz:
            _dd = np.load(args.replay_npz, allow_pickle=True)
            _i = len(models)
            xs_d = np.asarray(_dd['demos_xs'][_i], float)
            us_d = np.asarray(_dd['demos_us'][_i], float)
            print(f"[demo] {s}/cyc{c}: REUSED saved demo #{_i} from replay npz "
                  f"(T={len(xs_d)-1})", flush=True)
            models.append(human); demos_xs.append(xs_d); demos_us.append(us_d)
            continue
        if args.use_recorded:
            dq_rs = np.gradient(q_rs, dt_rs, axis=0)
            if args.project_kinematics:
                q_rs, _dqp, _prep = human.project_kinematics_to_surface(q_rs, dq_rs)
                human.q_traj = q_rs
                # v MUST equal d/dt(q_projected) for the state to be self-consistent
                dq_rs = np.gradient(q_rs, dt_rs, axis=0)
                print(f"[proj-kin] {s}/cyc{c}: projected {_prep['n_projected']}/{_prep['T']} "
                      f"nodes; penetration {_prep['pen_before_mm']:.1f}->{_prep['pen_after_mm']:.1f}mm; "
                      f"q shift={_prep['q_shift_deg']:.2f}deg")
            ddq_rs = np.gradient(dq_rs, dt_rs, axis=0)
            if args.demo_euler_consistent:
                dq_rs = np.vstack([(q_rs[1:] - q_rs[:-1]) / dt_rs, np.zeros((1, q_rs.shape[1]))])
                dq_rs[-1] = dq_rs[-2]
                ddq_rs = np.vstack([(dq_rs[1:] - dq_rs[:-1]) / dt_rs, np.zeros((1, q_rs.shape[1]))])
                ddq_rs[-1] = ddq_rs[-2]
            xs_d = np.column_stack([q_rs, dq_rs])
            if args.avg_force:
                fprof = load_avg_force_profile(s, tk, T_target, fill=args.force_ref_fill)
                if fprof is not None:
                    fscale = force_scales.get(f"{s}/{tk}", force_scales.get(s, 1.0))
                    fprof = fprof * float(fscale)
                    if getattr(args, 'flatten_force', False):
                        fprof = np.full_like(np.asarray(fprof, float), float(args.target_force))
                    if args.force_mean and float(np.mean(fprof)) > 1e-6:
                        fprof = fprof * (float(args.force_mean) / float(np.mean(fprof)))
                    human.set_force_target_profile(np.asarray(fprof[:human.T], float))
                    us_d = consistent_us(human, q_rs, dq_rs, ddq_rs,
                                         contact_aware_demo=args.contact_aware_demo,
                                         bake_press=args.press_normal_dual,
                                         contact_consistent_accel=args.contact_consistent_accel)[:, :human.nu]
                    _dmode = ('BAKED-PRESS' if args.press_normal_dual
                              else ('PRESSING' if args.contact_aware_demo else 'non-pressing'))
                    print(f"[demo] {s}/cyc{c}: AVG force target x{fscale} "
                          f"mean={fprof.mean():.1f}N max={fprof.max():.1f}N; {_dmode} demo us")
                else:
                    print(f"[demo] {s}/cyc{c}: no avg force -> fallback constant")
                    from run_moirl_batch import _friction_aware_tau
                    us_d = _friction_aware_tau(human, q_rs, dq_rs, ddq_rs)[:len(q_rs)-1, :human.nu]
            elif args.measured_force:
                fprof = load_measured_force(s, tk, c, T_target)
                if fprof is not None:
                    human.set_force_target_profile(np.asarray(fprof[:human.T], float))
                    us_d = consistent_us(human, q_rs, dq_rs, ddq_rs,
                                         contact_aware_demo=args.contact_aware_demo,
                                         bake_press=args.press_normal_dual,
                                         contact_consistent_accel=args.contact_consistent_accel)[:, :human.nu]
                    print(f"[demo] {s}/cyc{c}: MEASURED force target "
                          f"mean={fprof.mean():.1f}N max={fprof.max():.1f}N; "
                          f"{'BAKED-PRESS' if args.press_normal_dual else 'consistent'} us")
                else:
                    print(f"[demo] {s}/cyc{c}: no measured force -> fallback constant")
                    from run_moirl_batch import _friction_aware_tau
                    us_d = _friction_aware_tau(human, q_rs, dq_rs, ddq_rs)[:len(q_rs)-1, :human.nu]
            else:
                if args.q_norm_force_weight > 0:
                    human.target_force_profile = np.full(T_target, float(human.target_force))
                from run_moirl_batch import _friction_aware_tau
                us_d = _friction_aware_tau(human, q_rs, dq_rs, ddq_rs)[:len(q_rs)-1, :human.nu]
            print(f"[demo] {s}/cyc{c}: RECORDED T={human.T} dt={dt_rs*1e3:.2f}ms "
                  f"|v|max={np.abs(dq_rs).max():.2f} |u|max={np.abs(us_d).max():.1f}")
        else:
            xi = [np.concatenate([q_rs[i], np.zeros(human.nv)]) for i in range(len(q_rs))]
            ui = [np.zeros(human.nu) for _ in range(len(q_rs) - 1)]
            if args.tv_wstar:
                ramp_keys = {k for k in keys if k.startswith("Tau")
                             or k.startswith("Eng") or k == "press_force"}
                W_tv = tv_wstar_windows(wstar, keys, human.T, ramp_keys, lo=args.tv_lo)
                tv_ground_truth = W_tv
                xs_d, us_d = human.solve(xs_init=list(xi), us_init=list(ui),
                                         w_run_windows=W_tv)
            elif args.demo_force_continuation and wstar.get('press_capacity', 0) > 0:
                base_cap = float(wstar['press_capacity'])
                if args.demo_press_warmstart > 0:
                    _N = float(args.demo_press_warmstart)
                    dq_w = np.gradient(q_rs, dt_rs, axis=0)
                    ddq_w = np.gradient(dq_w, dt_rs, axis=0)
                    human.set_force_target_profile(np.full(human.T, _N, float))
                    _usw = consistent_us(human, q_rs, dq_w, ddq_w, bake_press=True,
                                         contact_consistent_accel=args.contact_consistent_accel)[:, :human.nu]
                    xs_d = [np.concatenate([q_rs[i], dq_w[i]]) for i in range(len(q_rs))]
                    us_d = [np.asarray(_usw[i], float) for i in range(len(_usw))]
                    print(f"[demo-cont] {s}/cyc{c}: seeded from {_N:.0f}N pressing warmstart")
                else:
                    xs_d, us_d = list(xi), list(ui)
                _fracs = [float(x) for x in str(args.ramp_fracs).split(',') if x.strip()]
                for frac in _fracs:
                    w_stage = dict(wstar); w_stage['press_capacity'] = base_cap * frac
                    human.update_solver_weights(w_stage, {})
                    xs_d, us_d = human.solve(xs_init=list(xs_d), us_init=list(us_d),
                                             use_given_warmstart=True)
                    try:
                        _fc = np.linalg.norm(contact_forces(human), axis=1).mean()
                        print(f"[demo-cont] {s}/cyc{c}: cap x{frac:.2f} -> |f|={_fc:.1f}N "
                              f"KKT={getattr(human.solver,'KKT',float('nan')):.1e}")
                    except Exception:
                        pass
                human.update_solver_weights(wstar, {})
            elif args.demo_press_warmstart > 0:
                _N = float(args.demo_press_warmstart)
                dq_w = np.gradient(q_rs, dt_rs, axis=0)
                ddq_w = np.gradient(dq_w, dt_rs, axis=0)
                human.set_force_target_profile(np.full(human.T, _N, float))
                _usw = consistent_us(human, q_rs, dq_w, ddq_w, bake_press=True,
                                     contact_consistent_accel=args.contact_consistent_accel)[:, :human.nu]
                xs_w = [np.concatenate([q_rs[i], dq_w[i]]) for i in range(len(q_rs))]
                us_w = [np.asarray(_usw[i], float) for i in range(len(_usw))]
                human.update_solver_weights(wstar, {})
                xs_d, us_d = human.solve(xs_init=xs_w, us_init=us_w, use_given_warmstart=True)
                print(f"[demo-warm] {s}/cyc{c}: light-press seed N={_N:.0f}N -> "
                      f"solve KKT={getattr(human.solver,'KKT',float('nan')):.1e} "
                      f"(cost discovers final press)")
            else:
                human.update_solver_weights(wstar, {})
                xs_d, us_d = human.solve(xs_init=list(xi), us_init=list(ui))
            print(f"[demo] {s}/cyc{c}: SYNTH{'(tv)' if args.tv_wstar else ''} "
                  f"T={human.T} feasible={getattr(human.solver,'isFeasible','?')} "
                  f"KKT={getattr(human.solver,'KKT',float('nan')):.1e}")
            try:
                _fd = np.linalg.norm(contact_forces(human), axis=1)
                print(f"[demo-force] {s}/cyc{c}: SYNTH press |f| mean={_fd.mean():.1f}N "
                      f"max={_fd.max():.1f}N  (moveable region if >>17N)")
                if args.q_norm_force_weight > 0:
                    human.set_force_target_profile(np.asarray(_fd[:human.T], float))
            except Exception as _e:
                print(f"[demo-force] {s}/cyc{c}: could not read force ({_e})")
        keys = list(human.keys_run)
        models.append(human)
        demos_xs.append(np.asarray(xs_d)); demos_us.append(np.asarray(us_d))

    nr = len(keys)
    wstar_vec = np.array([wstar[k] for k in keys], float)
    bio_idx = [keys.index(k) for k in biomech if k in keys]

    if args.check_press_grad:
        m = models[0]
        prob = m.solver.problem
        dxs = [np.asarray(x, float) for x in demos_xs[0]]
        dus = [np.asarray(u, float) for u in demos_us[0]]
        for t in [len(dus)//4, len(dus)//2, 3*len(dus)//4]:
            rm, rd = prob.runningModels[t], prob.runningDatas[t]
            x, u = dxs[t].copy(), dus[t].copy()
            rm.calc(rd, x, u); rm.calcDiff(rd, x, u)
            try:
                cd = rd.differential.multibody.contacts.contacts['rail_sliding']
            except Exception:
                cd = list(rd.differential.multibody.contacts.contacts.todict().values())[0]
            lam0 = np.asarray(cd.f.linear).copy()
            dfdu_an = np.asarray(cd.df_du).copy()
            dfdx_an = np.asarray(cd.df_dx).copy()
            eps = 1e-6
            fd_u = np.zeros((3, m.nu)); fd_x = np.zeros((3, 2*m.nv))
            for j in range(m.nu):
                up = u.copy(); up[j] += eps
                rm.calc(rd, x, up)
                fd_u[:, j] = (np.asarray(cd.f.linear) - lam0) / eps
            rm.calc(rd, x, u)
            print(f"[grad] node {t}: |lambda|={np.linalg.norm(lam0):.2f}N  "
                  f"||df_du analytical||={np.linalg.norm(dfdu_an):.3e}  "
                  f"||df_du finite-diff||={np.linalg.norm(fd_u):.3e}  "
                  f"||df_dx analytical||={np.linalg.norm(dfdx_an):.3e}", flush=True)
        return

    if args.verify_demo_force:
        vf_force, vf_target, vf_gap = [], [], []
        for i, m in enumerate(models):
            dxs = [np.asarray(x, float).copy() for x in demos_xs[i]]
            dus = [np.asarray(u, float).copy() for u in demos_us[i]]
            m.solver.solve([x.copy() for x in dxs], [u.copy() for u in dus], 0)
            fc = np.linalg.norm(m.get_contact_forces(), axis=1)
            gp = np.array([float(np.linalg.norm(np.asarray(f))) for f in m.solver.fs])
            prof = getattr(m, 'target_force_profile', None)
            tgt = (np.asarray(prof[:len(fc)], float) if prof is not None
                   else np.full(len(fc), float(m.target_force)))
            vf_force.append(fc); vf_gap.append(gp); vf_target.append(tgt)
            print(f"[verify] {pairs[i]}: demo |f_n| mean={fc.mean():.1f}N "
                  f"(target {tgt.mean():.1f}N)  gap/node mean={gp.mean():.3f} max={gp.max():.3f}",
                  flush=True)
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, "demo_force_verify.npz")
        np.savez(out, force=np.asarray(vf_force), target=np.asarray(vf_target),
                 gap=np.asarray(vf_gap), pair_cycles=np.asarray([c for _, _, c in pairs]))
        print(f"[verify] saved -> {out}")
        return

    if args.project_demo:
        import crocoddyl as _croc
        import pinocchio as pin
        def _pernode_gap(sv):
            return np.array([float(np.linalg.norm(np.asarray(f))) for f in sv.fs])
        def _deactivate_costs(costs):
            for nm in list(costs.active_set):
                costs.changeCostStatus(nm, False)
        proj_xs, proj_us, gaps_before, gaps_after = [], [], [], []
        for i, m in enumerate(models):
            dxs = [np.asarray(x, float).copy() for x in demos_xs[i]]
            dus = [np.asarray(u, float).copy() for u in demos_us[i]]
            _dt = float(getattr(m, 'dt', 1.0))
            _off = []
            for t in range(len(dxs)):
                _n = (m._R_surface_tv[t][:, 2] if getattr(m, '_R_surface_tv', None)
                      is not None and t < len(m._R_surface_tv) else m.R_surface[:, 2])
                _cp = m.get_contact_point(dxs[t])
                _off.append(float((_cp - m.p_start_world) @ _n))
            _off = np.asarray(_off)
            _nvel = np.gradient(_off, _dt)
            print(f"[project][pen] {pairs[i]}: normal offset min={_off.min()*1e3:+.2f}mm "
                  f"max={_off.max()*1e3:+.2f}mm | into-surface vel max="
                  f"{np.abs(_nvel).max():.3f} m/s (neg=penetrating)", flush=True)
            prob = m.solver.problem
            if i == 0:
                _cids = [id(rm.differential.costs) for rm in prob.runningModels]
                print(f"[project][dbg] T={prob.T} nRunning={len(prob.runningModels)} "
                      f"uniqueCostObjs={len(set(_cids))} "
                      f"termCost={id(prob.terminalModel.differential.costs)}")
            ndx = prob.terminalModel.state.ndx
            nvq = m.nv
            wdiag = np.concatenate([np.full(nvq, args.project_wq),
                                    np.full(ndx - nvq, args.project_wv)])
            act = _croc.ActivationModelWeightedQuad(wdiag)
            fprof = getattr(m, 'target_force_profile', None)
            cfid = m.contact_frame_id
            def _fresh_solver():
                np_ = _croc.ShootingProblem(np.asarray(prob.x0, float).copy(),
                                            list(prob.runningModels), prob.terminalModel)
                return m._make_solver(np_)
            for t, rm in enumerate(prob.runningModels):
                diff = rm.differential
                _deactivate_costs(diff.costs)
                res = _croc.ResidualModelState(diff.state, dxs[t], diff.actuation.nu)
                diff.costs.addCost("proj_track",
                                   _croc.CostModelResidual(diff.state, act, res), 1.0)
            tm = prob.terminalModel.differential
            _deactivate_costs(tm.costs)
            resT = _croc.ResidualModelState(tm.state, dxs[-1], tm.actuation.nu)
            tm.costs.addCost("proj_track",
                             _croc.CostModelResidual(tm.state, act, resT), 1.0)
            m.solver = _fresh_solver()
            m.solver.solve([x.copy() for x in dxs], [u.copy() for u in dus], 0)
            gaps_before.append(_pernode_gap(m.solver))
            g0 = float(m.solver.gap_norm)
            m.solver.solve([x.copy() for x in dxs], [u.copy() for u in dus],
                           args.project_iter)
            xs1 = [np.asarray(x).copy() for x in m.solver.xs]
            us1 = [np.asarray(u).copy() for u in m.solver.us]
            g1 = float(m.solver.gap_norm)
            f1 = np.abs(m.get_contact_forces()[:, 2])
            nforce = 0
            if float(args.project_bake_ureg) > 0:
                for t, rm in enumerate(prob.runningModels):
                    diff = rm.differential
                    ures = _croc.ResidualModelControl(diff.state, np.asarray(dus[t], float))
                    diff.costs.addCost("proj_ureg_bake",
                                       _croc.CostModelResidual(diff.state, ures),
                                       float(args.project_bake_ureg))
                if i == 0:
                    print(f"[project] stage2 PATH B: u-reg->baked wu={args.project_bake_ureg}")
                m.solver = _fresh_solver()
                m.solver.solve([x.copy() for x in xs1], [u.copy() for u in us1],
                               args.project_iter)
            elif float(args.project_wf) > 0 and fprof is not None:
                wdiag2 = np.concatenate([np.full(nvq, args.project_wq2),
                                         np.full(ndx - nvq, args.project_wv)])
                act2 = _croc.ActivationModelWeightedQuad(wdiag2)
                for t, rm in enumerate(prob.runningModels):
                    diff = rm.differential
                    diff.costs.changeCostStatus("proj_track", False)
                    res2 = _croc.ResidualModelState(diff.state, dxs[t], diff.actuation.nu)
                    diff.costs.addCost("proj_track2",
                                       _croc.CostModelResidual(diff.state, act2, res2), 1.0)
                    if not hasattr(diff, 'contacts'):
                        continue
                    fn = float(fprof[t]) if t < len(fprof) else float(m.target_force)
                    fref = pin.Force(np.array([0.0, 0.0, fn]), np.zeros(3))
                    fres = _croc.ResidualModelContactForce(diff.state, cfid, fref, 1,
                                                           diff.actuation.nu)
                    diff.costs.addCost("proj_force",
                                       _croc.CostModelResidual(diff.state, fres),
                                       float(args.project_wf))
                    nforce += 1
                tmc = prob.terminalModel.differential
                tmc.costs.changeCostStatus("proj_track", False)
                res2T = _croc.ResidualModelState(tmc.state, dxs[-1], tmc.actuation.nu)
                tmc.costs.addCost("proj_track2",
                                  _croc.CostModelResidual(tmc.state, act2, res2T), 1.0)
                if i == 0:
                    print(f"[project] stage2 force-pinned nodes={nforce}/{prob.T} "
                          f"wf={args.project_wf}")
                m.solver = _fresh_solver()
                m.solver.solve([x.copy() for x in xs1], [u.copy() for u in us1],
                               args.project_iter)
            gaps_after.append(_pernode_gap(m.solver))
            f2 = np.abs(m.get_contact_forces()[:, 2])
            _fm = np.asarray(fprof[:len(f2)], float) if fprof is not None else np.zeros(len(f2))
            def _rmse(a, b): return float(np.sqrt(np.mean((a[:len(b)] - b[:len(a)])**2)))
            print(f"[project]   {pairs[i]} gap s1={g1:.2e} s2={float(m.solver.gap_norm):.2e}"
                  f" KKT={float(m.solver.KKT):.1e} | "
                  f"F_meas={_fm.mean():.1f}N  F_model(s1)={f1.mean():.1f}N "
                  f"(rmse {_rmse(f1,_fm):.1f})  F_proj(s2)={f2.mean():.1f}N "
                  f"(rmse {_rmse(f2,_fm):.1f})", flush=True)
            pxs = np.stack(m.solver.xs.copy()); pus = np.stack(m.solver.us.copy())
            proj_xs.append(pxs); proj_us.append(pus)
            qerr = np.degrees(np.sqrt(np.mean(
                (pxs[:, :nvq] - np.stack(dxs)[:, :nvq])**2)))
            print(f"[project] {pairs[i]}: gap {g0:.2e} -> {m.solver.gap_norm:.2e} "
                  f"| KKT={m.solver.KKT:.2e} iter={m.solver.iter} "
                  f"| q drift from demo={qerr:.2f} deg")
        out = os.path.join(args.outdir, "projected_demos.npz")
        os.makedirs(args.outdir, exist_ok=True)
        np.savez(out,
                 proj_xs=np.asarray(proj_xs), proj_us=np.asarray(proj_us),
                 demos_xs=np.asarray(demos_xs), demos_us=np.asarray(demos_us),
                 gaps_before=np.asarray(gaps_before), gaps_after=np.asarray(gaps_after),
                 keys=np.asarray(keys), pair_cycles=np.asarray([c for _, _, c in pairs]),
                 pair_subjects=np.asarray([s for s, _, _ in pairs]),
                 pair_tasks=np.asarray([tk for _, tk, _ in pairs]))
        print(f"[project] saved -> {out}")
        return

    if args.replay_npz and not args.reuse_demo:
        replay(args, models, pairs, keys, demos_xs, demos_us, nr)
        return

    style_init = float(args.style_init)
    w_run = {k: style_init for k in keys}
    if args.init_press_cap > 0 and 'press_capacity' in w_run:
        w_run['press_capacity'] = float(args.init_press_cap)
        print(f"[pop] init press_capacity = {args.init_press_cap} (not floor)")
        if args.balanced_press_init and 'press_force' in w_run:
            w_run['press_force'] = float(args.init_press_cap)
            print(f"[pop] BALANCED press init: press_force = press_capacity = {args.init_press_cap} -> F*=Fmax/2")
        if args.init_press_force >= 0.0 and 'press_force' in w_run:
            # DELIBERATELY away from the demo's Fmax/2 so a genuine recovery must MOVE it
            w_run['press_force'] = float(args.init_press_force)
            _fstar = float(args.force_max) * float(args.init_press_cap) / (
                float(args.init_press_cap) + float(args.init_press_force) + 1e-12)
            print(f"[pop] OFF-BALANCE press init: press_force={args.init_press_force} "
                  f"press_capacity={args.init_press_cap} -> F*≈{_fstar:.1f}N (demo=Fmax/2)")
    for tk, tv in TASK_INIT.items():
        if tk in w_run:
            w_run[tk] = tv
    if args.learn_task:
        mask_out = []
    elif args.learn_task_only:
        _unfreeze = set(args.learn_task_only)
        mask_out = [t for t in TASK_FEATURES if t in keys and t not in _unfreeze]
    else:
        mask_out = [t for t in TASK_FEATURES if t in keys]
    if args.lock_thorax:
        for _tt in ("Tau_thoracic", "Eng_thoracic"):
            if _tt in keys and _tt not in mask_out:
                mask_out.append(_tt)
        print("[pop] lock_thorax=True -> thoracic held at q0; "
              "Tau_thoracic/Eng_thoracic frozen (prior-set)")
    print(f"[pop] frozen task terms (mask_out) = {mask_out or '[] (learning all)'}  "
          f"| normalize_w={bool(args.normalize_w)}  | learning biomech = "
          f"{[k for k in keys if k not in mask_out]}")
    n_w_eff = T_target if (args.mode == "basis" and args.basis_pernode) else args.n_w
    if args.basis_pernode and args.mode == "basis":
        print(f"[pop] basis per-node: n_w = T = {n_w_eff} (smooth W(t) every timestep)")
    irl_args = {
        "model": models[0], "models": models,
        "w_run": w_run, "w_term": {},
        "xs_opt": demos_xs, "us_opt": demos_us,
        "irl_iter": 1000, "stopping": "q_norm", "tol": 1e-10,
        "max_iter": args.max_iter, "min_iter": 1,
        "compare_desired": False, "verbose": True, "n_w": n_w_eff,
        "sqp_iter": args.sqp_iter,
    }
    if args.mode == "basis":
        K_eff = int(args.K or args.n_w)
        irl_args["weight_mode"] = "basis"
        irl_args["K"] = K_eff
        irl_args["basis_type"] = "gaussian"
        irl_args["basis_sigma"] = None
    K_for_mask = irl_args.get("K", args.n_w)
    mo_args = {
        "with_temp_adjust": False, "sqp_iter": args.sqp_iter, "next_traj": "worst",
        "line_search_steps": args.line_search_steps, "line_search_base": args.ls_base,
        "use_jac": True, "use_hess": False, "with_dmp": False, "basis_num": 20,
        "alpha_x": 1.5, "alpha_z": 50.0, "beta_z": 10, "noise_f": 50.0, "tau": 10.0,
        "rollout_N": 5, "opt_vars": "dw", "temperature": 1, "l_type": 1,
        "l_reg": "elastic", "use_bad": False, "normalize_w": bool(args.normalize_w),
        "use_dq_norm": bool(args.ls_use_dq), "dq_norm_tol": float(args.ls_dq_tol),
        "use_accel_guard": bool(args.ls_use_accel), "accel_tol": float(args.ls_accel_tol),
        "normalize_features": False, "normalizing_thrs": 1.0, "K_set": 1,
        "N_samples": 50, "scaled_sum": True, "use_best": False,
        "Lambda": float(args.reg_lambda), "Beta": float(args.reg_beta),
        "dyn_reg": False, "kappa": 9.0,
        "lambda_thrs": 1e-5, "lambda_init": 1.0, "delta": 1.0,
        "stopping": "q_norm", "tol": 1e-4,
        "gradient_mask": _build_csqp_mask(sorted(w_run.keys()), [], K_for_mask, mask_out),
        "use_mppi_grad": False, "verbose_irl": False,
        "per_feature_scale": bool(args.feature_scale),
        "q_norm_force_weight": float(args.q_norm_force_weight),
        "q_norm_meanjoint": bool(args.q_norm_meanjoint),
        "demo_force_at_target": bool(args.demo_force_at_target),
        "demo_force_measured": bool(args.demo_force_measured),
        "pool_scope": str(args.pool_scope),
        "pool_last_k": (int(args.pool_last_k) if args.pool_last_k is not None else None),
        "ls_cold_start": bool(args.ls_cold_start),
        "challenger_press_warmstart": bool(args.challenger_press_warmstart),
        "tau_norm_accept": bool(args.tau_norm_accept),
        "pareto_qnorm_cap": float(args.pareto_qnorm_cap),
        "challenger_continuation": bool(args.challenger_continuation),
        "cont_fracs": tuple(float(x) for x in str(args.cont_fracs).split(",") if x.strip()),
        "slsqp_inner": bool(args.slsqp_inner),
    }
    if args.feature_scale:
        mo_args["Lambda"] = 0.0
        print(f"[pop] feature_scale=True → Lambda(L1)=0; "
              f"Beta(L2)={mo_args['Beta']:g} kept (damps boundary blow-up)")

    if getattr(args, 'warmstart_neutral', False):
        for _m in models:
            _m._warmstart_neutral = True
        print(f"[pop] warmstart_neutral: challengers seed from STATIC q0 (no IK rail guess)")
    print(f"[pop] MO_IRL over {len(models)} subjects, shared w, "
          f"max_iter={args.max_iter} ls={args.line_search_steps} ...", flush=True)
    irl = MO_IRL(mo_args, irl_args)
    irl.solve()

    if args.best_last and len(getattr(irl, "ws", [])) > 0:
        best = len(irl.ws) - 1
        _od = np.asarray(getattr(irl, "opt_div", [np.nan]), float)
        print(f"[pop] best_last: last accepted iterate {best}  q_norm={irl.q_norm[best]:.3f}  "
              f"opt_div={_od[best] if best < len(_od) else float('nan'):.4g}")
    elif args.select_best_feature and len(getattr(irl, "opt_div", [])) > 1:
        _od = np.asarray(irl.opt_div, float)
        best = 1 + int(np.argmin(_od[1:]))
        print(f"[pop] select_best_feature: iterate {best} "
              f"opt_div={_od[best]:.4g} (last {_od[-1]:.4g})")
    else:
        best = int(np.argmin(irl.q_norm)) if len(getattr(irl, "q_norm", [])) else -1
    w_hat_full = np.asarray(irl.ws[best][0], float)
    w_hat = w_hat_full[:nr]
    cos_bio = cosine(wstar_vec, w_hat, bio_idx)
    weight_mode = irl_args.get("weight_mode",
                               "windowed" if args.n_w > 1 else "single")
    B_window = np.asarray(getattr(irl, "B_window", None)) \
        if getattr(irl, "B_window", None) is not None else np.zeros((0, 0))

    mode_str = "RECORDED demos" if args.use_recorded else "SYNTHETIC demos"
    print("\n" + "=" * 64)
    print(f"POPULATION RECOVERY  ({'+'.join(subjects)})  [{mode_str}]  "
          f"target={args.dominant or 'uniform'}")
    print("=" * 64)
    cos_label = ("cos(w_hat, nominal-prior) [NO ground truth]" if args.use_recorded
                 else "cos_bio(w*, w_hat)")
    print(f"best iter {best}  q_norm {irl.q_norm[best]:.3f}  |  shared-w {cos_label} = {cos_bio:.4f}")
    ws = wstar_vec / max(np.abs(wstar_vec[bio_idx]).max(), 1e-12)
    wh = w_hat / max(np.abs(w_hat[bio_idx]).max(), 1e-12)
    w_col = "prior" if args.use_recorded else "w*"
    print(f"\n{'feature':14s}{w_col:>10s}{'w_hat':>10s}")
    for k, a, b in zip(keys, ws, wh):
        print(f"{k:14s}{a:10.3f}{b:10.3f}{'' if k in biomech else '  (task)'}")
    try:
        _qrj = np.asarray(getattr(irl, "q_rmse", [[]])[best], float)
    except Exception:
        _qrj = np.zeros(0)
    try:
        _m0 = (getattr(irl, "models", None) or [getattr(irl, "model", None)])[0]
        _jn = list(_m0.pin_model.names)[1:]
    except Exception:
        _jn = []
    print("\nper-joint RMSE (deg) at best iterate:")
    for _n, _e in zip(_jn, _qrj):
        print(f"   {_n:24s}{float(_e):7.2f}")
    np.savez(os.path.join(args.outdir, "population_recovery.npz"),
             subjects=np.array(subjects), keys=np.array(keys),
             q_rmse_perjoint=_qrj, joint_names=np.array(_jn),
             wstar=wstar_vec, w_hat=w_hat, w_hat_full=w_hat_full, cos_bio=cos_bio,
             weight_mode=weight_mode, n_w=int(args.n_w),
             K=int(args.K or args.n_w), B_window=B_window,
             use_recorded=bool(args.use_recorded),
             static_stick=bool(args.static_stick),
             target_force=float(args.target_force),
             measured_force=bool(args.measured_force),
             avg_force=bool(args.avg_force),
             contact_windows=json.dumps(contact_windows),
             force_scales=json.dumps(force_scales),
             tau_split=str(args.tau_split),
             tv_wstar=bool(args.tv_wstar),
             tv_ground_truth=(np.asarray(tv_ground_truth) if tv_ground_truth is not None
                              else np.zeros((0, 0))),
             pair_subjects=np.array([p[0] for p in pairs]),
             pair_tasks=np.array([p[1] for p in pairs]),
             pair_cycles=np.array([p[2] for p in pairs]),
             demos_xs=np.asarray(demos_xs), demos_us=np.asarray(demos_us),
             q_norm=np.asarray(getattr(irl, "q_norm", [])),
             opt_div=np.asarray(getattr(irl, "opt_div", []), float),
             best_iter=int(best),
             ws_all=np.asarray([np.asarray(w[0], float) for w in getattr(irl, "ws", [])]),
             sigma=np.asarray(getattr(irl, "_feature_scale", np.ones(nr)), float),
             keys_run=np.array([str(k) for k in getattr(irl, "keys_run", keys)]))
    print(f"\nsaved -> {args.outdir}/population_recovery.npz")

    if getattr(args, "dump_anim", False):
        _Xs = [np.asarray(x, float) for x in getattr(irl, "Xs", [])]
        _Us = [np.asarray(u, float) for u in getattr(irl, "Us", [])]
        _nd = int(getattr(irl, "n_demos", 1))
        _dx = np.asarray(demos_xs)
        _tfp = getattr(models[0], "target_force_profile", None) if models else None
        _tfp = np.asarray(_tfp, float).ravel() if _tfp is not None else np.array([])
        np.savez(os.path.join(args.outdir, "ocp_anim.npz"),
                 keys_run=np.array(keys), n_w=int(args.n_w), K=int(args.K or args.n_w),
                 T=int(_dx.shape[1] - 1) if _dx.ndim >= 2 else 0, n_demos=_nd,
                 w_iters=np.asarray([np.asarray(w[0], float) for w in getattr(irl, "ws", [])]),
                 q_norm_iters=np.asarray(getattr(irl, "q_norm", []), float),
                 opt_div_iters=np.asarray(getattr(irl, "opt_div", []), float),
                 Xs=np.array(_Xs, dtype=object), Us=np.array(_Us, dtype=object),
                 demos_xs=_dx, demos_us=np.asarray(demos_us), B_window=B_window,
                 target_force=float(args.target_force), target_force_profile=_tfp,
                 subject=str(subjects[0]) if subjects else "", task=str(args.task),
                 force_strict_slack_frac=float(getattr(args, "force_strict_slack_frac", 0.0)))
        print(f"[dump_anim] saved {args.outdir}/ocp_anim.npz: "
              f"{len(getattr(irl,'ws',[]))} weight-iterates, {len(_Xs)} Xs entries "
              f"(n_demos={_nd}, T={int(_dx.shape[1]-1) if _dx.ndim>=2 else 0})")

if __name__ == "__main__":
    main()
