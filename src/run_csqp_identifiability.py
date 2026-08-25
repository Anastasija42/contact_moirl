"""
run_csqp_identifiability.py
===========================
OCP feature-sensitivity / identifiability analysis on the final 9-DOF CSQP
human model (final_models.HumanCrocoddyl), the CSQP analogue of the toy
toy_feature_analysis.py. This is the "CSQP on synthetic data" experiment of the
ablation study.

Method (resolve-based, the numerical realisation of the KKT sensitivity in the
paper). At a weight vector w we re-solve the OCP at w ± eps (central finite
differences, CSQP is deterministic so this is clean), recompute the IRL features
phi via human.get_traj_features, and build the feature Jacobian

    H[i, j] = d phi_i / d w_j .

SVD(H) -> rank, condition number, and the LEFT-singular vectors for the ~0
singular values name the REDUNDANT features (dependent rows). This directly
answers: are the 5 per-joint Eng groups identifiable, or do they collide with
each other / with Tau / JV?

A/B: the SAME analysis with the 5-group Eng split vs a single global Eng, to
decide whether splitting Eng buys identifiable dimensions or just rank deficiency.

Requires the contact_aware_cost model (Tau/Eng/JTC == features). CSQP need not be
fully feasible — only deterministic — for the Jacobian to be clean.

Usage (run in the unified_env conda env):
    conda run -n unified_env python run_csqp_identifiability.py \
        --subject S2 --task down_long --cycle 5 --outdir toy_irl_study/csqp_ident
"""

from subject_lookup import find_subject_dir
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from repo_paths import REPO
DATE = os.environ.get("MOCAP_DATE", "27_02")
TASK_DIR = {"up_long": "up", "up_short": "up",
            "down_long": "down", "down_short": "down"}
SLICE_START, SLICE_END = 30, -1

W_STAR = {
    "Tau": 0.01, "JA": 0.01, "JV": 0.01, "JTC": 0.01, "Geo": 0.01,
    "progress_vel": 1.0, "press_force": 0.1, "rail_lat": 1e-6, "rock_ori": 1e-6,
}
ENG_SPLIT = {f"Eng_{g}": 0.01 for g in
             ("thoracic", "clavicle", "shoulder", "elbow", "wrist")}
ENG_GLOBAL = {"Eng": 0.01}
EFFORT_MERGED = {f"Effort_{g}": 0.01 for g in
                 ("thoracic", "clavicle", "shoulder", "elbow", "wrist")}
TAU_SPLIT_STAR = {"Tau_thoracic": 0.005, "Tau_clavicle": 0.005,
                  "Tau_shoulder": 0.03,  "Tau_elbow": 0.02, "Tau_wrist": 0.008}

def load_geometry(subject, task, cycle):
    """Mirror run_moirl_batch.build_common_models trajectory loading."""
    subj_dir = find_subject_dir(REPO / f"trajectories_from_mocap/{DATE}", subject)
    ik_csv = subj_dir / task / "ik_joint_angles.csv"
    q0_full = pd.read_csv(ik_csv).to_numpy()[0][2:]
    # Do NOT correct it here -- gravity (world -z) is entangled with the base orientation via
    # that was RECOVERED in this frame and blows up KKT. The ~94deg upright rotation belongs
    # only at the rendering / Cartesian-figure stage, never in the solved dynamics.
    cyc = np.load(subj_dir / task / "elaborated" / TASK_DIR[task]
                  / f"cycle_{cycle:02d}.npz", allow_pickle=True)
    q_ref = cyc["q_matrix"].astype(np.float64)
    cycle_keys = list(cyc["joint_names"]) if "joint_names" in cyc.files else []
    extras = [j for j in ("middle_thoracic_X", "right_clavicle_joint_X")
              if j not in cycle_keys]
    if extras:
        from utils_slice_trajectories import JOINT_TO_CSV
        df_ik = pd.read_csv(ik_csv)
        t_all = df_ik["Relative_Time[s]"].to_numpy(float)
        t_real = np.linspace(float(cyc["t_real_start"]), float(cyc["t_real_end"]),
                             len(q_ref))
        eq = np.zeros((len(q_ref), len(extras)))
        for k, jn in enumerate(extras):
            cv = df_ik[JOINT_TO_CSV[jn]].to_numpy(float)
            eq[:, k] = cv[[int(np.abs(t_all - t).argmin()) for t in t_real]]
        q_ref = np.hstack([eq, q_ref])
    q_traj = np.hstack([q_ref, np.zeros((len(q_ref), 1))])[SLICE_START:SLICE_END]
    dt = float(cyc["dt"]) if "dt" in cyc.files else 0.0083
    return q0_full, q_traj, dt

def build_model(subject, task, q0_full, q_traj, dt, eng_weights, drop=(),
                force_strict_slack_frac=0.0, stick_static=False, target_force=60.0,
                t_contact=None, contact_window=None, p_start=None, p_end=None,
                eps_abs=None, termination_tolerance=None, max_qp_iters=None,
                hard_rail=False, hard_rail_tol=1e-3, windowed_rail=False,
                rail_from_contact=False, contact_normal_slack=0.0,
                contact_mask_from_force=False, contact_mask_thresh=0.15,
                two_cost_force=False, force_max=80.0, press_in_actuation=False,
                press_normal_dual=True, press_friction_dual=True, press_in_effort=False,
                progress_vel_target_mode=False, target_rail_vel=0.5,
                lock_thorax=False, lock_thorax_tol=1e-3,
                effort_limits=False, effort_limit_scale=1.0,
                contact_normal_track=False, force_track_max=False, term_pos_slack=0.02):
    from final_models import HumanCrocoddyl
    w_run = dict(W_STAR)
    for k in drop:
        w_run.pop(k, None)
    w_run.update(eng_weights)
    if force_track_max:
        w_run['press_peak'] = 0.1
    if two_cost_force:
        if 'press_force' not in drop:
            w_run.setdefault('press_force', 0.1)
        if 'press_capacity' not in drop:
            w_run['press_capacity'] = 0.1

    args = dict(subject_id=subject.lower(), date=DATE, task=task,
                mu=0.25, target_force=target_force, contact="automatic", dt=dt,
                q0_full=q0_full, q0=q_traj[0], q_traj=q_traj, T=len(q_traj) - 1,
                force_strict_slack_frac=force_strict_slack_frac, w_run=w_run, w_term={},
                term_pos_slack=float(term_pos_slack),
                solver_type="CSQP", contact_aware_cost=True,
                stick_static=stick_static,
                force_two_cost=bool(two_cost_force), force_max=float(force_max),
                press_in_actuation=bool(press_in_actuation),
                press_normal_dual=bool(press_normal_dual),
                press_friction_dual=bool(press_friction_dual),
                press_in_effort=bool(press_in_effort),
                progress_vel_target_mode=bool(progress_vel_target_mode),
                target_rail_vel=float(target_rail_vel),
                lock_thorax=bool(lock_thorax),
                lock_thorax_tol=float(lock_thorax_tol),
                effort_limits=bool(effort_limits),
                effort_limit_scale=float(effort_limit_scale),
                contact_normal_track=bool(contact_normal_track))

    if contact_window is not None:
        args["contact_window"] = list(contact_window)
    elif t_contact is not None:
        args["T_contact"] = int(t_contact)
    if p_start is not None and p_end is not None:
        args["p_start"] = np.asarray(p_start, dtype=float)
        args["p_end"]   = np.asarray(p_end, dtype=float)
    if eps_abs is not None:
        args["eps_abs"] = float(eps_abs)
    if termination_tolerance is not None:
        args["termination_tolerance"] = float(termination_tolerance)
    if max_qp_iters is not None:
        args["max_qp_iters"] = int(max_qp_iters)
    if hard_rail:
        args["hard_rail"] = True
        args["hard_rail_tol"] = float(hard_rail_tol)
    if windowed_rail:
        args["windowed_rail"] = True
    if rail_from_contact:
        args["rail_from_contact"] = True
    if contact_normal_slack and float(contact_normal_slack) > 0.0:
        args["contact_normal_slack"] = float(contact_normal_slack)
    if contact_mask_from_force:
        args["contact_mask_from_force"] = True
        args["contact_mask_thresh"] = float(contact_mask_thresh)
    return HumanCrocoddyl(args=args), w_run

def feature_jacobian(human, w_run, *, eps_rel=0.25, eps_floor=2e-3):
    """H[i,j] = d phi_i / d w_j by central finite differences, re-solving the
    CSQP OCP at each perturbed weight. Returns (H, Hs, S, U, Vt, keys, phi0,
    rank, cond, feasibility-of-baseline)."""
    keys = list(human.keys_run)
    nr = len(keys)
    w0 = np.array([w_run[k] for k in keys], float)
    nv = human.nv
    xs_init = [np.concatenate([human.q_traj[i], np.zeros(nv)])
               for i in range(len(human.q_traj))]
    us_init = [np.zeros(human.nu) for _ in range(len(human.q_traj) - 1)]

    def solve_phi(wvec):
        human.update_solver_weights(dict(zip(keys, wvec)), {})
        human.solve(xs_init=list(xs_init), us_init=list(us_init))
        xs = np.asarray(human.solver.xs.tolist())
        us = np.asarray(human.solver.us.tolist())
        Phi, *_ = human.get_traj_features(xs, us)
        return np.asarray(Phi[:nr], float)

    phi0 = solve_phi(w0)
    base_feasible = bool(getattr(human.solver, "isFeasible", False))
    base_kkt = float(getattr(human.solver, "KKT", np.nan))
    H = np.zeros((nr, nr))
    for j in range(nr):
        e = max(eps_floor, eps_rel * abs(w0[j]))
        wp = w0.copy(); wp[j] += e
        wm = w0.copy(); wm[j] -= e
        H[:, j] = (solve_phi(wp) - solve_phi(wm)) / (2 * e)
        print(f"    [{j+1}/{nr}] d/d {keys[j]:14s} done", flush=True)

    Hs = H / np.maximum(np.abs(phi0), 1e-9)[:, None]
    U, S, Vt = np.linalg.svd(Hs)
    thr = S[0] * 1e-3 if S[0] > 0 else 1e-3
    rank = int(np.sum(S > thr))
    cond = float(S[0] / S[-1]) if S[-1] > 1e-300 else np.inf
    return dict(H=H, Hs=Hs, S=S, U=U, Vt=Vt, keys=keys, phi0=phi0, rank=rank,
                cond=cond, thr=thr, base_feasible=base_feasible, base_kkt=base_kkt)

def report(res, label):
    keys, S, U, rank, nr = res["keys"], res["S"], res["U"], res["rank"], len(res["keys"])
    print("\n" + "=" * 70)
    print(f"IDENTIFIABILITY  {label}")
    print("=" * 70)
    print(f"baseline solve: feasible={res['base_feasible']}  KKT={res['base_kkt']:.2e}")
    print(f"features ({nr}): {keys}")
    print(f"phi(w*): {np.round(res['phi0'], 3)}")
    print(f"\nrank = {rank}/{nr}    cond = {res['cond']:.3e}")
    print("singular values (feature-scaled):")
    for i, s in enumerate(S):
        tag = "load-bearing" if s > res["thr"] else "REDUNDANT/noise"
        print(f"  sigma_{i+1:2d} = {s:.4e}  [{tag}]")
    if rank < nr:
        print(f"\n{nr-rank} REDUNDANT direction(s) — left-singular vectors "
              f"(which features collide):")
        for idx in range(rank, nr):
            u = U[:, idx]
            order = np.argsort(np.abs(u))[::-1]
            top = [(keys[i], round(float(u[i]), 3)) for i in order if abs(u[i]) > 0.2]
            print(f"  null #{idx-rank+1}: {top}")
    else:
        print("\nall features locally identifiable (full rank)")
    if rank >= 1:
        uw = U[:, rank - 1]
        print(f"weakest identifiable mode dominated by: "
              f"{keys[int(np.argmax(np.abs(uw)))]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S2")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--outdir", default="toy_irl_study/csqp_ident")
    ap.add_argument("--mode", choices=["split", "global", "merged", "tausplit",
                                       "both", "all"], default="both")
    ap.add_argument("--press_force", type=float, default=None,
                    help="override press_force weight in w* (e.g. 0.5 to make "
                         "pressing a prominent learnable preference).")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    q0_full, q_traj, dt = load_geometry(args.subject, args.task, args.cycle)
    print(f"[geom] q_traj {q_traj.shape}  dt={dt:.5f}  T={len(q_traj)-1}")

    out = {}
    configs = []
    if args.mode in ("split", "both", "all"):
        configs.append(("Eng-split (Tau + 5 Eng)", ENG_SPLIT, "split", ()))
    if args.mode in ("global", "both", "all"):
        configs.append(("Eng-global (Tau + 1 Eng)", ENG_GLOBAL, "global", ()))
    if args.mode in ("merged", "all"):
        configs.append(("Effort-merged (5 Tau+Eng groups, no Tau)",
                        EFFORT_MERGED, "merged", ("Tau",)))
    if args.mode in ("tausplit", "all"):
        configs.append(("Tau-split (5 Tau_g + 5 Eng_g + press_force, no Tau)",
                        {**ENG_SPLIT, **TAU_SPLIT_STAR}, "tausplit", ("Tau",)))

    pf = args.press_force
    for label, eng_w, key, drop in configs:
        print(f"\n########## building {label} ##########")
        if pf is not None:
            eng_w = {**eng_w, "press_force": float(pf)}
        human, w_run = build_model(args.subject, args.task, q0_full, q_traj, dt,
                                   eng_w, drop=drop)
        human.q_traj = q_traj
        print(f"  keys_run ({len(human.keys_run)}): {human.keys_run}")
        res = feature_jacobian(human, w_run)
        report(res, label)
        out[key] = dict(keys=res["keys"], S=res["S"].tolist(),
                        rank=res["rank"], cond=res["cond"],
                        phi0=res["phi0"].tolist(),
                        base_feasible=res["base_feasible"], base_kkt=res["base_kkt"],
                        nullspace=[res["U"][:, i].tolist()
                                   for i in range(res["rank"], len(res["S"]))])
        np.savez(os.path.join(args.outdir, f"H_{key}.npz"),
                 H=res["H"], Hs=res["Hs"], S=res["S"], U=res["U"], Vt=res["Vt"],
                 keys=np.array(res["keys"]))

    with open(os.path.join(args.outdir, "identifiability.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {args.outdir}/identifiability.json (+ H_*.npz)")
    if len(out) > 1:
        print("\n=== effort-cost variants compared ===")
        print(f"  {'variant':10s}{'#feat':>6s}{'rank':>6s}{'cond':>12s}")
        for key in out:
            o = out[key]
            print(f"  {key:10s}{len(o['keys']):>6d}{o['rank']:>6d}{o['cond']:>12.2e}")
        print("  -> lower cond / fewer redundant directions = better-posed cost.")

if __name__ == "__main__":
    main()
