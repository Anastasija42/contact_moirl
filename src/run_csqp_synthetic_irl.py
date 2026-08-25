"""
run_csqp_synthetic_irl.py
=========================
One SYNTHETIC IRL recovery run on the final CSQP human model:

  1. Build HumanCrocoddyl (contact_aware_cost=True) at a KNOWN ground-truth w*.
  2. Solve the OCP at w*  ->  that solved (xs*, us*) IS the synthetic demo.
  3. Run MO_IRL from a naive init; check whether it recovers w*.
  4. Re-solve at the recovered weights -> IRL rollout. Save demo + rollout
     trajectories and contact forces for plotting.

`--dominant FEATURE` makes ONE biomechanical feature dominant (w*=dominant_val)
and the rest small (low_val); without it, w* is near-uniform (the hardest,
least-identifiable target). progress_vel is always the task driver.

Recovery is the direction cosine between max-abs-normalised weight vectors
(absolute scale is not identifiable in MaxEnt IRL), plus a per-feature table.

Usage (unified_env):
    conda run -n unified_env python run_csqp_synthetic_irl.py \
        --dominant Eng_shoulder --max_iter 20 --line_search_steps 8 \
        --outdir toy_irl_study/csqp_synth_irl/eng_shoulder
"""
import argparse
import os
import numpy as np

from run_csqp_identifiability import load_geometry, build_model, ENG_SPLIT

BIOMECH = ["Tau", "JA", "JV", "JTC", "Geo",
           "Eng_thoracic", "Eng_clavicle", "Eng_shoulder", "Eng_elbow", "Eng_wrist"]

def cosine(a, b):
    a = np.asarray(a, float) / max(np.max(np.abs(a)), 1e-12)
    b = np.asarray(b, float) / max(np.max(np.abs(b)), 1e-12)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def contact_forces(human):
    """Per-step contact reaction at 'rail_sliding' (3D linear) on the current
    solver solution. Returns (T,3)."""
    human.solver.problem.calc(human.solver.xs, human.solver.us)
    f = []
    for d in human.solver.problem.runningDatas:
        try:
            cd = d.differential.multibody.contacts.contacts["rail_sliding"]
            f.append(np.asarray(cd.f.linear).copy())
        except (KeyError, AttributeError):
            f.append(np.zeros(3))
    return np.asarray(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S2")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--dominant", default=None,
                    help="biomech feature to make dominant in w* (else near-uniform)")
    ap.add_argument("--dominant_val", type=float, default=1.0)
    ap.add_argument("--low_val", type=float, default=1e-3)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--line_search_steps", type=int, default=8)
    ap.add_argument("--outdir", default="toy_irl_study/csqp_synth_irl")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from MO_IRL import MO_IRL
    from run_moirl_batch import _build_csqp_mask

    q0_full, q_traj, dt = load_geometry(args.subject, args.task, args.cycle)
    human, _ = build_model(args.subject, args.task, q0_full, q_traj, dt, ENG_SPLIT)
    human.q_traj = q_traj
    keys = list(human.keys_run)

    low = args.dominant_val if args.dominant is None else args.low_val
    wstar = {k: (low if args.dominant is None else args.low_val) for k in BIOMECH}
    if args.dominant is not None:
        if args.dominant not in BIOMECH:
            raise SystemExit(f"--dominant must be one of {BIOMECH}")
        wstar[args.dominant] = args.dominant_val
    else:
        wstar = {k: 0.01 for k in BIOMECH}
    wstar.update({"progress_vel": 1.0, "press_force": 0.1, "rail_lat": 1e-6, "rock_ori": 1e-6})
    wstar_vec = np.array([wstar[k] for k in keys], float)
    tag = args.dominant or "uniform"
    print(f"[synth] target = {tag}")
    print(f"[synth] w* = {dict(zip(keys, np.round(wstar_vec, 4)))}")

    xs_init = [np.concatenate([q_traj[i], np.zeros(human.nv)]) for i in range(len(q_traj))]
    us_init = [np.zeros(human.nu) for _ in range(len(q_traj) - 1)]
    human.update_solver_weights(wstar, {})
    xs_demo, us_demo = human.solve(xs_init=list(xs_init), us_init=list(us_init))
    xs_demo, us_demo = np.asarray(xs_demo), np.asarray(us_demo)
    f_demo = contact_forces(human)
    print(f"[synth] demo: xs {xs_demo.shape}  feasible={getattr(human.solver,'isFeasible','?')}"
          f"  KKT={getattr(human.solver,'KKT',float('nan')):.2e}")

    w_run = {k: 1e-6 for k in keys}
    w_run["progress_vel"] = 1.0
    irl_args = {
        "model": human, "w_run": w_run, "w_term": {},
        "xs_opt": [xs_demo], "us_opt": [us_demo],
        "irl_iter": 1000, "stopping": "q_norm", "tol": 1e-10,
        "max_iter": args.max_iter, "min_iter": 1,
        "compare_desired": False, "verbose": True, "n_w": 1,
    }
    mo_args = {
        "with_temp_adjust": False, "sqp_iter": 50, "next_traj": "worst",
        "line_search_steps": args.line_search_steps, "line_search_base": "q_norm",
        "use_jac": True, "use_hess": False, "with_dmp": False, "basis_num": 20,
        "alpha_x": 1.5, "alpha_z": 50.0, "beta_z": 10, "noise_f": 50.0, "tau": 10.0,
        "rollout_N": 5, "opt_vars": "dw", "temperature": 1, "l_type": 1,
        "l_reg": "elastic", "use_bad": False, "normalize_w": False,
        "normalize_features": False, "normalizing_thrs": 1.0, "K_set": 1,
        "N_samples": 50, "scaled_sum": True, "use_best": False,
        "Lambda": 1e-5, "Beta": 1e-8, "dyn_reg": False, "kappa": 9.0,
        "lambda_thrs": 1e-5, "lambda_init": 1.0, "delta": 1.0,
        "stopping": "q_norm", "tol": 1e-4,
        "gradient_mask": _build_csqp_mask(sorted(w_run.keys()), [], 1, None),
        "use_mppi_grad": False, "verbose_irl": False,
        "per_feature_scale": False, "q_norm_force_weight": 0.0,
    }
    print(f"[synth] MO_IRL max_iter={args.max_iter} ls={args.line_search_steps} ...", flush=True)
    irl = MO_IRL(mo_args, irl_args)
    irl.solve()
    best = int(np.argmin(irl.q_norm)) if len(getattr(irl, "q_norm", [])) else -1
    w_hat = np.asarray(irl.ws[best][0], float)[:len(keys)]

    human.update_solver_weights(dict(zip(keys, w_hat)), {})
    xs_irl, us_irl = human.solve(xs_init=list(xs_init), us_init=list(us_init))
    xs_irl, us_irl = np.asarray(xs_irl), np.asarray(us_irl)
    f_irl = contact_forces(human)

    cos = cosine(wstar_vec, w_hat)
    nq = human.nq
    q_d, q_i = xs_demo[:, :nq], xs_irl[:, :nq]
    Tm = min(len(q_d), len(q_i))
    joint_rmse_deg = np.sqrt(np.mean((q_d[:Tm] - q_i[:Tm]) ** 2, axis=0)) * 180 / np.pi
    print("\n" + "=" * 64)
    print(f"SYNTHETIC RECOVERY  ({tag})")
    print("=" * 64)
    print(f"best iter {best}  q_norm {irl.q_norm[best]:.3f} deg  |  cosine(w*,w_hat) = {cos:.4f}")
    print(f"per-joint RMSE (deg): {np.round(joint_rmse_deg, 2)}")
    ws_n = wstar_vec / max(np.abs(wstar_vec).max(), 1e-12)
    wh_n = w_hat / max(np.abs(w_hat).max(), 1e-12)
    print(f"\n{'feature':14s}{'w*':>10s}{'w_hat':>10s}")
    for k, a, b in zip(keys, ws_n, wh_n):
        print(f"{k:14s}{a:10.3f}{b:10.3f}")

    np.savez(os.path.join(args.outdir, "synth_recovery.npz"),
             keys=np.array(keys), wstar=wstar_vec, w_hat=w_hat,
             q_norm=np.asarray(getattr(irl, "q_norm", [])),
             cosine=cos, tag=tag, dt=dt,
             xs_demo=xs_demo, us_demo=us_demo, f_demo=f_demo,
             xs_irl=xs_irl, us_irl=us_irl, f_irl=f_irl,
             nq=nq, nv=human.nv, target_force=float(human.target_force))
    print(f"\nsaved -> {args.outdir}/synth_recovery.npz")

if __name__ == "__main__":
    main()
