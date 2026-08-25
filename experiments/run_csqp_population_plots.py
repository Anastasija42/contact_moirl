"""
run_csqp_population_plots.py
============================
Per-subject demo-vs-recovered plots for a population-IRL run
(run_csqp_population_irl.py output). For EACH (subject, cycle) demo it rebuilds
that subject's own-body model, solves the OCP at the ground-truth w* (the demo)
and at the recovered w_hat (loaded from population_recovery.npz), and plots:
  - joint angles: demo vs recovered rollout (per joint), and
  - contact force: demo vs recovered, against the target.

Reconstruction only (no IRL re-run); CSQP is deterministic so re-solving at the
saved w_hat reproduces the run's recovered rollout. Pass the SAME run config.

Usage (unified_env):
    conda run -n unified_env python run_csqp_population_plots.py \
        --subjects S2,S3,S1 --task down_long --cycle 5 --n_cycles 1 \
        --dominant Tau --q_norm_force_weight 0.1 \
        --indir toy_irl_study/csqp_population_fs
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cb_style

from demo_prep import (consistent_us, load_avg_force_profile,
                       load_measured_force, resample_q)
from run_csqp_population_irl import BIOMECH
from run_csqp_identifiability import load_geometry, build_model, ENG_SPLIT
from run_csqp_synthetic_irl import contact_forces
from run_moirl_batch import _friction_aware_tau

JOINT_LABELS = ["thoracic", "clavicle", "shoulder_1", "shoulder_2", "shoulder_3",
                "elbow_1", "elbow_2", "wrist_1", "wrist_2", "rock_pad"]

_TASK_PRETTY = {"down_long": "long down-stroke", "up_long": "long up-stroke",
                "down_short": "short down-stroke", "up_short": "short up-stroke"}
_FEAT_PRETTY = {"press_force": "press force", "progress_vel": "progress speed",
                "rail_lat": "lateral rail", "rock_ori": "rock orientation",
                "JA": "joint acceleration", "JV": "joint velocity",
                "JTC": "joint-torque change", "Geo": "geodesic path"}

def pretty_task(t):
    return _TASK_PRETTY.get(t, str(t).replace("_", " "))

def pretty_feat(k):
    if k in _FEAT_PRETTY:
        return _FEAT_PRETTY[k]
    if k.startswith("Eng_"):
        return f"energy·{k[4:]}"
    if k.startswith("Tau_"):
        return f"torque·{k[4:]}"
    return k.replace("_", " ")

def tv_windows_from_npz(d, npz_keys, model_keys):
    """Rebuild the recovered per-window run weights (n_win, nr) from the saved
    provenance, mirroring LineSearch._push_weights_to_one, then reorder columns
    to the model's keys_run. Falls back to constant w_hat for legacy npz."""
    nr = len(npz_keys)
    mode = str(d["weight_mode"]) if "weight_mode" in d.files else "single"
    n_w = int(d["n_w"]) if "n_w" in d.files else 1
    K = int(d["K"]) if "K" in d.files else n_w
    full = (np.asarray(d["w_hat_full"], float) if "w_hat_full" in d.files
            else np.asarray(d["w_hat"], float))
    if mode == "basis":
        B = np.asarray(d["B_window"], float)
        theta = full[:K * nr].reshape(K, nr)
        W_win = B @ theta
    elif mode == "windowed" or n_w > 1:
        W_win = full[:n_w * nr].reshape(n_w, nr)
    else:
        W_win = full[:nr].reshape(1, nr)
    idx = [npz_keys.index(k) for k in model_keys]
    return W_win[:, idx], mode

def expand_to_nodes(W_win, T):
    """Piecewise-constant expansion to per-node weights, same mapping the solver
    uses in update_solver_weights_tv: node i -> window min(i//(T//n_w), n_w-1)."""
    n_w = len(W_win)
    ws = max(1, T // n_w)
    return np.stack([W_win[min(i // ws, n_w - 1)] for i in range(T)])

def smooth_basis_Wt(d, npz_keys, model_keys, npts=200):
    """Reconstruct the recovered W(t) at FULL time resolution for a BASIS run,
    W(t) = B(t) @ theta with the SAME row-normalized Gaussian basis the solver
    built (IRL._build_basis): centers evenly spaced, sigma = 1/(K-1) in
    normalized time. The npz only stores the window-averaged B_window (hence the
    step plot); theta = w_hat_full[:K*nr] gives the smooth curve back.

    Returns (tfrac (npts,), W (npts, len(model_keys)), True) for basis mode,
    else (None, None, False)."""
    mode = str(d["weight_mode"]) if "weight_mode" in d.files else "single"
    if mode != "basis":
        return None, None, False
    nr = len(npz_keys)
    n_w = int(d["n_w"]) if "n_w" in d.files else 1
    K = int(d["K"]) if "K" in d.files else n_w
    full = (np.asarray(d["w_hat_full"], float) if "w_hat_full" in d.files
            else np.asarray(d["w_hat"], float))
    theta = full[:K * nr].reshape(K, nr)
    tfrac = np.linspace(0.0, 1.0, npts)
    centers = np.linspace(0.0, 1.0, K)
    sigma = max(1.0 / max(K - 1, 1), 1e-6)
    B_raw = np.exp(-((tfrac[:, None] - centers[None, :]) ** 2) / (2.0 * sigma * sigma))
    B = B_raw / (B_raw.sum(axis=1, keepdims=True) + 1e-12)
    W = B @ theta
    idx = [npz_keys.index(k) for k in model_keys]
    return tfrac, W[:, idx], True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="S2,S3,S1")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--n_cycles", type=int, default=1)
    ap.add_argument("--dominant", default=None)
    ap.add_argument("--dominant_val", type=float, default=1.0)
    ap.add_argument("--low_val", type=float, default=0.01)
    ap.add_argument("--q_norm_force_weight", type=float, default=0.0)
    ap.add_argument("--indir", required=True, help="dir with population_recovery.npz")
    ap.add_argument("--hard_rail", action="store_true",
                    help="rebuild with the hard rail constraint (match a run that used "
                         "--hard_rail; not saved in the npz so pass it explicitly).")
    ap.add_argument("--hard_rail_tol", type=float, default=1e-3)
    ap.add_argument("--rail_from_contact", action="store_true",
                    help="re-fit rail from rock_contact_point (match a run trained "
                         "with --rail_from_contact; not saved in npz, pass explicitly).")
    ap.add_argument("--windowed_rail", action="store_true",
                    help="estimate rail from the contact-window segment only (match a "
                         "run trained with --windowed_rail; needed for sub-window "
                         "subjects like S2/S1).")
    ap.add_argument("--progress_vel_target_mode", action="store_true",
                    help="rebuild progress_vel in quadratic target mode (match the run).")
    ap.add_argument("--target_rail_vel", type=float, default=0.5,
                    help="pace target for target mode; <0 = demo average speed (match run).")
    ap.add_argument("--press_in_effort", action="store_true",
                    help="fold press into effort at rebuild (match a run trained with it).")
    ap.add_argument("--test_cycles", default=None,
                    help="comma list of HELD-OUT cycle ids to test generalization "
                         "(e.g. 9,10). For each subject, the recovered cost is solved "
                         "on these unseen cycles and compared to their recorded motion.")
    args = ap.parse_args()
    cb_style.apply(); C = cb_style.OKABE_ITO
    subjects = [s.strip() for s in args.subjects.split(",")]

    d = np.load(os.path.join(args.indir, "population_recovery.npz"), allow_pickle=True)
    keys = [str(k) for k in d["keys"]]
    w_hat = dict(zip(keys, np.asarray(d["w_hat"], float)))
    cos_bio = float(d["cos_bio"])
    use_recorded = bool(d["use_recorded"]) if "use_recorded" in d.files else False
    demos_xs = np.asarray(d["demos_xs"]) if "demos_xs" in d.files else None
    demos_us = np.asarray(d["demos_us"]) if "demos_us" in d.files else None
    build_eng = {k: 0.01 for k in keys if k.startswith("Eng_")
                 or k.startswith("Tau_") or k.startswith("Effort_")} or ENG_SPLIT
    build_drop = tuple(k for k in ("Tau",) if k not in keys)
    static_stick = bool(d["static_stick"]) if "static_stick" in d.files else False
    target_force = float(d["target_force"]) if "target_force" in d.files else 60.0
    measured_force = bool(d["measured_force"]) if "measured_force" in d.files else False
    avg_force = bool(d["avg_force"]) if "avg_force" in d.files else False
    import json as _json
    contact_windows = _json.loads(str(d["contact_windows"])) if "contact_windows" in d.files else {}
    force_scales = _json.loads(str(d["force_scales"])) if "force_scales" in d.files else {}

    wstar = {k: (args.low_val if args.dominant else 0.01) for k in BIOMECH}
    if args.dominant:
        wstar[args.dominant] = args.dominant_val
    wstar.update({"progress_vel": 1.0, "press_force": 0.1, "rail_lat": 1e-6, "rock_ori": 1e-6})

    if "pair_subjects" in d.files and "pair_cycles" in d.files:
        pairs = list(zip([str(s) for s in d["pair_subjects"]],
                         [int(c) for c in d["pair_cycles"]]))
        subjects = list(dict.fromkeys(p[0] for p in pairs))
    else:
        pairs = [(s, c) for s in subjects
                 for c in range(args.cycle, args.cycle + args.n_cycles)]
    _pop = "population" if len(subjects) > 1 else subjects[0]
    run_desc = f"{_pop} ({', '.join(subjects)}), {len(pairs)} demo cycles"
    geoms = {p: load_geometry(p[0], args.task, p[1]) for p in pairs}
    T_target = min(len(q) - 1 for (_, q, _) in geoms.values())

    def process_pair(s, c, xs_d_saved=None, us_d_saved=None):
        """Build the (s,c) model, get its recorded demo (saved or fresh), set the
        press target to the demo's own force (matching the driver), solve at the
        recovered W(t), and return a row dict. xs_d_saved=None => held-out cycle:
        build the recorded demo from scratch (q + friction-aware us)."""
        q0_full, q_traj, dt = load_geometry(s, args.task, c)
        q_rs = resample_q(q_traj, T_target)
        dt_rs = dt * (len(q_traj) - 1) / T_target
        cw = contact_windows.get(f"{s}/{args.task}")
        human, _ = build_model(s, args.task, q0_full, q_rs, dt_rs, build_eng,
                               drop=build_drop, stick_static=static_stick,
                               target_force=target_force, contact_window=cw,
                               hard_rail=args.hard_rail, hard_rail_tol=args.hard_rail_tol,
                               rail_from_contact=args.rail_from_contact,
                               windowed_rail=args.windowed_rail,
                               progress_vel_target_mode=args.progress_vel_target_mode,
                               target_rail_vel=args.target_rail_vel,
                               press_in_effort=args.press_in_effort)
        nq = human.nq
        if use_recorded:
            dq = np.gradient(q_rs, dt_rs, axis=0); ddq = np.gradient(dq, dt_rs, axis=0)
            fprof = None
            if avg_force:
                p = load_avg_force_profile(s, args.task, T_target)
                if p is not None:
                    fsc = force_scales.get(f"{s}/{args.task}", force_scales.get(s, 1.0))
                    fprof = np.asarray(p[:human.T], float) * float(fsc)
            elif measured_force:
                p = load_measured_force(s, args.task, c, T_target)
                if p is not None:
                    fprof = np.asarray(p[:human.T], float)
            if fprof is not None:
                human.set_force_target_profile(fprof)
            if xs_d_saved is not None:
                xs_d, us_d = np.asarray(xs_d_saved), np.asarray(us_d_saved)
            else:
                xs_d = np.column_stack([q_rs, dq])
                if fprof is not None:
                    us_d = consistent_us(human, q_rs, dq, ddq)[:, :human.nu]
                else:
                    us_d = _friction_aware_tau(human, q_rs, dq, ddq)[:T_target, :human.nu]
            human.solver.xs = list(xs_d); human.solver.us = list(us_d)
            if fprof is not None:
                f_d = fprof
            else:
                f_d = np.linalg.norm(contact_forces(human), axis=1)
        else:
            xi = [np.concatenate([q_rs[i], np.zeros(human.nv)]) for i in range(len(q_rs))]
            ui = [np.zeros(human.nu) for _ in range(len(q_rs) - 1)]
            human.update_solver_weights(wstar, {})
            xs_d, _ = human.solve(xs_init=list(xi), us_init=list(ui)); xs_d = np.asarray(xs_d)
            f_d = np.linalg.norm(contact_forces(human), axis=1)
        W_win, _ = tv_windows_from_npz(d, keys, list(human.keys_run))
        if W_win.shape[0] > 1:
            human.update_solver_weights_tv(W_win, None)
        else:
            human.update_solver_weights(
                {k: float(W_win[0, j]) for j, k in enumerate(human.keys_run)}, {})
        human.solver.solve(list(xs_d), [np.zeros(human.nu) for _ in range(len(xs_d) - 1)], 1000)
        xs_r = np.stack(human.solver.xs.copy())
        f_r = np.linalg.norm(contact_forces(human), axis=1)
        _, phis_r, _, _ = human.get_traj_features(list(xs_r), list(human.solver.us))
        phis_r = np.asarray(phis_r, float)
        W_node = expand_to_nodes(W_win, human.T); Tc = min(len(phis_r), len(W_node))
        rmse = np.sqrt(np.mean((xs_r[:, :nq] - xs_d[:, :nq]) ** 2)) * 180 / np.pi
        print(f"[plot] {s}/cyc{c}: mean joint RMSE {rmse:.2f} deg  "
              f"force RMSE {np.sqrt(np.mean((f_d[:len(f_r)]-f_r[:len(f_d)])**2)):.1f} N", flush=True)
        return dict(s=s, c=c, nq=nq, dt=dt_rs, tgt=float(human.target_force),
                    measured=bool(measured_force), prof=bool(fprof is not None),
                    q_d=xs_d[:, :nq], q_r=xs_r[:, :nq], f_d=f_d, f_r=f_r,
                    W_node=W_node[:Tc], contrib=(W_node[:Tc] * phis_r[:Tc]),
                    mkeys=list(human.keys_run))

    rows = [process_pair(s, c, (demos_xs[di] if (use_recorded and demos_xs is not None) else None),
                         (demos_us[di] if (use_recorded and demos_us is not None) else None))
            for di, (s, c) in enumerate(pairs)]

    test_rows = []
    if args.test_cycles:
        if ":" in args.test_cycles:
            per_subj = {}
            for grp in args.test_cycles.split(";"):
                sj, cs = grp.split(":")
                per_subj[sj.strip()] = [int(x) for x in cs.split(",")]
        else:
            flat = [int(x) for x in args.test_cycles.split(",")]
            per_subj = {s: flat for s in subjects}
        trained = {(s, c) for (s, c) in pairs}
        for s in subjects:
            for c in per_subj.get(s, []):
                if (s, c) in trained:
                    continue
                try:
                    test_rows.append({**process_pair(s, c), "_held_out": True})
                except Exception as e:
                    print(f"[plot] held-out {s}/cyc{c} skipped: {e}")

    ncol = 4
    from collections import OrderedDict
    by_subj = OrderedDict()
    for row in rows:
        by_subj.setdefault(row["s"], []).append(row)

    for s, srows in by_subj.items():
        nq = srows[0]["nq"]
        npan = nq + 1
        nrow = int(np.ceil(npan / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                                 squeeze=False)
        ncyc = len(srows)
        col_of = lambda i: plt.cm.viridis(i / max(ncyc - 1, 1))
        rmse_j = np.mean([np.sqrt(np.mean((r["q_r"] - r["q_d"]) ** 2, axis=0))
                          for r in srows], axis=0) * 180 / np.pi
        for j in range(nq):
            ax = axes[j // ncol][j % ncol]
            lbl = JOINT_LABELS[j] if j < len(JOINT_LABELS) else f"j{j}"
            for i, r in enumerate(srows):
                t = np.arange(len(r["q_d"])) * r["dt"]
                ax.plot(t, np.rad2deg(r["q_d"][:, j]), color=col_of(i), lw=1.6,
                        label=(f"cyc{r['c']} demo" if j == 0 else None))
                ax.plot(t, np.rad2deg(r["q_r"][:, j]), color=col_of(i), lw=1.3, ls="--",
                        label=(f"cyc{r['c']} recov" if j == 0 else None))
            ax.set_title(f"{lbl}  (mean RMSE {rmse_j[j]:.1f}°)", fontsize=9)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            if j == 0:
                ax.legend(fontsize=5, ncol=2); ax.set_ylabel("angle (deg)", fontsize=8)
        axf = axes[nq // ncol][nq % ncol]
        for i, r in enumerate(srows):
            tf = np.arange(len(r["f_d"])) * r["dt"]
            axf.plot(tf, r["f_d"], color=col_of(i), lw=1.6)
            axf.plot(np.arange(len(r["f_r"])) * r["dt"], r["f_r"], color=col_of(i),
                     lw=1.3, ls="--")
        if srows[0].get("measured") or srows[0].get("prof"):
            axf.set_title("contact force (— demo target profile, -- recovered)",
                          fontsize=9)
        else:
            axf.axhline(srows[0]["tgt"], color="k", ls=":", alpha=0.6,
                        label=f"target {srows[0]['tgt']:.0f}N")
            axf.set_title("contact force (— demo, -- recovered)", fontsize=9)
        axf.set_ylabel("|f| (N)", fontsize=8); axf.grid(alpha=0.3); axf.legend(fontsize=6)
        for k in range(npan, nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"{s}: recovered cost reproduces the {pretty_task(args.task)} "
                     f"(training fit, {ncyc} cycles)\n"
                     f"solid = recorded demo, dashed = recovered rollout   |   "
                     f"mean joint RMSE {float(np.mean(rmse_j)):.1f}°", fontsize=12)
        fig.tight_layout()
        out = os.path.join(args.indir, f"recovery_{s}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}", flush=True)

    if test_rows:
        by_subj_t = OrderedDict()
        for row in test_rows:
            by_subj_t.setdefault(row["s"], []).append(row)
        for s, srows in by_subj_t.items():
            nq = srows[0]["nq"]; npan = nq + 1; nrow = int(np.ceil(npan / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow), squeeze=False)
            ncyc = len(srows); col_of = lambda i: plt.cm.plasma(i / max(ncyc - 1, 1))
            rmse_j = np.mean([np.sqrt(np.mean((r["q_r"] - r["q_d"]) ** 2, axis=0))
                              for r in srows], axis=0) * 180 / np.pi
            for j in range(nq):
                ax = axes[j // ncol][j % ncol]
                lbl = JOINT_LABELS[j] if j < len(JOINT_LABELS) else f"j{j}"
                for i, r in enumerate(srows):
                    t = np.arange(len(r["q_d"])) * r["dt"]
                    ax.plot(t, np.rad2deg(r["q_d"][:, j]), color=col_of(i), lw=1.6,
                            label=(f"cyc{r['c']} held-out" if j == 0 else None))
                    ax.plot(t, np.rad2deg(r["q_r"][:, j]), color=col_of(i), lw=1.3, ls="--",
                            label=(f"cyc{r['c']} predicted" if j == 0 else None))
                ax.set_title(f"{lbl}  (RMSE {rmse_j[j]:.1f}°)", fontsize=9)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
                if j == 0:
                    ax.legend(fontsize=5, ncol=2); ax.set_ylabel("angle (deg)", fontsize=8)
            axf = axes[nq // ncol][nq % ncol]
            for i, r in enumerate(srows):
                axf.plot(np.arange(len(r["f_d"])) * r["dt"], r["f_d"], color=col_of(i), lw=1.6)
                axf.plot(np.arange(len(r["f_r"])) * r["dt"], r["f_r"], color=col_of(i), lw=1.3, ls="--")
            axf.set_title("contact force (— held-out, -- predicted)", fontsize=9)
            axf.grid(alpha=0.3)
            for k in range(npan, nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            fig.suptitle(f"{s}: generalization to a HELD-OUT {pretty_task(args.task)} cycle "
                         f"(cost trained on other cycles, applied to unseen data)\n"
                         f"solid = held-out demo, dashed = predicted rollout   |   "
                         f"mean joint RMSE {float(np.mean(rmse_j)):.1f}°", fontsize=12)
            fig.tight_layout()
            out = os.path.join(args.indir, f"test_recovery_{s}.png")
            fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
            print(f"saved -> {out}  [generalization]", flush=True)

        keyj = [(j, JOINT_LABELS[j]) for j in (5, 6) if j < rows[0]["nq"]]
        sty_s = cb_style.styles(len(by_subj_t))
        figc, axc = plt.subplots(1, len(keyj) + 1, figsize=(4.0 * (len(keyj) + 1), 3.2),
                                 squeeze=False)
        axc = axc[0]
        for si, (s, srows) in enumerate(by_subj_t.items()):
            r = srows[0]; col = sty_s[si]["color"]
            tf = np.linspace(0, 1, len(r["q_d"]))
            for pi, (j, lbl) in enumerate(keyj):
                axc[pi].plot(tf, np.rad2deg(r["q_d"][:, j]), color=col, lw=1.9,
                             label=(s if pi == 0 else None))
                axc[pi].plot(tf, np.rad2deg(r["q_r"][:, j]), color=col, lw=1.4, ls="--")
            ff = np.linspace(0, 1, len(r["f_d"]))
            axc[-1].plot(ff, r["f_d"], color=col, lw=1.9)
            axc[-1].plot(np.linspace(0, 1, len(r["f_r"])), r["f_r"], color=col, lw=1.4, ls="--")
        for pi, (j, lbl) in enumerate(keyj):
            axc[pi].set_title(f"{lbl.replace('_', ' ')} angle", fontsize=11)
            axc[pi].set_ylabel("angle (deg)", fontsize=9)
        axc[-1].set_title("contact press force", fontsize=11)
        axc[-1].set_ylabel("$|f|$ (N)", fontsize=9)
        for a in axc:
            a.set_xlabel("normalized stroke time", fontsize=9); a.grid(alpha=0.3)
        axc[0].legend(fontsize=9, title="— demo, -- predicted")
        figc.suptitle(f"Held-out {pretty_task(args.task)}: the recovered shared cost predicts "
                      f"unseen strokes across subjects", fontsize=12)
        figc.tight_layout()
        outc = os.path.join(args.indir, "clean_recovery.png")
        figc.savefig(outc, dpi=160, bbox_inches="tight"); plt.close(figc)
        print(f"saved -> {outc}  [compact main-text figure]", flush=True)

    mkeys = rows[0]["mkeys"]
    plot_keys = [k for k in mkeys if k in BIOMECH or k == "press_force"
                 or k.startswith("Tau_") or k.startswith("Eng_")]
    pk_idx = [mkeys.index(k) for k in plot_keys]
    W0 = rows[0]["W_node"]
    is_tv = W0.shape[0] > 1 and not np.allclose(W0, W0[0], atol=1e-12)
    tfrac = np.linspace(0, 1, len(W0))
    gt = np.asarray(d["tv_ground_truth"], float) if "tv_ground_truth" in d.files else np.zeros((0, 0))
    has_gt = gt.ndim == 2 and gt.shape[0] > 1
    if has_gt:
        idx = [keys.index(k) for k in mkeys]
        gt = gt[:, idx]
        gt = gt * (np.abs(W0).max() / (np.abs(gt).max() + 1e-12))
        tfrac_gt = np.linspace(0, 1, len(gt))
    tfrac_s, W_s, is_basis = smooth_basis_Wt(d, keys, plot_keys, npts=240)
    if is_basis:
        tfrac, Wp = tfrac_s, W_s
        shape_note = "Gaussian basis"
    else:
        tfrac, Wp = np.linspace(0, 1, len(W0)), W0[:, pk_idx]
        shape_note = "piecewise-constant"

    gt_note = "; faint band = ground-truth W*(t)" if has_gt else ""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    sty = cb_style.styles(len(plot_keys))
    for ci, k in enumerate(plot_keys):
        if has_gt:
            ax.plot(tfrac_gt, gt[:, ci], lw=5.0, color=sty[ci]["color"], alpha=0.18)
        ax.plot(tfrac, Wp[:, ci], lw=2.0, label=pretty_feat(k), **sty[ci])
    ax.set_xlabel("normalized stroke time  (0 = start, 1 = end)")
    ax.set_ylabel("recovered cost weight  W(t)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, title="cost feature")
    ax.set_title(f"Recovered shared cost weights over the {pretty_task(args.task)}\n"
                 f"{run_desc} — {shape_note} weights, time-normalised{gt_note}",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(args.indir, "weights_Wt.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}", flush=True)

    all_keys = list(mkeys)
    tf_a, W_a, isb_a = smooth_basis_Wt(d, keys, all_keys, npts=240)
    if isb_a:
        tfrac_a, Wa = tf_a, W_a
    else:
        W_win_a, _ = tv_windows_from_npz(d, keys, all_keys)
        Wa = expand_to_nodes(W_win_a, len(W0)) if W_win_a.shape[0] > 1 \
            else np.repeat(W_win_a, len(W0), axis=0)
        tfrac_a = np.linspace(0, 1, len(Wa))
    gt_a = gt if (has_gt and gt.shape[1] == len(all_keys)) else None
    nall = len(all_keys); ncol = 4; nrow = int(np.ceil(nall / ncol))
    figg, axg = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.1 * nrow),
                             squeeze=False, sharex=True)
    for pi, k in enumerate(all_keys):
        a = axg[pi // ncol][pi % ncol]; col = C[pi % len(C)]
        a.plot(tfrac_a, Wa[:, pi], lw=1.8, color=col)
        if gt_a is not None:
            a.plot(tfrac_gt, gt_a[:, pi], lw=1.2, ls="--", color=col, alpha=0.7)
        a.set_title(pretty_feat(k), fontsize=8.5); a.grid(alpha=0.3); a.tick_params(labelsize=6.5)
        pkmax = float(np.abs(Wa[:, pi]).max())
        if pkmax < 1e-4:
            a.text(0.5, 0.5, "≈0", transform=a.transAxes, ha="center",
                   va="center", fontsize=11, color="0.6")
        if pi % ncol == 0:
            a.set_ylabel("W(t)", fontsize=7.5)
        if pi // ncol == nrow - 1:
            a.set_xlabel("norm. time", fontsize=7.5)
    for k in range(nall, nrow * ncol):
        axg[k // ncol][k % ncol].axis("off")
    figg.suptitle(f"Recovered shared cost weights over the {pretty_task(args.task)} — "
                  f"all {nall} features\n{run_desc}; each panel auto-scaled, "
                  f"\"≈0\" marks an inactive feature", fontsize=12)
    figg.tight_layout()
    outg = os.path.join(args.indir, "weights_Wt_separate.png")
    figg.savefig(outg, dpi=140, bbox_inches="tight"); plt.close(figg)
    print(f"saved -> {outg}", flush=True)

    for s, srows in by_subj.items():
        Tc = min(r["contrib"].shape[0] for r in srows)
        contrib = np.mean([r["contrib"][:Tc] for r in srows], axis=0)
        tfrac = np.linspace(0, 1, Tc)
        cc = contrib[:, pk_idx]
        tot = np.abs(contrib).sum(axis=0)
        order = np.argsort(tot)[::-1]
        share = tot / (tot.sum() + 1e-12)
        rank = ", ".join(f"{rows[0]['mkeys'][i]}={share[i]*100:.0f}%"
                         for i in order[:6])
        print(f"[contrib-rank] {s} ({args.task}): {rank}", flush=True)
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
        fst = cb_style.fills(cc.shape[1])
        polys = axs[0].stackplot(tfrac, *[cc[:, i] for i in range(cc.shape[1])],
                                 labels=[pretty_feat(k) for k in plot_keys],
                                 colors=[f["facecolor"] for f in fst])
        for poly, f in zip(polys, fst):
            poly.set_hatch(f["hatch"]); poly.set_edgecolor("white"); poly.set_linewidth(0.3)
        axs[0].set_title("composition: share of running cost over the stroke", fontsize=10)
        axs[0].set_xlabel("normalized stroke time"); axs[0].set_ylabel("weighted cost  W(t)·φ(t)")
        axs[0].legend(fontsize=6, ncol=2, loc="upper left", title="cost feature")
        axs[0].grid(alpha=0.3)
        sty = cb_style.styles(len(plot_keys))
        for i, k in enumerate(plot_keys):
            axs[1].plot(tfrac, cc[:, i] + 1e-12, lw=1.8, label=pretty_feat(k), **sty[i])
        axs[1].set_yscale("log")
        axs[1].set_title("same contributions on a log scale (small terms visible)", fontsize=10)
        axs[1].set_xlabel("normalized stroke time"); axs[1].grid(alpha=0.3, which="both")
        axs[1].legend(fontsize=6, ncol=2, title="cost feature")
        fig.suptitle(f"{s}: where the recovered cost spends effort over the "
                     f"{pretty_task(args.task)} (mean over {len(srows)} cycles)", fontsize=12)
        fig.tight_layout()
        out = os.path.join(args.indir, f"costcontrib_{s}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"saved -> {out}", flush=True)

        ck = list(mkeys); ncc = len(ck)
        nrowc = int(np.ceil(ncc / ncol))
        figc, axc = plt.subplots(nrowc, ncol, figsize=(3.4 * ncol, 2.1 * nrowc),
                                 squeeze=False, sharex=True)
        for pi, k in enumerate(ck):
            a = axc[pi // ncol][pi % ncol]
            a.plot(tfrac, contrib[:Tc, pi], lw=1.8, color=C[pi % len(C)])
            a.set_title(pretty_feat(k), fontsize=8.5); a.grid(alpha=0.3); a.tick_params(labelsize=6.5)
            if float(np.abs(contrib[:Tc, pi]).max()) < 1e-9:
                a.text(0.5, 0.5, "≈0", transform=a.transAxes, ha="center",
                       va="center", fontsize=11, color="0.6")
            if pi % ncol == 0:
                a.set_ylabel("W·φ", fontsize=7.5)
            if pi // ncol == nrowc - 1:
                a.set_xlabel("norm. time", fontsize=7.5)
        for kk in range(ncc, nrowc * ncol):
            axc[kk // ncol][kk % ncol].axis("off")
        figc.suptitle(f"{s}: per-feature cost contribution W(t)·φ(t) over the "
                      f"{pretty_task(args.task)}\n(all {ncc} features, "
                      f"mean over {len(srows)} cycles, each panel auto-scaled)",
                      fontsize=12)
        figc.tight_layout()
        outc = os.path.join(args.indir, f"costcontrib_separate_{s}.png")
        figc.savefig(outc, dpi=140, bbox_inches="tight"); plt.close(figc)
        print(f"saved -> {outc}", flush=True)

    nq0 = rows[0]["nq"]
    npan = nq0 + 1
    nrow = int(np.ceil(npan / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow), squeeze=False)
    sty_rows = cb_style.styles(len(rows))
    for j in range(nq0):
        ax = axes[j // ncol][j % ncol]
        for ri, row in enumerate(rows):
            t = np.arange(len(row["q_d"])) * row["dt"]
            ax.plot(t, np.rad2deg(row["q_d"][:, j]), lw=1.5, **sty_rows[ri],
                    label=(f"{row['s']}/c{row['c']}" if j == 0 else None))
        ax.set_title(JOINT_LABELS[j] if j < len(JOINT_LABELS) else f"j{j}", fontsize=9)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=6); ax.set_ylabel("angle (deg)", fontsize=8)
    axf = axes[nq0 // ncol][nq0 % ncol]
    for ri, row in enumerate(rows):
        tf = np.arange(len(row["f_d"])) * row["dt"]
        axf.plot(tf, row["f_d"], lw=1.5, **sty_rows[ri],
                 label=f"{row['s']}/c{row['c']}")
    axf.set_title("contact press force (per-step target)", fontsize=9)
    axf.grid(alpha=0.3); axf.legend(fontsize=6)
    for k in range(npan, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    demo_kind = ("recorded mocap" if use_recorded
                 else "per-subject optimal at w*")
    fig.suptitle(f"Input demonstrations: {len(rows)} {pretty_task(args.task)} cycles "
                 f"({demo_kind})\n{run_desc}, one colour/dash per subject-cycle",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.indir, "demos_overlay.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}", flush=True)

if __name__ == "__main__":
    main()
