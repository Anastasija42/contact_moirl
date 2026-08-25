"""
plot_paper_figures.py
=====================
Publication-ready figures for the cross-morphology Results section, split out of
the busy 5-panel compare__*.png diagnostic. Reads the forward-transfer outputs
({species}__{task}/forward.npz from run_species_forward.py --generated) and emits
clean single-message figures:

  R2  per-joint movement divergence vs human (RMSE, deg)        [neutral_start data]
  R4  cost re-partition (w*phi per feature) + KKT/cost strain    [neutral_start data]
  R5  contact-force profile + reach gap per taxon                [fixed_stick data]

Colorblind-safe Okabe-Ito palette, taxa ordered ape->human along the brachial
gradient. Each figure is independent so it can be dropped straight into the .tex.

Usage (unified_env):
  conda run -n unified_env python morphologies_study/plot_paper_figures.py \
      --indir analysis/special/pop_downlong/forward_transfer \
      --task down_long --figs R2 R4
  # R5 (reach/force) wants the fixed_stick run:
  conda run -n unified_env python morphologies_study/plot_paper_figures.py \
      --indir analysis/special/pop_downlong/forward_fixedstick --figs R5
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["bonobo", "chimp", "australopithecus_prometheus", "australopithecus_sediba",
         "homo_naledi", "homo_neanderthal", "human"]
LABELS = {"human": "Modern human", "homo_neanderthal": "Neanderthal",
          "homo_naledi": "H. naledi", "australopithecus_sediba": "A. sediba",
          "australopithecus_prometheus": "A. prometheus", "chimp": "Chimpanzee",
          "bonobo": "Bonobo"}
COLOR = {"human": "#000000", "homo_neanderthal": "#009E73", "homo_naledi": "#0072B2",
         "australopithecus_sediba": "#E69F00", "australopithecus_prometheus": "#56B4E9",
         "chimp": "#D55E00", "bonobo": "#CC79A7"}
JOINTS = ["thoracic_X", "clavicle_X", "shoulder_Z", "shoulder_X", "shoulder_Y",
          "elbow_Z", "elbow_Y", "wrist_Z", "wrist_X"]

def load(indir, task):
    runs = {}
    for sp in ORDER:
        f = os.path.join(indir, f"{sp}__{task}", "forward.npz")
        if os.path.exists(f):
            runs[sp] = np.load(f, allow_pickle=True)
    if not runs:
        raise SystemExit(f"no forward.npz under {indir} for task {task}")
    return runs

def fig_R2(runs, task, outdir):
    """Per-joint RMSE vs human -- the headline 'where each body re-routes'."""
    if "human" not in runs:
        print("[R2] no human run -> skip"); return
    naj = min(9, int(runs["human"]["nq"]))
    qh = runs["human"]["xs"][:, :naj]
    others = [sp for sp in ORDER if sp in runs and sp != "human"]
    x = np.arange(naj); wbar = 0.82 / len(others)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, sp in enumerate(others):
        qs = runs[sp]["xs"][:, :naj]; T = min(len(qh), len(qs))
        rmse = np.sqrt(np.mean((qh[:T] - qs[:T]) ** 2, axis=0)) * 180 / np.pi
        ax.bar(x + i * wbar, rmse, wbar, color=COLOR[sp], label=LABELS[sp])
    ax.set_xticks(x + 0.41 - wbar / 2)
    ax.set_xticklabels([JOINTS[j] if j < len(JOINTS) else str(j) for j in range(naj)],
                       rotation=20, ha="right")
    ax.set_ylabel("RMSE vs human (deg)")
    ax.set_title("Where the shared cost re-routes the movement, by body plan")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    out = os.path.join(outdir, f"R2_divergence__{task}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")

def fig_R4(runs, task, outdir):
    """Top: per-feature w*phi per taxon (log). Bottom: KKT/cost strain index."""
    keys = [str(k) for k in next(iter(runs.values()))["keys"]]
    sps = [sp for sp in ORDER if sp in runs]
    cc = {sp: np.abs(runs[sp]["cost_contrib"]) for sp in sps}
    mean_mag = np.mean([cc[sp] for sp in sps], axis=0)
    sel = sorted(np.argsort(mean_mag)[::-1][:7])
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.45))
    x = np.arange(len(sel)); wbar = 0.82 / len(sps)
    for i, sp in enumerate(sps):
        axA.bar(x + i * wbar, cc[sp][sel], wbar, color=COLOR[sp], label=LABELS[sp])
    axA.set_yscale("log")
    axA.set_xticks(x + 0.41 - wbar / 2)
    axA.set_xticklabels([keys[j] for j in sel], rotation=25, ha="right")
    axA.set_ylabel(r"$|w\cdot\phi|$  (log)")
    axA.set_title("Realised cost re-partitions across the limb (same fixed cost)")
    axA.grid(alpha=0.3, axis="y"); axA.legend(ncol=2, fontsize=8, frameon=False)
    strain = {sp: float(runs[sp]["kkt"]) / max(float(np.abs(runs[sp]["cost_contrib"]).sum()), 1e-9)
              for sp in sps}
    xb = np.arange(len(sps))
    axB.bar(xb, [strain[sp] * 100 for sp in sps], 0.6, color=[COLOR[sp] for sp in sps])
    axB.set_xticks(xb); axB.set_xticklabels([LABELS[sp] for sp in sps], rotation=20, ha="right")
    axB.set_ylabel("KKT / cost (%)")
    axB.set_title("How hard the human cost strains to fit each body")
    axB.grid(alpha=0.3, axis="y")
    out = os.path.join(outdir, f"R4_cost_strain__{task}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")

def _diverged(run, tf):
    """A rollout is diverged if its contact force is non-physical (>> the press
    target). The contact force is the weight-invariant signal: absolute KKT scales
    with the cost magnitude (a well-converged solve under large weights still has a
    large KKT), so it is only used as a very-high backstop, not the primary test."""
    kkt = float(run["kkt"]) if "kkt" in run.files else np.nan
    pk = float(np.linalg.norm(run["f_contact"], axis=1).max()) if "f_contact" in run.files else 0.0
    return pk > 10 * tf or (np.isfinite(kkt) and kkt > 1e8)

def fig_R5(runs, task, outdir):
    """Contact-force profile (sustainability) + reach-gap bar (reach)."""
    sps = [sp for sp in ORDER if sp in runs]
    tf = float(next(iter(runs.values()))["target_force"])
    fig, (axF, axR) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw=dict(width_ratios=[1.7, 1.0], wspace=0.28))
    diverged, ymax = [], tf * 1.3
    for sp in sps:
        f = np.linalg.norm(runs[sp]["f_contact"], axis=1)
        t = np.arange(len(f)) * float(runs[sp]["dt"])
        if _diverged(runs[sp], tf):
            diverged.append(sp); continue
        axF.plot(t, f, color=COLOR[sp], lw=1.8, label=LABELS[sp])
        ymax = max(ymax, float(np.percentile(f, 98)) * 1.15)
    axF.axhline(tf, color="gray", ls="--", lw=1, label=f"target {tf:.0f} N")
    axF.set_ylim(0, ymax)
    axF.set_xlabel("time (s)"); axF.set_ylabel("|F contact| (N)")
    axF.set_title("Contact sustainability under a shared world stick")
    if diverged:
        axF.text(0.02, 0.97, "diverged (OCP non-converged), omitted:\n" +
                 ", ".join(LABELS[s] for s in diverged), transform=axF.transAxes,
                 va="top", ha="left", fontsize=7, color="#B00",
                 bbox=dict(boxstyle="round", fc="#fff0f0", ec="#B00", lw=0.6))
    axF.grid(alpha=0.3); axF.legend(ncol=2, fontsize=7, frameon=False, loc="upper right")
    gaps = [float(runs[sp]["reach_gap"]) for sp in sps]
    xb = np.arange(len(sps))
    axR.bar(xb, gaps, 0.6, color=[COLOR[sp] for sp in sps],
            hatch=["//" if sp in diverged else "" for sp in sps],
            edgecolor="k", linewidth=0.5)
    axR.set_xticks(xb); axR.set_xticklabels([LABELS[sp] for sp in sps], rotation=25, ha="right")
    axR.set_ylabel("reach gap (m)")
    axR.set_title("Reach gap to the shared stick" +
                  ("  (// = force diverged)" if diverged else ""))
    axR.grid(alpha=0.3, axis="y")
    out = os.path.join(outdir, f"R5_reach_force__{task}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="folder of {species}__{task}/forward.npz")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--outdir", default=None, help="defaults to --indir")
    ap.add_argument("--figs", nargs="+", default=["R2", "R4"],
                    choices=["R2", "R4", "R5"])
    args = ap.parse_args()
    outdir = args.outdir or args.indir
    os.makedirs(outdir, exist_ok=True)
    runs = load(args.indir, args.task)
    print(f"[loaded] {len(runs)} taxa from {args.indir}")
    if "R2" in args.figs: fig_R2(runs, args.task, outdir)
    if "R4" in args.figs: fig_R4(runs, args.task, outdir)
    if "R5" in args.figs: fig_R5(runs, args.task, outdir)

if __name__ == "__main__":
    main()
