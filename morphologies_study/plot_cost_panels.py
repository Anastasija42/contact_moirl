"""
plot_cost_panels.py
===================
Two-panel realised-cost decomposition (free vs shared station) for the morphology
paper, companion to plot_divergence_panels. Shows |w*phi| per feature per taxon on
a shared log axis:

  LEFT  : free per-body workpiece station (own-stick / plain run)
  RIGHT : shared world workpiece (fixed-stick run)

The same fixed weights cash out into a wildly different total when each body may
move fast against its own station --- the long-armed apes pay an enormous cost
(left) --- whereas forcing all bodies onto the human's workpiece flattens it
(right). Shared log y so the contrast reads directly. Strain (KKT/cost) ladder is
dropped: all taxa now converge <3%, so it no longer separates them.

Usage (unified_env):
  python morphologies_study/plot_cost_panels.py \
    --plain analysis/special/morpho_tt/forward_final_down_long \
    --fixed analysis/special/morpho_tt/forward_final_fixedstick_down_long \
    --task down_long --out figures/morpho/R4_cost_strain.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paper_figures import ORDER, LABELS, COLOR, load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--fixed", default=None,
                    help="optional shared-stick dir; omit for a single-panel figure")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--out", default="figures/morpho/R4_cost_strain.png")
    ap.add_argument("--ntop", type=int, default=8)
    args = ap.parse_args()

    panels = [("Free per-body station", load(args.plain, args.task))]
    if args.fixed:
        panels.append(("Shared workpiece (human's)", load(args.fixed, args.task)))
    keys = [str(k) for k in next(iter(panels[0][1].values()))["keys"]]

    mags = [np.abs(runs[sp]["cost_contrib"]) for _, runs in panels for sp in runs]
    sel = sorted(np.argsort(np.mean(mags, axis=0))[::-1][:args.ntop])
    x = np.arange(len(sel))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4.6 * len(panels)),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    handles = labels = None
    for ax, (title, runs) in zip(axes, panels):
        sps = [sp for sp in ORDER if sp in runs]
        wbar = 0.82 / len(sps)
        for i, sp in enumerate(sps):
            cc = np.maximum(np.abs(runs[sp]["cost_contrib"])[sel], 1e-6)
            ax.bar(x + i * wbar, cc, wbar, color=COLOR[sp], label=LABELS[sp])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(r"$|w\cdot\phi|$  realised cost (log)")
        ax.grid(alpha=0.3, axis="y")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[-1].set_xticks(x + 0.41 - wbar / 2)
    axes[-1].set_xticklabels([keys[j] for j in sel], rotation=25, ha="right")
    fig.suptitle("Realised cost re-partitions across the limb (same fixed cost)",
                 fontsize=14)
    fig.legend(handles, labels, ncol=7, fontsize=9, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.03 if len(panels) > 1 else -0.16))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()
