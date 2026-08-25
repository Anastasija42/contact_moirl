"""
plot_divergence_panels.py
=========================
Two-panel per-joint divergence figure for the morphology paper: the SAME fixed
cost re-solved on each body, shown for two stationings side by side ---

  LEFT  : free per-body workpiece station (own-stick / plain run)
  RIGHT : shared world workpiece (fixed-stick run)

Story: re-stationing freedom diffuses the divergence (left); forcing every body
onto the human's workpiece isolates the morphological re-routing (right) --- apes
at the elbow, H. naledi at the shoulder girdle, the Neanderthal barely. Shared
y-axis so the "diffuse+large vs localised" contrast reads directly. Reuses the
Okabe-Ito palette and joint/taxon ordering from plot_paper_figures.

Usage (unified_env):
  python morphologies_study/plot_divergence_panels.py \
    --plain  analysis/special/morpho_tt/forward_final_down_long \
    --fixed  analysis/special/morpho_tt/forward_final_fixedstick_down_long \
    --task down_long --out figures/morpho/R2_divergence.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paper_figures import ORDER, LABELS, COLOR, JOINTS, load


def per_joint_rmse(runs, naj):
    qh = runs["human"]["xs"][:, :naj]
    others = [sp for sp in ORDER if sp in runs and sp != "human"]
    out = {}
    for sp in others:
        qs = runs[sp]["xs"][:, :naj]
        T = min(len(qh), len(qs))
        out[sp] = np.sqrt(np.mean((qh[:T] - qs[:T]) ** 2, axis=0)) * 180 / np.pi
    return others, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="own-station forward dir")
    ap.add_argument("--fixed", required=True, help="shared-stick forward dir")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--out", default="figures/morpho/R2_divergence.png")
    args = ap.parse_args()

    panels = [("Free per-body station", load(args.plain, args.task)),
              ("Shared workpiece (human's)", load(args.fixed, args.task))]

    naj = min(9, int(panels[0][1]["human"]["nq"]))
    x = np.arange(naj)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True, sharey=True)
    handles = labels = None
    for ax, (title, runs) in zip(axes, panels):
        others, rmse = per_joint_rmse(runs, naj)
        wbar = 0.82 / len(others)
        for i, sp in enumerate(others):
            ax.bar(x + i * wbar, rmse[sp], wbar, color=COLOR[sp], label=LABELS[sp])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("RMSE vs human (deg)")
        ax.grid(alpha=0.3, axis="y")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[-1].set_xticks(x + 0.41 - wbar / 2)
    axes[-1].set_xticklabels([JOINTS[j] if j < len(JOINTS) else str(j) for j in range(naj)],
                             rotation=20, ha="right")
    fig.suptitle("Where the shared cost re-routes the movement, by body plan",
                 fontsize=14)
    fig.legend(handles, labels, ncol=6, fontsize=9, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
