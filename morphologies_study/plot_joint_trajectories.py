"""
plot_joint_trajectories.py
==========================
Joint-angle trajectories at the most telling joints, by body plan. For each joint
where the shared cost re-routes the movement most (elbow-Y = the long-armed-ape
signature; shoulder girdle = H. naledi), plot q(t) over the normalised stroke for
every taxon, human in bold black. Shows HOW each body deviates (the actual shape
of the compensation), not just the RMSE magnitude of Fig. R2.

Usage (unified_env):
  python morphologies_study/plot_joint_trajectories.py \
    --indir analysis/special/morpho_tt/forward_final_fixedstick_down_long \
    --task down_long --out figures/morpho/R3_joint_traj.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paper_figures import ORDER, LABELS, COLOR, JOINTS, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--joints", nargs="+",
                    default=["elbow_Y", "shoulder_Y", "shoulder_X", "clavicle_X"])
    ap.add_argument("--out", default="figures/morpho/R3_joint_traj.png")
    args = ap.parse_args()

    runs = load(args.indir, args.task)
    jidx = {name: i for i, name in enumerate(JOINTS)}
    sel = [(j, jidx[j]) for j in args.joints if j in jidx]
    ncol = 2
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.0 * nrow), sharex=True)
    axes = np.array(axes).ravel()
    handles = labels = None
    for ax, (jname, ji) in zip(axes, sel):
        for sp in ORDER:
            if sp not in runs:
                continue
            q = np.degrees(runs[sp]["xs"][:, ji])
            t = np.linspace(0, 1, len(q))
            ax.plot(t, q, color=COLOR[sp], lw=2.6 if sp == "human" else 1.6,
                    label=LABELS[sp], zorder=5 if sp == "human" else 2)
        ax.set_title(jname.replace("_", " "), fontsize=12)
        ax.set_ylabel("angle (deg)")
        ax.grid(alpha=0.3)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for ax in axes[len(sel):]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel("normalised stroke phase")
    fig.suptitle("Joint trajectories at the most telling joints", fontsize=13)
    fig.legend(handles, labels, ncol=7, fontsize=9, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
