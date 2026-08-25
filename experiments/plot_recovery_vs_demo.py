#!/usr/bin/env python
"""Overlay the recovered CSQP rollout against the recorded demonstration:
joint positions, joint torques, and the contact force. Reads a rollout npz
(rollout_A.npz / rollout_B.npz) written by run_csqp_population_irl --replay_npz.

Recovered rollout is ALWAYS plotted against the recorded demo (never the demo
alone). Saves three PNGs: <prefix>_positions.png, _torques.png, _force.png.

Usage:
    python experiments/plot_recovery_vs_demo.py <rollout.npz> <out_prefix> "Title" [cycle]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"demo": "#0072B2", "rec": "#E69F00", "f": "#009E73"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
DEG = 180.0 / np.pi


def _clean(n):
    return (n.replace("right_", "").replace("_joint", "").replace("middle_", "")
             .replace("_X", " X").replace("_Y", " Y").replace("_Z", " Z"))


def main(npz, prefix, title, cyc=0):
    d = np.load(npz, allow_pickle=True)
    jn = [_clean(str(x)) for x in d["joint_names"]][:9]
    rx = np.asarray(d["rollout_xs"][cyc], float); dx = np.asarray(d["demos_xs"][cyc], float)
    ru = np.asarray(d["rollout_us"][cyc], float); du = np.asarray(d["demos_us"][cyc], float)
    rf = np.asarray(d["roll_force_profiles"][cyc], float)
    df = np.asarray(d["demo_force_targets"][cyc], float)
    tq = np.linspace(0, 1, len(rx)); tu = np.linspace(0, 1, len(ru))

    def grid(getter, unit, fname, sup):
        fig, ax = plt.subplots(3, 3, figsize=(11, 8.2))
        for j in range(9):
            a = ax.flat[j]
            dd, rr = getter(j)
            a.plot(tq[:len(dd)], dd, color=OI["demo"], lw=2.0, label="recorded demo")
            a.plot(tq[:len(rr)], rr, color=OI["rec"], lw=1.8, ls="--", label="recovered")
            a.set_title(jn[j], fontsize=9)
            if j >= 6: a.set_xlabel("stroke fraction")
            if j % 3 == 0: a.set_ylabel(unit)
        ax.flat[0].legend(fontsize=8, frameon=False, loc="best")
        fig.suptitle(f"{title}: {sup} (recovered vs demo)", fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"saved -> {fname}")

    grid(lambda j: (dx[:, j] * DEG, rx[:, j] * DEG), "deg", f"{prefix}_positions.png", "joint positions")
    grid(lambda j: (du[:, j], ru[:, j]), "N·m", f"{prefix}_torques.png", "joint torques")

    fig, a = plt.subplots(figsize=(7, 3.6))
    a.plot(tu[:len(df)], df, color=OI["demo"], lw=2.2, label="measured demo force")
    a.plot(tu[:len(rf)], rf, color=OI["f"], lw=2.0, ls="--", label="recovered force")
    a.fill_between(tu[:len(rf)], rf, df[:len(rf)], color=OI["f"], alpha=0.12)
    a.set_xlabel("stroke fraction"); a.set_ylabel("contact normal force (N)")
    a.set_title(f"{title}: contact force (recovered vs demo)")
    a.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{prefix}_force.png", dpi=140, bbox_inches="tight")
    print(f"saved -> {prefix}_force.png")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
