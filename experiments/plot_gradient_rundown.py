#!/usr/bin/env python
"""Plot the MO-IRL rundown (q_norm and optimality divergence per iteration).

Usage:
    python experiments/plot_gradient_rundown.py <population_recovery.npz> <out.png> ["title"]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def main(npz_path, out_png, title=None):
    d = np.load(npz_path, allow_pickle=True)
    q = np.asarray(d["q_norm"], float)
    od = np.asarray(d["opt_div"], float) if "opt_div" in d.files else np.full_like(q, np.nan)
    best = int(d["best_iter"]) if "best_iter" in d.files else int(np.argmin(q))
    keys = [str(k) for k in d["keys"]]
    w_hat = np.asarray(d["w_hat"], float)
    ws_all = np.asarray(d["ws_all"], float)
    it = np.arange(len(q))
    nf = len(keys)
    K = ws_all.shape[1] // nf if nf and ws_all.ndim == 2 and ws_all.shape[1] % nf == 0 else None

    W = None
    if K:
        for order in ((len(q), K, nf), (len(q), nf, K)):
            try:
                resh = ws_all.reshape(order)
                W = resh.mean(axis=1) if order[1] == K else resh.mean(axis=2)
                if W.shape == (len(q), nf):
                    break
            except ValueError:
                W = None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    a = ax[0]
    a.plot(it, q, "-o", color=OI[0], lw=2, label="joint error  $q_{\\mathrm{norm}}$")
    a.set_xlabel("IRL iteration"); a.set_ylabel("$q_{\\mathrm{norm}}$ (deg)", color=OI[0])
    a.tick_params(axis="y", colors=OI[0])
    a2 = a.twinx(); a2.grid(False)
    a2.plot(it, od, "-s", color=OI[1], lw=1.6, alpha=0.8, label="feature div.  opt\\_div")
    a2.set_ylabel("opt\\_div", color=OI[1]); a2.tick_params(axis="y", colors=OI[1])
    a.axvline(best, ls="--", color="0.4", lw=1)
    a.annotate(f"selected\niter {best}", (best, q[best]), textcoords="offset points",
               xytext=(6, 8), fontsize=9, color="0.3")
    a.set_title("(A) IRL descent")

    b = ax[1]
    if W is not None:
        top = np.argsort(-np.abs(w_hat))[:6]
        for i, f in enumerate(top):
            b.plot(it, W[:, f], "-o", ms=3, color=OI[i % len(OI)], label=keys[f])
        b.legend(fontsize=8, ncol=2, frameon=False)
        b.set_xlabel("IRL iteration"); b.set_ylabel("time-avg weight")
    else:
        b.text(0.5, 0.5, "weight history\nunavailable", ha="center", va="center")
    b.set_title("(B) leading weights over iterations")

    c = ax[2]
    SEED = {"progress_vel", "rock_ori", "rail_lat", "approach", "surface", "traveled"}
    order = np.argsort(np.abs(w_hat))
    cols = ["0.75" if keys[i] in SEED else OI[0] for i in order]
    c.barh([keys[i] for i in order], w_hat[order], color=cols)
    c.set_xlabel("recovered weight  $\\hat w$")
    c.set_title("(C) recovered cost  (grey = seeded task)")
    c.tick_params(axis="y", labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_png}   (iters={len(q)}, best={best}, "
          f"q_norm {q[0]:.2f} -> {q[best]:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
