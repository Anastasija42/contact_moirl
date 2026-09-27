"""Normal-force approximation and effective friction coefficient, from the shipped force slices.

Two figures and one table, from the per-cycle force-sensor slices under
``trajectories_from_mocap/27_02_sensor/<subject>/<stroke>/elaborated/force_slices/``:

  force_normal_justification.png
      (a) axial (-f_z) vs transverse |f_t| over a few cycles of one subject's long
          down-stroke; (b) transverse against normal during contact, with the
          through-origin friction slope; (c) off-axis angle against normal force.
  friction_mu_sweep.png
      per subject, the normalised residual of |f_t| - mu f_n over mu for each stroke.

  table (stdout): per (subject, stroke) the f_n^2-weighted slope
      mu_hat = sum(f_n |f_t|) / sum(f_n^2)   over the contact phase (f_n > FN_MIN),
  with a 5-95% bootstrap interval and the intercept fit |f_t| = m f_n + c.

mu_hat is an EFFECTIVE coefficient (sliding friction plus the cutting work of the
scrape), which is the quantity the contact model needs. Magnitude alone cannot
separate friction from a fixed tool tilt; the intercept c is a diagnostic of a
press-independent tangential component.

    python experiments/plot_friction_estimate.py [--out figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.insert(0, "src")
from repo_paths import REPO  # noqa: E402

try:
    import cb_style  # noqa: E402
    cb_style.apply()
    C = cb_style.OKABE_ITO
except Exception:  # pragma: no cover
    C = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

SUBJECTS = ["S1", "S2", "S3"]
STROKES = ["down_long", "down_short", "up_long", "up_short"]
DATA = REPO / "trajectories_from_mocap" / "27_02_sensor"
FN_MIN = 10.0        # contact threshold on f_n = -f_z [N]
MU_GRID = np.linspace(0.0, 1.0, 101)


def load_slices(subject, stroke):
    """All force-sensor slices of one recording, in time order, bias-removed.

    Each recording holds both stroke directions; the cycles of the recorded
    direction live under ``force_slices/<down|up>``. The sensor bias is taken as
    the mean over free samples (|f_z| < 3 N), which every slice contains at its
    ends, so the interaction force is what remains.
    """
    root = DATA / subject / stroke / "elaborated" / "force_slices"
    frames = []
    for d in ("down", "up"):
        for f in sorted((root / d).glob("cycle_*.csv")):
            df = pd.read_csv(f)
            df["dir"] = d
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames).sort_values("mocap_ref_time").reset_index(drop=True)
    free = df["fz"].abs() < 3.0
    if free.sum() > 50:
        for k in ("fx", "fy", "fz"):
            df[k] = df[k] - df.loc[free, k].mean()
    return df


def contact_samples(df, direction):
    d = df[df["dir"] == direction]
    fn = -d["fz"].to_numpy()
    ft = np.sqrt(d["fx"].to_numpy() ** 2 + d["fy"].to_numpy() ** 2)
    c = fn > FN_MIN
    return fn[c], ft[c]


def mu_weighted(fn, ft):
    return float(np.sum(fn * ft) / np.sum(fn ** 2))


def mu_intercept(fn, ft):
    A = np.vstack([fn, np.ones_like(fn)]).T
    (m, c), *_ = np.linalg.lstsq(A, ft, rcond=None)
    return float(m), float(c)


def bootstrap_ci(fn, ft, n=500):
    rs = np.random.RandomState(0)
    mus = np.empty(n)
    for i in range(n):
        idx = rs.randint(0, len(fn), len(fn))
        mus[i] = mu_weighted(fn[idx], ft[idx])
    return float(np.percentile(mus, 5)), float(np.percentile(mus, 95))


def residual_curve(fn, ft):
    r = np.array([np.sum((mu * fn - ft) ** 2) for mu in MU_GRID])
    return (r - r.min()) / (r.max() - r.min() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--panel_subject", default="S3", help="subject shown in panel (a)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    print(f"{'subject/stroke':16} | n     | mu_hat [5-95% CI]     | slope + intercept")
    print("-" * 66)
    for s in SUBJECTS:
        for st in STROKES:
            df = load_slices(s, st)
            if df is None:
                continue
            fn, ft = contact_samples(df, st.split("_")[0])
            if len(fn) < 50:
                print(f"{s + '/' + st:16} | {len(fn):5d} | (too little contact)")
                continue
            mu = mu_weighted(fn, ft)
            lo, hi = bootstrap_ci(fn, ft)
            m, c = mu_intercept(fn, ft)
            data[(s, st)] = (fn, ft)
            print(f"{s + '/' + st:16} | {len(fn):5d} | {mu:.3f} [{lo:.3f}, {hi:.3f}]  | {m:.3f} + {c:4.1f} N")
    print("-" * 66)
    fn_all = np.concatenate([v[0] for v in data.values()])
    ft_all = np.concatenate([v[1] for v in data.values()])
    mu_global = mu_weighted(fn_all, ft_all)
    fn_dl = np.concatenate([v[0] for k, v in data.items() if k[1] == "down_long"])
    ft_dl = np.concatenate([v[1] for k, v in data.items() if k[1] == "down_long"])
    print(f"{'pooled, all':16} | {len(fn_all):5d} | {mu_global:.3f}")
    print(f"{'pooled, down_long':16} | {len(fn_dl):5d} | {mu_weighted(fn_dl, ft_dl):.3f}")

    # ---- figure 1: the f_n = -f_z justification -----------------------------
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    axa = fig.add_subplot(gs[0, 0])
    df = load_slices(args.panel_subject, "down_long")
    t = df["mocap_ref_time"].to_numpy()
    fz = df["fz"].to_numpy()
    ft = np.sqrt(df["fx"].to_numpy() ** 2 + df["fy"].to_numpy() ** 2)
    fzs = savgol_filter(fz, 101, 3)
    fts = savgol_filter(ft, 101, 3)
    t0 = t[0] + 6.0
    w = (t > t0) & (t < t0 + 6.0)
    axa.plot(t[w] - t0, -fzs[w], color=C[0], lw=1.5, label=r"$-f_z$  (normal / press)")
    axa.plot(t[w] - t0, fts[w], color=C[1], lw=1.5, label=r"$|f_t|=\sqrt{f_x^2+f_y^2}$  (transverse)")
    axa.fill_between(t[w] - t0, 0, 1, where=(fzs[w] < -15.0), transform=axa.get_xaxis_transform(),
                     color=C[2], alpha=0.12, label="contact")
    axa.axhline(0, color="k", lw=0.6)
    axa.set(xlabel="time [s]", ylabel="force [N]",
            title=f"(a) {args.panel_subject} / down_long: axial dominates")
    axa.legend(fontsize=8, loc="upper right")

    N = fn_all
    T = ft_all
    ang = np.degrees(np.arctan(T / N))
    band = (np.degrees(np.arctan(0.2)), np.degrees(np.arctan(0.4)))

    axb = fig.add_subplot(gs[0, 1])
    sub = np.random.RandomState(0).choice(len(N), size=min(4000, len(N)), replace=False)
    axb.scatter(N[sub], T[sub], s=3, alpha=0.12, color=C[7], rasterized=True)
    xmax = np.percentile(N, 99)
    bins = np.linspace(0, xmax, 12)
    bc = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.median(T[(N >= bins[i]) & (N < bins[i + 1])])
                    if ((N >= bins[i]) & (N < bins[i + 1])).sum() > 30 else np.nan
                    for i in range(len(bins) - 1)])
    v = ~np.isnan(med)
    mu_fit = np.sum(bc[v] * med[v]) / np.sum(bc[v] ** 2)
    xx = np.linspace(0, xmax, 50)
    axb.plot(xx, mu_fit * xx, "-", color=C[3], lw=1.8, label=fr"friction $\mu\approx{mu_fit:.2f}$")
    axb.plot(bc, med, "o-", color="k", lw=1.8, ms=4, label="binned median")
    axb.set(xlabel=r"normal $|f_z|$ [N]", ylabel=r"transverse $|f_t|$ [N]",
            title=r"(b) transverse $\propto$ normal (friction-consistent)",
            xlim=(0, xmax), ylim=(0, np.percentile(T, 99)))
    axb.legend(fontsize=8, loc="upper left")

    axc = fig.add_subplot(gs[0, 2])
    binsn = np.linspace(np.percentile(N, 2), np.percentile(N, 98), 9)
    bcn = 0.5 * (binsn[1:] + binsn[:-1])
    meda, q1, q3 = [], [], []
    for i in range(len(binsn) - 1):
        m = (N >= binsn[i]) & (N < binsn[i + 1])
        if m.sum() > 30:
            meda.append(np.median(ang[m])); q1.append(np.percentile(ang[m], 25)); q3.append(np.percentile(ang[m], 75))
        else:
            meda.append(np.nan); q1.append(np.nan); q3.append(np.nan)
    axc.axhspan(band[0], band[1], color=C[2], alpha=0.18, label=r"friction $\mu\,0.2$–$0.4$")
    axc.fill_between(bcn, q1, q3, color=C[0], alpha=0.2)
    axc.plot(bcn, meda, "o-", color=C[0], lw=1.8, ms=4, label="median $\\pm$ IQR")
    axc.set(xlabel=r"normal force $|f_z|$ [N]", ylabel="off-axis angle [deg]",
            title=r"(c) near-axial at firm press $\Rightarrow$ $-f_z\!\approx$ normal")
    axc.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "force_normal_justification.png", dpi=160, bbox_inches="tight")
    print(f"saved {out / 'force_normal_justification.png'}: slope mu~{mu_fit:.2f}, "
          f"off-axis {meda[-1]:.0f} deg at firm press, {meda[0]:.0f} deg at light contact")

    # ---- figure 2: the mu sweep ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    tcol = {st: C[i] for i, st in enumerate(STROKES)}
    for ax, s in zip(axes, SUBJECTS):
        for st in STROKES:
            if (s, st) not in data:
                continue
            fn, ft = data[(s, st)]
            r = residual_curve(fn, ft)
            mh = MU_GRID[np.argmin(r)]
            ax.plot(MU_GRID, r, color=tcol[st], lw=1.8, label=f"{st} ($\\hat\\mu$={mh:.2f})")
            ax.axvline(mh, color=tcol[st], ls=":", lw=1, alpha=0.6)
        ax.axvline(mu_global, color="k", ls="--", lw=1.2, alpha=0.8)
        ax.set(xlabel=r"$\mu$", title=s, xlim=(0, 1))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("normalised fit residual")
    fig.suptitle(rf"Friction $\mu$ sweep: residual of $|f_t|-\mu f_n$ (dashed = pooled $\hat\mu$={mu_global:.2f})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "friction_mu_sweep.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'friction_mu_sweep.png'}")


if __name__ == "__main__":
    main()
