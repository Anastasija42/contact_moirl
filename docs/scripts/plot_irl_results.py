"""plot_irl_results.py — turn per-run csqp_irl outputs into figures embedded
in docs/results.md.

Reads everything that run_moirl_batch.py's method_csqp_irl writes:
    analysis/moirl/<combo>/<tag>/{q_norm.npy, ws_history.npz,
                                  weights.json, xs_irl.npz, force_target.npy}

and produces three figures under docs/assets/figures/:
    irl_<combo>__convergence.png   — q_norm vs iter for every n_w in the sweep
    irl_<combo>__weights.png       — final learned weights, grouped by n_w
                                     (bars per basis center within each block)
    irl_<combo>__force_target.png  — smoothed measured force used as target

Run from the repo root after a sweep finishes:

    python docs/scripts/plot_irl_results.py --combo miras_dl

By default it picks up every csqp_irl__nwN[__basis]/ subdirectory under
analysis/moirl/<combo>/.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO     = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis/moirl"
OUT_DIR  = REPO / "docs/assets/figures"

TAG_RE = re.compile(r"^csqp_irl__nw(\d+)__(windowed|basis|adaptive)(?:__.*)?$")


def _runs_for(combo: str, mode_filter=None, tag_substr=None):
    """Discover csqp_irl runs under analysis/moirl/<combo>.

    tag_substr: if given, only include directories whose name contains this
    substring. Use to filter to a specific run_id when the combo dir
    accumulates many sweeps (e.g. tag_substr='felix_0506_1623').
    """
    base = ANALYSIS / combo
    if not base.exists():
        return []
    rows = []
    for sub in sorted(base.iterdir()):
        m = TAG_RE.match(sub.name)
        if not m:
            continue
        if tag_substr and tag_substr not in sub.name:
            continue
        n_w = int(m.group(1))
        mode = m.group(2)
        if mode_filter and mode not in mode_filter:
            continue
        rows.append((n_w, mode, sub))
    return sorted(rows, key=lambda r: (r[1], r[0]))


def plot_convergence(combo, runs, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    plotted = 0
    style = {"windowed": "-", "basis": "--", "adaptive": ":"}
    for n_w, mode, run_dir in runs:
        q_path = run_dir / "q_norm.npy"
        if not q_path.exists():
            continue
        q = np.load(q_path)
        if q.size == 0:
            continue
        label = f"n_w={n_w}  {mode}"
        ax.plot(q, lw=1.6, ls=style.get(mode, "-"), label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False
    ax.set_xlabel("IRL iteration")
    ax.set_ylabel(r"$q_{\mathrm{norm}}$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{combo} — IRL convergence by parametrization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_weights(combo, runs, out_path):
    """One row per run; each row has a bar chart of the final learned weights
    (running + terminal). For n_w > 1 the bars are grouped per parameter
    block (window for 'windowed', basis center for 'basis'/'adaptive')."""
    panels = []
    for n_w, mode, run_dir in runs:
        ws_path = run_dir / "ws_history.npz"
        wj_path = run_dir / "weights.json"
        if not ws_path.exists() or not wj_path.exists():
            continue
        wsd = np.load(ws_path, allow_pickle=True)
        ws_run, ws_term = wsd["ws_run"], wsd["ws_term"]
        keys_run  = list(wsd["keys_run"])
        keys_term = list(wsd["keys_term"])
        if ws_run.shape[0] == 0:
            continue
        # last iteration's flat weight vector → reshape into K (or n_w) blocks
        K = ws_run.shape[1] // len(keys_run)
        if K * len(keys_run) != ws_run.shape[1]:
            print(f"[warn] {run_dir.name}: w_run len {ws_run.shape[1]} "
                  f"!= K*nr_run, skipping")
            continue
        Wr = ws_run[-1].reshape(K, len(keys_run))
        if ws_term.shape[1] == 0:
            Wt = np.zeros((K, 0))
        else:
            Wt = ws_term[-1].reshape(K, len(keys_term))
        panels.append((n_w, mode, K, keys_run, keys_term, Wr, Wt))

    if not panels:
        return False

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(9, 3.0 * len(panels)),
                             squeeze=False)
    for ax, (n_w, mode, K, keys_run, keys_term, Wr, Wt) in zip(axes[:, 0], panels):
        feat = list(keys_run) + list(keys_term)
        x = np.arange(len(feat))
        width = 0.8 / K
        cmap = plt.get_cmap("viridis", K)
        for k in range(K):
            block = np.concatenate([Wr[k],
                                     Wt[k] if Wt.size else np.zeros(0)])
            # normalize within the block so visual scale is comparable
            scale = max(np.max(np.abs(block)), 1e-12)
            ax.bar(x + (k - (K - 1) / 2) * width, block / scale,
                   width=width, color=cmap(k),
                   label=f"$\\theta_{k+1}$" if K > 1 else "$w$")
        ax.set_xticks(x)
        ax.set_xticklabels(feat, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("weight (block-normalised)")
        ax.set_title(f"n_w={n_w}  {mode}  (K={K})")
        ax.grid(True, axis="y", alpha=0.3)
        if K > 1:
            ax.legend(fontsize=8, ncol=K, loc="upper right")
    fig.suptitle(f"{combo} — learned weights per parameter block", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_force_target(combo, runs, out_path):
    for n_w, mode, run_dir in runs:
        fpath = run_dir / "force_target.npy"
        if not fpath.exists():
            continue
        f = np.load(fpath)
        if f.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(f, lw=1.6, color="tab:blue")
        ax.set_xlabel("timestep")
        ax.set_ylabel("force [N]")
        ax.set_title(f"{combo} — smoothed measured force target  "
                     f"(mean {f.mean():.1f} N, max {f.max():.1f} N)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return True
    return False


def plot_cost_contributions(combo, runs, out_path):
    """Per run, bar chart of w_block · phi_demo_mean for every (block, feature).

    This is the "true importance" the notebook prints: a feature with a small
    weight but huge magnitude can still dominate the cost. Mirrors the
    `cost_contrib_run = wr * phi_mean_run` pattern from MO_IRL_mocap.ipynb.
    """
    panels = []
    for n_w, mode, run_dir in runs:
        wsf  = run_dir / "ws_history.npz"
        phif = run_dir / "phi_demo.npy"
        if not (wsf.exists() and phif.exists()):
            continue
        wsd = np.load(wsf, allow_pickle=True)
        ws_run, ws_term = wsd["ws_run"], wsd["ws_term"]
        keys_run  = list(wsd["keys_run"])
        keys_term = list(wsd["keys_term"])
        if ws_run.shape[0] == 0:
            continue
        phis = np.load(phif)               # (n_demos, nr)
        nr_run  = len(keys_run)
        nr_term = len(keys_term)
        # Mean integrated phi across demos, then split into run/term.
        phi_mean = phis.mean(axis=0)
        phi_run_mean  = phi_mean[:nr_run]
        phi_term_mean = phi_mean[nr_run:nr_run + nr_term]

        K = ws_run.shape[1] // nr_run
        if K * nr_run != ws_run.shape[1]:
            continue
        Wr = ws_run[-1].reshape(K, nr_run)
        Wt = (ws_term[-1].reshape(K, nr_term) if ws_term.shape[1] else
              np.zeros((K, 0)))
        # Element-wise contribution per block.
        contrib_r = Wr * phi_run_mean[None, :]
        contrib_t = Wt * phi_term_mean[None, :] if Wt.size else np.zeros_like(Wt)
        panels.append((n_w, mode, K, keys_run, keys_term,
                        contrib_r, contrib_t))

    if not panels:
        return False

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(9, 3.0 * len(panels)),
                             squeeze=False)
    for ax, (n_w, mode, K, keys_run, keys_term, Cr, Ct) in zip(axes[:, 0], panels):
        feat = list(keys_run) + list(keys_term)
        x = np.arange(len(feat))
        width = 0.8 / K
        cmap = plt.get_cmap("plasma", K)
        for k in range(K):
            block = np.concatenate([Cr[k], Ct[k] if Ct.size else np.zeros(0)])
            scale = max(np.max(np.abs(block)), 1e-12)
            ax.bar(x + (k - (K - 1) / 2) * width, block / scale,
                   width=width, color=cmap(k),
                   label=(f"block {k+1}" if K > 1 else "all"))
        ax.set_xticks(x)
        ax.set_xticklabels(feat, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(r"$w_k \cdot \bar\phi$  (block-normalised)")
        ax.set_title(f"n_w={n_w}  {mode}  (K={K})")
        ax.grid(True, axis="y", alpha=0.3)
        if K > 1:
            ax.legend(fontsize=7, ncol=K, loc="upper right")
    fig.suptitle(f"{combo} — cost contributions per parameter block "
                 f"(weight × demo feature)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_basis_activation(combo, runs, out_path):
    """For basis/adaptive runs only: plot B_step[t, k] vs t for every basis k.

    Shows where each Gaussian bump has support along the cycle.
    """
    basis_runs = [r for r in runs if r[1] in ("basis", "adaptive")]
    if not basis_runs:
        return False
    fig, axes = plt.subplots(len(basis_runs), 1,
                             figsize=(8, 2.6 * len(basis_runs)),
                             squeeze=False)
    plotted = 0
    for ax, (n_w, mode, run_dir) in zip(axes[:, 0], basis_runs):
        bp = run_dir / "B_step.npy"
        if not bp.exists():
            ax.set_visible(False)
            continue
        B = np.load(bp)                       # (T+1, K)
        T1, K = B.shape
        cmap = plt.get_cmap("viridis", K)
        for k in range(K):
            ax.plot(B[:, k], lw=1.4, color=cmap(k), label=f"$B_{{{k+1}}}$")
        ax.set_xlabel("timestep")
        ax.set_ylabel(r"$B_k(t)$")
        ax.set_title(f"n_w={n_w}  {mode}  (K={K})  basis activation")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=min(K, 6), loc="upper right")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False
    fig.suptitle(f"{combo} — basis function activation along the cycle",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def plot_basis_influence(combo, runs, out_path):
    """For basis/adaptive runs only: plot recovered W(t) per feature.

    W(t) = B_step @ theta_run[k]. Shows how the basis parametrization
    actually shapes the per-step weight on each feature — i.e. which
    features the IRL emphasises at which phase of the cycle.
    """
    basis_runs = [r for r in runs if r[1] in ("basis", "adaptive")]
    if not basis_runs:
        return False

    fig, axes = plt.subplots(len(basis_runs), 1,
                             figsize=(9, 3.4 * len(basis_runs)),
                             squeeze=False)
    plotted = 0
    for ax, (n_w, mode, run_dir) in zip(axes[:, 0], basis_runs):
        bp  = run_dir / "B_step.npy"
        wsf = run_dir / "ws_history.npz"
        if not (bp.exists() and wsf.exists()):
            ax.set_visible(False)
            continue
        B = np.load(bp)                           # (T+1, K)
        wsd = np.load(wsf, allow_pickle=True)
        ws_run = wsd["ws_run"]
        keys_run = list(wsd["keys_run"])
        nr_run = len(keys_run)
        K = ws_run.shape[1] // nr_run
        if K * nr_run != ws_run.shape[1]:
            continue
        theta_run = ws_run[-1].reshape(K, nr_run)
        Wt = B @ theta_run                        # (T+1, nr_run)
        # Normalize each feature's W(t) to its max so they're visually
        # comparable on one axis.
        scale = np.maximum(np.abs(Wt).max(axis=0), 1e-12)
        Wt_n = Wt / scale[None, :]
        cmap = plt.get_cmap("tab10", nr_run)
        for j, key in enumerate(keys_run):
            ax.plot(Wt_n[:, j], lw=1.3, color=cmap(j), label=key)
        ax.set_xlabel("timestep")
        ax.set_ylabel("W(t) (per-feature, normalised)")
        ax.set_title(f"n_w={n_w}  {mode}  (K={K})  basis influence on W(t)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=min(nr_run, 4), loc="upper right")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return False
    fig.suptitle(f"{combo} — recovered per-feature W(t) from basis weights",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def _W_t_for_run(run_dir, mode):
    """Reconstruct the per-step weight matrix W(t) of shape (T+1, nr_run)
    for a single run. Returns (W, keys_run) or (None, None) if data missing.

    - basis / adaptive: W(t) = B_step[t] @ theta_run
    - windowed:         W(t) is the staircase from the per-window theta_run
    """
    wsf = run_dir / "ws_history.npz"
    if not wsf.exists():
        return None, None
    wsd = np.load(wsf, allow_pickle=True)
    ws_run = wsd["ws_run"]
    keys_run = list(wsd["keys_run"])
    if ws_run.shape[0] == 0:
        return None, None
    nr_run = len(keys_run)
    K = ws_run.shape[1] // nr_run
    if K * nr_run != ws_run.shape[1]:
        return None, None
    theta_run = ws_run[-1].reshape(K, nr_run)              # (K, nr_run)

    if mode in ("basis", "adaptive"):
        bp = run_dir / "B_step.npy"
        if not bp.exists():
            return None, None
        B_step = np.load(bp)                               # (T+1, K)
        return B_step @ theta_run, keys_run                # (T+1, nr_run)

    if mode == "windowed":
        # Staircase: per-window constants. Recover T from xs_demo / xs_irl.
        T1 = None
        for fname in ("xs_demo.npy", "xs_irl.npz"):
            p = run_dir / fname
            if not p.exists():
                continue
            arr = np.load(p)
            if fname.endswith(".npz"):
                arr = arr["xs"]
            T1 = len(arr)
            break
        if T1 is None:
            return None, None
        T = T1 - 1
        window_size = max(1, T // K)
        W = np.zeros((T1, nr_run))
        for ti in range(T1):
            k = min(ti // window_size, K - 1)
            W[ti] = theta_run[k]
        return W, keys_run

    return None, None


def plot_per_feature_W(combo, runs, out_path, mode_filter=None):
    """One panel per running feature; in each panel, overlay W_j(t) for
    every available n_w (1, 2, 3) of the chosen mode(s).

    If mode_filter is None, takes whatever modes are available. Otherwise
    a tuple/list of mode names ('windowed', 'basis', 'adaptive').
    """
    target_runs = ([r for r in runs if mode_filter is None or r[1] in mode_filter])
    if not target_runs:
        return False

    # Build (n_w, mode, W, keys_run) for each.
    per_run = []
    for n_w, mode, run_dir in target_runs:
        W, keys_run = _W_t_for_run(run_dir, mode)
        if W is None:
            continue
        per_run.append((n_w, mode, W, keys_run))
    if not per_run:
        return False

    keys_run = per_run[0][3]
    nr = len(keys_run)
    n_cols = 4 if nr > 6 else 3 if nr > 4 else 2
    n_rows = (nr + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.4 * n_rows),
                             sharex=True, squeeze=False)

    # Color by n_w; line style by mode.
    n_w_values = sorted({r[0] for r in per_run})
    cmap = plt.get_cmap("viridis", max(len(n_w_values), 2))
    colour_for = {nw: cmap(i) for i, nw in enumerate(n_w_values)}
    style_for  = {"windowed": "-", "basis": "--", "adaptive": ":"}

    for j, key in enumerate(keys_run):
        ax = axes[j // n_cols, j % n_cols]
        for n_w, mode, W, _ in per_run:
            x = np.linspace(0.0, 1.0, W.shape[0])
            ax.plot(x, W[:, j],
                    lw=1.6, color=colour_for[n_w],
                    ls=style_for.get(mode, "-"),
                    label=f"n_w={n_w}  {mode}")
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("cycle progress")
        ax.set_ylabel(rf"$W_{{{j+1}}}(t)$")

    # Hide unused panels
    for j in range(nr, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    # One shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set(); uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l)); seen.add(l)
    fig.legend([h for h, _ in uniq], [l for _, l in uniq],
               loc="upper center", ncol=min(len(uniq), 4), fontsize=9,
               bbox_to_anchor=(0.5, 1.0))

    title = (f"{combo} — per-feature W(t) across n_w"
             + (f"  (modes: {', '.join(mode_filter)})"
                if mode_filter else "  (all modes)"))
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_basis_decomposition(combo, runs, out_path, mode_pref="basis",
                             n_w_pref=3):
    """For ONE run, per feature, show the K weighted basis components
    θ_{k,j}·B_k(t) (dashed) + their sum W_j(t) (solid). Mirrors the
    top-right panel of basis_vs_windowed_explainer.png — makes it
    obvious how the basis weighted-by-theta gives the recovered W(t).

    Picks the (mode_pref, n_w_pref) run if available; falls back to any
    basis/adaptive run with the largest available n_w.
    """
    candidates = [r for r in runs if r[1] in ("basis", "adaptive")]
    if not candidates:
        return False
    pref = next(((nw, m, d) for nw, m, d in candidates
                 if m == mode_pref and nw == n_w_pref), None)
    if pref is None:
        candidates.sort(key=lambda r: (r[1] == mode_pref, r[0]), reverse=True)
        pref = candidates[0]
    n_w, mode, run_dir = pref

    bp  = run_dir / "B_step.npy"
    wsf = run_dir / "ws_history.npz"
    if not (bp.exists() and wsf.exists()):
        return False
    B   = np.load(bp)                                 # (T+1, K)
    wsd = np.load(wsf, allow_pickle=True)
    ws_run = wsd["ws_run"]
    keys_run = list(wsd["keys_run"])
    nr_run = len(keys_run)
    K = ws_run.shape[1] // nr_run
    if K * nr_run != ws_run.shape[1] or ws_run.shape[0] == 0:
        return False
    theta_run = ws_run[-1].reshape(K, nr_run)         # (K, nr_run)

    # Try to grab the matching windowed run at the same n_w so we can
    # overlay its staircase on each panel.
    W_windowed = None
    win_match = next(((nw, m, d) for nw, m, d in runs
                      if m == "windowed" and nw == n_w), None)
    if win_match is not None:
        W_windowed, kr_win = _W_t_for_run(win_match[2], "windowed")
        if W_windowed is not None and kr_win != keys_run:
            # Different feature ordering — skip overlay rather than mis-align
            W_windowed = None

    n_cols = 4 if nr_run > 6 else 3 if nr_run > 4 else 2
    n_rows = (nr_run + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.4 * n_rows),
                             sharex=True, squeeze=False)
    cmap = plt.get_cmap("viridis", K)
    t_norm = np.linspace(0.0, 1.0, B.shape[0])

    for j, key in enumerate(keys_run):
        ax = axes[j // n_cols, j % n_cols]
        for k in range(K):
            ax.plot(t_norm, B[:, k] * theta_run[k, j], "--",
                    lw=1.1, color=cmap(k), alpha=0.85,
                    label=(rf"$\theta_{{{k+1}}}\,B_{{{k+1}}}(t)$"
                           if j == 0 else None))
        W_j = B @ theta_run[:, j]
        ax.plot(t_norm, W_j, "k-", lw=2.0,
                label=(r"$W_j(t)\;=\;\sum_k\theta_{k,j}B_k(t)$"
                       if j == 0 else None))
        # Overlay windowed staircase at the same n_w (if available).
        if W_windowed is not None:
            t_win = np.linspace(0.0, 1.0, W_windowed.shape[0])
            ax.step(t_win, W_windowed[:, j], where="post",
                    color="tab:red", lw=1.6, alpha=0.85,
                    label=(rf"windowed (n_w={n_w}) step"
                           if j == 0 else None))
        ax.set_title(f"{key}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("cycle progress")
        ax.set_ylabel("weight")

    for j in range(nr_run, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    n_legend_cols = min(len(handles), K + 2)
    fig.legend(handles, labels, loc="upper center", ncol=n_legend_cols,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    overlay_note = (f"  +  windowed n_w={n_w} step overlay"
                    if W_windowed is not None else "")
    fig.suptitle(f"{combo} — basis × θ decomposition  "
                 f"(mode={mode}, n_w={n_w}, K={K}){overlay_note}",
                 y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_W_modes_compared(combo, runs, out_path, n_w_pref=3):
    """For one chosen n_w (default 3), overlay W_j(t) from every available
    mode (windowed + basis + adaptive) on the same per-feature panel.
    Direct apples-to-apples comparison of "what each parametrization
    actually recovered".
    """
    selected = []
    for mode in ("windowed", "basis", "adaptive"):
        match = [(nw, m, d) for nw, m, d in runs if m == mode and nw == n_w_pref]
        if match:
            selected.append(match[0])
    if not selected:
        return False

    by_mode = {}
    keys_run = None
    for nw, mode, run_dir in selected:
        W, kr = _W_t_for_run(run_dir, mode)
        if W is None:
            continue
        by_mode[mode] = W
        keys_run = kr
    if not by_mode or keys_run is None:
        return False

    nr_run = len(keys_run)
    n_cols = 4 if nr_run > 6 else 3 if nr_run > 4 else 2
    n_rows = (nr_run + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.4 * n_rows),
                             sharex=True, squeeze=False)

    style_for = {
        "windowed": ("-",  "tab:red"),
        "basis":    ("--", "tab:blue"),
        "adaptive": (":",  "tab:green"),
    }

    for j, key in enumerate(keys_run):
        ax = axes[j // n_cols, j % n_cols]
        for mode, W in by_mode.items():
            ls, color = style_for.get(mode, ("-", "k"))
            x = np.linspace(0.0, 1.0, W.shape[0])
            ax.plot(x, W[:, j], ls=ls, lw=1.7, color=color,
                    label=(mode if j == 0 else None))
        ax.set_title(key, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlabel("cycle progress")
        ax.set_ylabel(rf"$W_{{{j+1}}}(t)$")

    for j in range(nr_run, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=len(by_mode), fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{combo} — recovered W_j(t) across modes  "
                 f"(n_w={n_w_pref})", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combo", required=True, help="combo name (e.g. miras_dl)")
    ap.add_argument("--tag", default=None,
                    help="substring to filter run dirs by (e.g. 'felix_0506_1623' "
                         "to plot only one batch's 4 runs).")
    args = ap.parse_args()

    runs = _runs_for(args.combo, tag_substr=args.tag)
    if not runs:
        print(f"[err] no csqp_irl runs found under {ANALYSIS / args.combo}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"irl_{args.combo}"
    out = lambda suffix: base.with_name(base.name + f"__{suffix}.png")

    ok_c   = plot_convergence       (args.combo, runs, out("convergence"))
    ok_w   = plot_weights           (args.combo, runs, out("weights"))
    ok_f   = plot_force_target      (args.combo, runs, out("force_target"))
    ok_cc  = plot_cost_contributions(args.combo, runs, out("contributions"))
    ok_ba  = plot_basis_activation  (args.combo, runs, out("basis_activation"))
    ok_bi  = plot_basis_influence   (args.combo, runs, out("basis_influence"))
    # Per-feature W(t) — one figure per mode so the n_w sweep can be
    # compared cleanly without 9 lines per panel.
    ok_pf_w = plot_per_feature_W(args.combo, runs,
                                  out("W_per_feature_windowed"),
                                  mode_filter=("windowed",))
    ok_pf_b = plot_per_feature_W(args.combo, runs,
                                  out("W_per_feature_basis"),
                                  mode_filter=("basis",))
    ok_pf_a = plot_per_feature_W(args.combo, runs,
                                  out("W_per_feature_adaptive"),
                                  mode_filter=("adaptive",))
    ok_dec  = plot_basis_decomposition(args.combo, runs,
                                        out("basis_decomposition"))
    ok_cmp  = plot_W_modes_compared(args.combo, runs,
                                     out("W_modes_compared"))

    print(f"[plot] convergence:   {'ok' if ok_c  else 'skipped (no q_norm.npy)'}")
    print(f"[plot] weights:       {'ok' if ok_w  else 'skipped (no ws_history.npz)'}")
    print(f"[plot] force:         {'ok' if ok_f  else 'skipped (no force_target.npy)'}")
    print(f"[plot] contributions: {'ok' if ok_cc else 'skipped (no phi_demo.npy)'}")
    print(f"[plot] basis active:  {'ok' if ok_ba else 'skipped (no B_step.npy / no basis runs)'}")
    print(f"[plot] basis influ:   {'ok' if ok_bi else 'skipped (no B_step.npy / no basis runs)'}")
    print(f"[plot] W per feat windowed: {'ok' if ok_pf_w else 'skipped'}")
    print(f"[plot] W per feat basis:    {'ok' if ok_pf_b else 'skipped'}")
    print(f"[plot] W per feat adaptive: {'ok' if ok_pf_a else 'skipped'}")
    print(f"[plot] basis decomposition: {'ok' if ok_dec else 'skipped (no basis/adaptive run)'}")
    print(f"[plot] modes compared:      {'ok' if ok_cmp else 'skipped (need ≥1 mode at n_w_pref)'}")
    print(f"[plot] output dir:    {OUT_DIR}")


if __name__ == "__main__":
    main()
