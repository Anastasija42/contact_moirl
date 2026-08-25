"""basis_vs_windowed_demo.py — minimal visual demo for why basis weights
pool more data per parameter than windowed weights.

Renders one figure with 2 rows × 2 columns:

  top row    — same target W(t), approximated two ways:
               (left) windowed — piecewise-constant staircase
               (right) basis   — smooth Gaussian mixture

  bottom row — for each parameter θ_k, the kernel B_k(t) that decides
               which timesteps contribute to that θ_k's gradient:
               (left) windowed — rectangular indicator (only its chunk)
               (right) basis   — Gaussian (every timestep, weighted)

Both parametrizations use K=3 parameters per feature, so the comparison
is parameter-count-fair. The point: windowed gives each θ_k ~T/K
timesteps of data; basis gives each θ_k ALL T timesteps (weighted by
how much its bump overlaps each t).

Run:
    python docs/scripts/basis_vs_windowed_demo.py
    # → docs/assets/figures/basis_vs_windowed_explainer.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Setup ──────────────────────────────────────────────────────────────
T = 66          # cycle length (matches the actual experiments)
K = 3           # number of parameters per feature
t = np.arange(T + 1)

# Pretend "true" W(t): a smooth bump in the middle (think contact phase
# weighted higher than approach/release).
W_true = 0.5 + 1.5 * np.exp(-((t - 33.0) / 12.0) ** 2)

# ── Windowed parametrization ───────────────────────────────────────────
window_size = (T + 1) // K
edges = [k * window_size for k in range(K)] + [T + 1]
B_window = np.zeros((T + 1, K))
for k in range(K):
    B_window[edges[k]:edges[k + 1], k] = 1.0
theta_windowed = np.array([
    W_true[edges[k]:edges[k + 1]].mean() for k in range(K)
])
W_windowed = B_window @ theta_windowed

# ── Basis parametrization (Gaussian, partition-of-unity rows) ──────────
mu_centers = np.linspace(0, T, K)
sigma = T / (K - 1)
B_unnorm = np.exp(-((t[:, None] - mu_centers[None, :]) ** 2) / (2 * sigma ** 2))
B_basis  = B_unnorm / B_unnorm.sum(axis=1, keepdims=True)
# Best-fit θ via least squares
theta_basis, *_ = np.linalg.lstsq(B_basis, W_true, rcond=None)
W_basis = B_basis @ theta_basis

# ── Figure ─────────────────────────────────────────────────────────────
cmap = plt.get_cmap("viridis", K)
fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))

# Top-left: windowed W(t)
ax = axes[0, 0]
ax.plot(t, W_true,     "k-",     lw=2, alpha=0.35, label="true W(t)")
ax.plot(t, W_windowed, "tab:red", lw=2.2, label="windowed approx")
for x in edges[1:-1]:
    ax.axvline(x, color="gray", ls=":", alpha=0.5)
for k in range(K):
    cx = 0.5 * (edges[k] + edges[k + 1])
    ax.text(cx, theta_windowed[k] + 0.08,
            rf"$\theta_{k}={theta_windowed[k]:.2f}$",
            ha="center", fontsize=9, color="tab:red")
ax.set_title("Windowed: W(t) is piecewise-constant")
ax.set_xlabel("timestep t"); ax.set_ylabel("W(t)")
ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)

# Top-right: basis W(t) plus per-basis contributions
ax = axes[0, 1]
ax.plot(t, W_true, "k-",      lw=2, alpha=0.35, label="true W(t)")
ax.plot(t, W_basis, "tab:blue", lw=2.2, label="basis approx")
for k in range(K):
    ax.plot(t, B_basis[:, k] * theta_basis[k], "--", lw=1.2,
            color=cmap(k), alpha=0.85,
            label=rf"$B_{k}(t)\cdot\theta_{k}$  ($\theta_{k}={theta_basis[k]:.2f}$)")
ax.set_title("Basis: W(t) is a smooth Gaussian mixture")
ax.set_xlabel("timestep t"); ax.set_ylabel("W(t)")
ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)

# Bottom-left: windowed gradient kernels
ax = axes[1, 0]
for k in range(K):
    ax.plot(t, B_window[:, k], color=cmap(k), lw=2, label=rf"$B_{k}(t)$")
    ax.fill_between(t, 0, B_window[:, k], color=cmap(k), alpha=0.18)
ax.set_title(rf"Windowed gradient kernel: each $\theta_k$ sees "
             rf"only ~{(T + 1) // K} timesteps")
ax.set_xlabel("timestep t"); ax.set_ylabel(r"$B_k(t)$")
ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# Bottom-right: basis gradient kernels
ax = axes[1, 1]
for k in range(K):
    ax.plot(t, B_basis[:, k], color=cmap(k), lw=2, label=rf"$B_{k}(t)$")
    ax.fill_between(t, 0, B_basis[:, k], color=cmap(k), alpha=0.18)
ax.set_title(rf"Basis gradient kernel: every $\theta_k$ sees all "
             rf"{T + 1} timesteps (weighted)")
ax.set_xlabel("timestep t"); ax.set_ylabel(r"$B_k(t)$")
ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

fig.suptitle(f"Windowed vs Gaussian-basis parametrization of W(t)  "
             f"(both K={K}; T+1={T + 1} timesteps)",
             fontsize=13)
fig.tight_layout()

OUT = Path(__file__).resolve().parents[1] / "assets/figures/basis_vs_windowed_explainer.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
plt.close(fig)
print(f"saved {OUT}")
print("\nGradient-pooling summary at K=3, T+1=67:")
print(f"  windowed: each θ_k receives data from ~{(T + 1) // K} timesteps")
print(f"  basis   : each θ_k receives data from all {T + 1} timesteps "
      f"(weights sum to ~{B_basis.sum(axis=0).mean():.1f})")
