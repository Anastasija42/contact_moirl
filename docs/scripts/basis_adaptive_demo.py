"""basis_adaptive_demo.py — visual demo of why adaptive basis growth
beats fixed K when K is too small, and matches fixed K with fewer
parameters when K is generous.

Three approximations of the same target W(t) (a narrow peak in the
middle plus a flat baseline — think 'short contact phase, smooth
elsewhere'):

  - K=2 fixed Gaussian basis (uniform centers)           → underfits
  - K=8 fixed Gaussian basis (uniform centers)           → fits, 8 params
  - Adaptive (start K=2, insert new center at argmax     → fits, ~K=4
    of residual until tolerance reached)

Figure layout (2 rows × 3 cols):
  Row 1 — approximation: true W(t), the fit, residual annotated.
  Row 2 — basis placement: B_k(t) curves; for adaptive, the iteration
          number of each insertion is labeled near the center.

Run:
    python docs/scripts/basis_adaptive_demo.py
    # → docs/assets/figures/basis_adaptive_explainer.png
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Setup ──────────────────────────────────────────────────────────────
T = 66
t = np.arange(T + 1)

# Target: two bumps at different locations on a flat baseline. The two
# bumps are at t=18 (taller) and t=48 (shorter), mocking "early grasp"
# and "release-phase" emphasis. Bump width chosen broad enough that the
# basis at SIGMA can actually represent each one with ~1 center —
# otherwise adaptive would keep stacking centers at the same peak.
W_true = (0.2
          + 1.2 * np.exp(-((t - 18.0) / 9.0) ** 2)
          + 0.8 * np.exp(-((t - 48.0) / 9.0) ** 2))

SIGMA = 8.0   # fixed local scale for all basis functions in this demo


def fit_basis(centers, sigma, target):
    """Build Gaussian basis at `centers` (partition-of-unity rows),
    fit theta by LSQ to `target`, return (B, theta, fit, max_residual)."""
    centers = np.asarray(centers, dtype=float)
    B_unnorm = np.exp(-((t[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
    B = B_unnorm / B_unnorm.sum(axis=1, keepdims=True)
    theta, *_ = np.linalg.lstsq(B, target, rcond=None)
    fit = B @ theta
    max_res = float(np.max(np.abs(target - fit)))
    return B, theta, fit, max_res


def adaptive_fit(target, sigma=SIGMA, residual_tol=0.05, max_K=8,
                 K_start=2, dedup_radius=4.0):
    """Mirrors MO_IRL.solve_with_refinement: start with `K_start`
    uniform centers, repeatedly insert a new center at the timestep of
    largest residual, until residual < tol or K hits max_K.

    `dedup_radius`: if argmax-residual lands within this many timesteps
    of an existing center, the residual there is masked and we look
    elsewhere. Without this guard, an unrepresentable peak (basis σ
    wider than the target's local scale) makes argmax stick at the
    same spot and centers stack uselessly.
    """
    centers = list(np.linspace(0.0, T, K_start))
    history = []
    for k_iter in range(max_K - K_start + 1):
        B, theta, fit, max_res = fit_basis(centers, sigma, target)
        history.append((k_iter, list(centers), max_res, fit))
        if max_res < residual_tol:
            break
        residual = np.abs(target - fit)
        # Mask timesteps near existing centers so we don't stack.
        for c in centers:
            mask = np.abs(t - c) < dedup_radius
            residual[mask] = 0.0
        if residual.max() < residual_tol:
            break
        t_star = int(np.argmax(residual))
        centers.append(float(t[t_star]))
    return history


# ── Compute the three approximations ───────────────────────────────────
# 1. K=2 fixed
centers_K2 = list(np.linspace(0.0, T, 2))
B_K2, theta_K2, fit_K2, max_res_K2 = fit_basis(centers_K2, SIGMA, W_true)

# 2. K=8 fixed
centers_K8 = list(np.linspace(0.0, T, 8))
B_K8, theta_K8, fit_K8, max_res_K8 = fit_basis(centers_K8, SIGMA, W_true)

# 3. Adaptive
hist = adaptive_fit(W_true, sigma=SIGMA, residual_tol=0.15, max_K=8, K_start=2)
final_iter, centers_ad, max_res_ad, fit_ad = hist[-1]
B_ad, theta_ad, _, _ = fit_basis(centers_ad, SIGMA, W_true)
K_ad = len(centers_ad)
# Insertion order: the first K_start centers all came from the uniform
# start (label "1"), then each subsequent center got the next refinement
# iteration number (2, 3, 4, ...).
K_start = 2
n_inserted = K_ad - K_start
insertion_order = [1] * K_start + list(range(2, 2 + n_inserted))

# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))

panel_specs = [
    (axes[0, 0], axes[1, 0], "K = 2 fixed (uniform)",
     centers_K2, B_K2, fit_K2, max_res_K2, None),
    (axes[0, 1], axes[1, 1], "K = 8 fixed (uniform)",
     centers_K8, B_K8, fit_K8, max_res_K8, None),
    (axes[0, 2], axes[1, 2], f"Adaptive (start K=2 → K_final={K_ad})",
     centers_ad, B_ad, fit_ad, max_res_ad, insertion_order),
]

for ax_top, ax_bot, title, centers, B, fit, max_res, order in panel_specs:
    K = len(centers)
    cmap = plt.get_cmap("viridis", max(K, 2))

    ax_top.plot(t, W_true, "k-", lw=2, alpha=0.4, label="true W(t)")
    ax_top.plot(t, fit,    "tab:blue", lw=2.2, label="basis fit")
    ax_top.fill_between(t, W_true, fit, color="tab:red", alpha=0.18,
                        label="residual")
    ax_top.set_title(f"{title}\nmax |residual| = {max_res:.3f}", fontsize=11)
    ax_top.set_xlabel("timestep t"); ax_top.set_ylabel("W(t)")
    ax_top.set_ylim(-0.1, 2.1)
    ax_top.legend(loc="upper right", fontsize=8); ax_top.grid(alpha=0.3)

    for k, c in enumerate(centers):
        ax_bot.plot(t, B[:, k], color=cmap(k), lw=2)
        ax_bot.fill_between(t, 0, B[:, k], color=cmap(k), alpha=0.18)
        if order is not None:
            ax_bot.annotate(f"{order[k]}", xy=(c, 1.02),
                            xytext=(c, 1.18), ha="center",
                            fontsize=10, color=cmap(k), fontweight="bold",
                            arrowprops=dict(arrowstyle="-", color=cmap(k),
                                            alpha=0.5))
    ax_bot.set_title(rf"$B_k(t)$  (K = {K} centers, $\sigma$ = {SIGMA:.0f})",
                     fontsize=11)
    ax_bot.set_xlabel("timestep t"); ax_bot.set_ylabel(r"$B_k(t)$")
    ax_bot.set_ylim(-0.05, 1.4); ax_bot.grid(alpha=0.3)
    if order is not None:
        ax_bot.text(0.02, 0.92, "labels = insertion order\n"
                                "(1 = uniform start, ≥2 = refinement)",
                    transform=ax_bot.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="gray", alpha=0.8))

fig.suptitle("Underfit vs over-allocation vs adaptive growth\n"
             "(target = narrow peak at t=20 plus flat baseline)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

OUT = Path(__file__).resolve().parents[1] / "assets/figures/basis_adaptive_explainer.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
plt.close(fig)
print(f"saved {OUT}")

print("\nAdaptive refinement trace:")
for it, centers, max_res, _ in hist:
    print(f"  iter {it}: K={len(centers)}, "
          f"centers={[f'{c:.0f}' for c in centers]}, "
          f"max|residual|={max_res:.3f}")

print(f"\nFinal:")
print(f"  K=2  fixed   : max |residual| = {max_res_K2:.3f}  "
      f"(8 → params: 2)")
print(f"  K=8  fixed   : max |residual| = {max_res_K8:.3f}  "
      f"(    params: 8)")
print(f"  Adaptive K={K_ad:<2} : max |residual| = {max_res_ad:.3f}  "
      f"(    params: {K_ad})")
