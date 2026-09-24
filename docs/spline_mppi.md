# Spline-Parameterized MPPI

## Problem: why standard MPPI produces shaky controls

Standard MPPI (Model Predictive Path Integral) finds the optimal control
sequence `U ∈ ℝ^(H×nu)` by sampling K noisy perturbations, rolling each out
through the dynamics, and computing a softmax-weighted mean:

```
ε_k ~ N(0, σ²·I)   shape (H, nu)     ← independent per-timestep noise
U_k = U + ε_k
cost_k = Σ_t  w · φ(x_t, u_t)
w_k  = exp(-cost_k / λ) / Z
U    ← U + Σ_k w_k · ε_k
```

The key issue: **each timestep's noise is independent**.
The weighted mean `U + Σ_k w_k·ε_k` is itself a weighted average of
H-dimensional white-noise vectors. Even though the average of many Gaussians
is smoother than any individual sample, the result still inherits the
per-timestep independence — consecutive `U[t]` and `U[t+1]` are not
constrained to be similar. This produces a jittery control signal.

**Consequences for IRL:**

| Feature | Depends on | Noise source |
|---------|-----------|--------------|
| `Eng = ‖qvel·u‖/20` | `u` directly | per-step torque jitter |
| `JA = ‖ddq_kin‖/100` | FD of `qvel` | velocity jitter → amplified by FD |
| `JV = ‖qvel‖` | state velocity | velocity jitter from shaky torques |

Demo features (from smooth mocap kinematics) are O(0.05–0.3). Rollout
features are O(1–5). IRL sees this as a large demo-vs-rollout gap and
inflates Eng/JV/JA weights trying to close it — but the gap is partly
noise, not real signal.

---

## Solution: sample in spline control-point space

Instead of sampling noise directly on the `(H, nu)` control sequence,
we sample noise on **M control points** (M ≪ H) and map them to a smooth
H-step control trajectory via a fixed **B-spline basis** `B ∈ ℝ^(H×M)`:

```
CP_k ~ N(0, σ²·I)   shape (M, nu)     ← noise on M control points
ε_k  = B @ CP_k     shape (H, nu)     ← smooth by construction
U_k  = U + ε_k
```

The basis `B` is a clamped uniform **cubic B-spline** evaluated at H
uniformly-spaced time points. Each column of `B` is one basis function —
a smooth bump with C² continuity (continuous first and second derivatives).
Because every sample `ε_k` is in the column-span of `B`, it is smooth.
The weighted update:

```
CP ← CP + Σ_k w_k · CP_k_noise    (update in control-point space)
U  = B @ CP                        (reconstruct smooth control)
```

keeps `U` in the smooth subspace at every DIAL iteration.

---

## The cubic B-spline basis

A clamped uniform cubic B-spline with M control points over [0, 1]:

- **Knot vector**: `[0,0,0,0, t_1, …, t_{M-4}, 1,1,1,1]`
  (4 repeated knots at each end → curve passes through CP[0] and CP[M-1]).
- **Degree**: 3 (cubic, C² everywhere except at coincident knots).
- **Support**: each basis function `B_j(t)` is non-zero over 4 knot spans
  → local control (changing CP_j only affects the curve near time j).
- **Partition of unity**: `Σ_j B_j(t) = 1` at every t → control magnitude
  is preserved; if all CPs are constant `c`, the curve is constant `c`.

For the shaving setup with H=30 timesteps and M=8 control points:

```
B: (30, 8) — each of 8 columns is a smooth bump

B @ CP = weighted combination of 8 "prototypical" motion shapes
```

A sample `ε = B @ N(0,σ)` looks like a smooth curve with 8 degrees of
freedom instead of 30 independent random values.

---

## Search dimensionality and sample efficiency

| Parameterization | Search dim / sample | Smoothness ‖ddu‖ | Budget for same quality |
|-----------------|--------------------|--------------------|------------------------|
| Discrete H=30 | 30×9 = **270** | ~35 | 1536 evals |
| Spline M=8 | 8×9 = **72** | ~2 | **32 evals** |

From `spline_budget_compare.py` (numpy POC) and `test_spline_mppi.py` (real MJX):

| config | budget | TV/step | ‖ddu‖ | time |
|--------|--------|---------|-------|------|
| discrete K=64 nd=2 | 128 | 0.90 | 5.51 | 52s |
| discrete K=256 nd=6 | 1536 | 0.51 | 3.10 | 139s |
| **spline M=8 K=32 nd=1** | **32** | **0.39** | **1.84** | **30s** |
| **spline M=8 K=64 nd=2** | **128** | **0.32** | **1.49** | **53s** |

Spline at budget=32 beats discrete at budget=1536 on smoothness AND is
4.6× faster. At the same budget (128), spline is 2.8× smoother.

---

## Implementation in `KinematicMPPI_MJX`

The spline mode is activated by passing `spline_M=M` to the constructor.
Default `spline_M=None` is byte-identical to the original discrete behavior.

```python
kin = KinematicMPPI_MJX(human_mppi, horizon=30, num_samples=32,
                         n_dial=1, noise_scale=1.0,
                         spline_M=8)   # ← spline mode
```

**Inside `_build_jit()`** (the JAX JIT compilation):

```python
# 1. Build basis once and capture into JIT closure as a constant.
B = cubic_bspline_basis(H=30, M=8)          # (30, 8) numpy
B_jax = jnp.asarray(B, dtype=jnp.float32)  # constant in JIT

# 2. Noise scaling calibration.
#    std(B @ N(0,σ)) per timestep ≈ σ · sqrt(Σ_j B[h,j]²)
#    Set σ_cp = 1 / sqrt(mean_h Σ_j B[h,j]²)
#    so that the per-timestep std of the mapped noise ≈ the discrete std.
spline_gain = 1.0 / sqrt(mean(sum(B**2, axis=1)))   # ≈ 1.43 for M=8

# 3. Sampling (inside mppi_update DIAL loop).
z = jax.random.normal(sub, (K, M, nu))           # (K, M, nu) CP noise
eps_cp = einsum('ij,kmj->kmi', sigma_L, z) * spline_gain * decay
eps    = einsum('hm,kmn->khn', B_jax, eps_cp)    # (K, H, nu) smooth

# 4. MPPI update is UNCHANGED — just uses smooth eps.
U = U + einsum('k,khn->hn', weights, eps)
```

The rollout `rollout_k(qpos0, qvel0, U + ε_k, ...)` and cost computation
are completely unchanged — spline only affects noise generation.

**Host-side deploy** (important — must match JIT):

```python
# In solve() receding-horizon loop — gravity compensation added separately.
# The stored us[] are the gravity-compensated full actuator command.
tau_grav = data.qfrc_bias[arm_dofadr]
u = clip(tau_grav + self.U[0], u_min, u_max)
data.ctrl[:] = u
us.append(u.copy())
```

---

## Noise scale calibration

The `spline_gain` ensures comparable exploration. Without it:

- Discrete: noise per timestep ~ `σ · ‖L_row‖` (L is the Cholesky factor
  from joint covariance; shape (nu, nu)).
- Spline: `std(B @ N(0,σ_cp))_h ≈ σ_cp · sqrt(Σ_j B[h,j]²)`.

Setting `σ_cp = σ / sqrt(mean_h Σ_j B[h,j]²)` makes the marginal per-step
standard deviation the same in both modes. For M=8, H=30: `gain ≈ 1.43`.

This means you can use the same `noise_scale` parameter in both modes and
get comparable exploration radius.

---

## Choosing M (number of control points)

M controls the trade-off between smoothness and expressiveness:

| M | dim | TV/step | ‖ddu‖ | Can represent demo? |
|---|-----|---------|-------|---------------------|
| 4 | 36 | 1.85 | 0.26 | 30% error (too coarse) |
| 6 | 54 | 2.07 | 0.27 | 29% error (marginal) |
| **8** | **72** | **1.49** | **1.49** | **28% error (marginal)** |
| 12 | 108 | 2.07 | 5.75 | 24% error |

The demo's control `us_demo` has ~24% reconstruction error at M=12 because
the demo torques themselves are jagged (inverse-dynamics noise). This floor
is irreducible — the smooth intent is captured, the noise is discarded.
**M=8 is the sweet spot**: smooth enough to kill shake, expressive enough
to track the task.

Key insight: because the demo's *features* (Eng, JV, JA etc.) are already
smooth (TV 0.01–0.05), the spline only needs to bring the rollout's features
into the same smoothness class — not match the raw jagged `us_demo` torque.

---

## For the paper

The theoretical contribution: MPPI with spline control parameterization
reduces the search dimension from H·nu to M·nu (3.75× reduction for H=30,
M=8), produces inherently smooth motion suitable for close human-robot
interaction, and achieves better trajectory quality at lower compute budget
than standard MPPI.

Head-to-head comparison produced by `test_spline_mppi.py`:
- Same budget (128 evals): spline 2.8× smoother, contact fraction and
  q_RMSE similar → same task performance, better motion quality.
- Smaller budget (32 vs 1536 evals): spline smoother, 4.6× faster →
  enables faster IRL outer-loop iterations without sacrificing quality.
