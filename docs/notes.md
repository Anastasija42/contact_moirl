---
layout: default
title: Technical Notes
---

# Technical Notes

*Internal notes on what we learned building MPPI-IRL for contact-rich manipulation.*

---

## 1. MJX is Nondeterministic with Contact

MuJoCo's GPU backend (MJX via JAX) produces different trajectories from the same inputs when contact is involved. The nondeterminism comes from floating-point reduction order in `vmap`'d rollouts — not from the RNG, not from the contact solver algorithm.

**How we found it:**
- Same weights, same `PRNGKey(42)` → different `U` from `update()`
- Max diff in controls: 19.7 between two identical calls
- Without contact (no stick), MJX is perfectly deterministic

**Attempted fixes that failed:**
- `XLA_FLAGS=--xla_gpu_deterministic_ops=true` → 2x slower, still nondeterministic
- Newton solver with 100 iterations → 24s/step, still nondeterministic
- Implicit integrator → MJX doesn't support it (`NotImplementedError`)
- Disable warmstart → NaN in controls

**Solution:** CPU-only MPPI (`mppi_cpu.py`). MuJoCo's C API is perfectly deterministic. Max diff = 0.0 across runs. ~3-5x slower than GPU but required for IRL.

**Important detail:** Each sample rollout must create a **fresh `MjData`** — reusing from a pool causes nondeterminism due to residual internal state.

---

## 2. Feature Normalization Matters More Than You Think

Original features used `0.5 * ||x||^2` convention (matching Crocoddyl). This creates 9 orders of magnitude difference:

```
JA:       ~10^6  (squared acceleration)
approach: ~0.005 (squared distance)
```

No weight tuning can bridge this. Weight 100,000 on approach gives the same cost contribution as weight 0.001 on JA.

**Fix:** Linear norms normalized by reference values:

| Old | New |
|-----|-----|
| `0.5 * sum(u^2)` | `||u|| / 50` |
| `0.5 * sum(a^2)` | `||a|| / 100` |
| `||d||^2` | `||d|| / stick_radius` |

Now all features are O(1) and weight 1.0 means "this matters as much as anything else."

**Both the MPPI cost AND the IRL feature extraction must use the same normalization.** We had a bug where the MPPI used normalized features but the IRL feature computation (`_compute_run_feature`) used the old squared ones → gradient direction was wrong.

---

## 3. The Temperature Controls Everything

The IRL gradient is:

```
dw = P_nopt * temperature * (phi_nopt - phi_demo)
```

- `P_nopt = exp(-temperature * Cost_nopt) / Z`
- If `temperature * Cost` is too large → `P_nopt = 0` → gradient = 0
- If `temperature * Cost` is too small → `P_nopt ≈ 0.5` → gradient tiny (temperature factor)

**Symptom of wrong temperature:** `Fcn Val` constant across iterations, gradients O(10^-100).

**Calibration:** `temperature * |Cost_demo - Cost_current| ≈ 1-5`

With weights O(1) and O(1) features, costs are O(10-50), so `temperature ≈ 0.05-0.5`. We spent days debugging before realizing the temperature was 3.5e-7.

---

## 4. L-BFGS-B vs Direct Gradient: What We Actually Know

### The bug that masqueraded as a fundamental problem

For a long stretch we believed L-BFGS-B simply could not work with MPPI: every run returned `dw = 0` on iteration 1, and we built up increasingly elaborate theories (softmax saturation, piecewise-constant elite boundaries, bound-corner degeneracy) to explain it.

The actual cause was much less interesting: `Optimization_utils.fg_w` and `f_w` had branches for `var == 'dw'` and `var == 'w'`, but our IRL configs pass `'opt_vars': 'all'`. There was no `'all'` branch, so both functions silently fell through and returned `(f=0.0, g=zeros)` on every scipy call. L-BFGS-B saw a perfectly zero gradient, correctly declared `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`, and returned `nit=0, dw=0` — exactly as it should have given the inputs it received.

After adding an `'all'` branch (and an `else: raise` so future dispatch bugs die loudly), L-BFGS-B runs normally: on the single-feature test it converged in 9 inner iterations with `f` dropping from 125 → 17 and `dw[8] ≈ -0.97`, exactly as the MaxEnt inner problem's analytical solution says.

### So which do we actually use?

> **Updated.** We now ship **plain SGD** for the MPPI path (`use_adam=False`,
> the default everywhere in `MO_IRL.py`), and **L-BFGS-B** for the deterministic
> OCP path (`use_mppi_grad=False`). Adam was the earlier default but was dropped:
> its per-coordinate normalisation over-normalises the masked-out features and
> saturates the MPPI softmax at every line-search probe. See
> [Method — gradient step rule](method#gradient-step-rule-which-step-rule-for-which-solver)
> for the current story. The discussion below is retained for the historical
> reasoning that led to the comparison.

The mapping is: the **deterministic OCP** trajectory is smooth in $w$, so a
quasi-Newton method (L-BFGS-B, via `use_mppi_grad=False`) can trust its curvature
estimates; the **MPPI** rollouts are stochastic, so the noisy gradient corrupts
curvature and we fall back to a plain masked SGD step. The historical Adam-vs-LBFGS
notes:

- Adam's fixed-magnitude per-coordinate step gives more predictable behaviour when MPPI's elite sample flips between iterations (gradient spikes get damped by $\hat v_k$, and momentum carries the weights through flat stretches) — but in practice that same per-coordinate rescaling fights the feature mask, which is why it was dropped.
- L-BFGS-B commits a larger step per iteration because it converges on the current buffer. Whether that step is useful depends on how the next MPPI rollout responds — sometimes the big converged step shifts the sampler meaningfully, sometimes it lands inside a plateau.

### What the real constraint is

The limitation we originally thought was "L-BFGS-B can't handle MPPI" is actually:

**The map $w \mapsto \phi_{\text{MPPI elite}}(w)$ is piecewise-constant.** For a range of weight vectors, MPPI's $\arg\min$ over its sample population picks the same elite, then at some threshold switches. This affects the *outer* IRL loop regardless of which inner optimiser we use: the buffer grows slowly with near-duplicate entries until a weight update crosses a plateau boundary.

The right mitigations for this — top-$K$ elite buffer, or re-weighting by MPPI's own softmax over all samples — are on the list for future work. Adam vs L-BFGS-B is orthogonal to this problem; both inner optimisers would benefit from a richer buffer.

### The gradient-poisoning claim is separate

An unrelated, still-valid observation: when combining features with opposing gradient signs, the shared $P_{\text{nopt}}$ factor shrinks the effective gradient on every feature (see section 12). This is a property of the MaxEnt softmax formulation, not of the inner optimiser, and is addressed by Phase 1 / Phase 2 separation.

---

## 5. The Masked opt_div Was the Key Fix

The `opt_div` (optimality divergence) measures `||phi_irl - phi_demo||` and is used by the line search to accept/reject weight updates.

**The problem:** With all 14 features in the norm, unweighted features (JA with diff=300, JV with diff=174) dominated, even when their weights were 0. A trajectory that improved approach and traveled would be REJECTED because JA got worse.

**The fix:** Only compare features with nonzero weight:

```python
mask = w_curr > 1e-10
opt_div = norm(p1[mask] - p2[mask]) / (T + 1)
```

Before this fix: the IRL could never accept a trajectory that made progress, because making progress = more motion = more JA/JV = worse opt_div.

After: the line search only checks the features we're actually learning. Immediate improvement.

---

## 6. Features That Work Alone vs Features That Need Context

We tested each feature individually (1 IRL iteration, single feature weight):

**Work alone (Phase 1):**
- `approach` — drives arm to stick, establishes contact
- `traveled` — drives forward motion (but flies through air without approach)

**Don't work alone:**
- `JA, JV, Tau, Eng, Geo` — pure penalties. Optimal = don't move. Weight change has no effect because "stay still" minimizes acceleration/velocity/torque regardless of weight value.
- `rock_ori, rail_lat` — negative gradient! A stationary arm has LESS rotation and lateral deviation than the demo (which is scraping). The gradient says "decrease weight" → weight hits 0 and stays there.
- `surface` — redundant with approach. Same gradient direction, weaker signal.
- `press_force` — flat when not in contact. All MPPI samples have force=0 → identical cost → no gradient.

**Key insight:** Style features only work after task features establish the motion. In Phase 2, the trajectory is already moving and has MORE jerk/rotation than the demo → gradients flip positive → weights grow.

---

## 7. The `progress` Feature Traps

`progress = max(0, 1 - traveled_dist / rail_len)` clips at 0. Once the arm passes the goal, progress = 0 regardless of how far past. This means:

- Overshooting and being at the goal have the same cost
- The gradient says "the IRL trajectory has LESS progress cost than the demo" → increase weight → arm goes faster → overshoots MORE

We replaced it with `traveled = -(traveled_dist / rail_len)` which is unbounded — overshooting increases the cost, providing a natural stopping signal.

---

## 8. The Sign Bug That Wasted a Day

In the simple test script (`test_mppi_irl_simple.py`), the gradient was:

```python
return -grad  # WRONG: dw decreases when nopt has MORE of a feature
```

Should have been:

```python
return grad   # RIGHT: dw increases when nopt has MORE → more penalty
```

The MO_IRL code had the correct sign (`dw = -g` where `g` already includes a negative from `get_dldw`). But the test script had it backwards, causing velocity/effort weights to decrease when they should have increased. The trajectory got worse every iteration.

**Lesson:** Always trace the sign through the full chain: `get_dldw` → `get_l_g_w` → `log_likelihood_f` → `_compute_dw` → `w_new = w + dw`. One extra negative anywhere inverts the learning.

---

## 9. CPU MPPI: Fresh MjData Per Sample

When parallelizing CPU MPPI with threading, we pre-allocated a pool of `MjData` objects (one per sample). This caused nondeterminism even on CPU — residual internal state from previous rollouts affected the next.

**Fix:** Create a fresh `MjData(model)` for each sample rollout. Slower (allocation overhead) but guarantees determinism.

Threading didn't help performance anyway — each sample rollout (~3ms for 40 `mj_step` calls) is too fast for thread dispatch overhead to amortize.

---

## 10. The Contact Barrier

MPPI cannot cross the no-contact → contact discontinuity through weight tuning alone. The `press_force` and `surface` features are **constant when not in contact** — they provide no gradient signal to guide the arm toward the stick.

Only `approach` (distance to stick axis) provides a smooth gradient from any distance. It must be the first feature learned, with high enough weight to overcome the MPPI dead zone.

This is fundamentally different from Crocoddyl, where the OCP solver can discover contact through its own gradient-based optimization. With MPPI, contact discovery requires explicit guidance.

---

## 11. Two-Phase Learning is Not a Hack

It might seem like fixing approach/traveled first and then learning style is "cheating" — the IRL should discover everything. But:

1. **The simple test proves it:** With a known ground truth, learning all weights from scratch works but takes 3x more iterations than fixed-progress + learn-penalties. The ratios converge to the same values either way.

2. **Phase separation is physically meaningful:** "What to do" (approach stick, move forward) vs "how to do it" (smoothly, with the right force) are genuinely different concerns. Humans also learn tasks this way — first the gross motion, then the fine control.

3. **The gradient naturally prioritizes phases:** Even without explicit separation, approach/traveled get the biggest gradients (biggest feature gaps) and converge first. Phase separation just makes this explicit and faster.

---

## 12. Gradient Poisoning: Why Features Can't Be Mixed Freely

The IRL gradient for each feature is:

```
dw_i = P_nopt * temperature * (phi_irl_i - phi_demo_i)
```

But `P_nopt` is shared across all features — it's computed from the **total cost** over all weighted features. If feature A has `phi_irl > phi_demo` (positive gradient) but feature B has `phi_irl < phi_demo` (negative gradient), they fight inside the softmax:

- Feature A (e.g., approach): IRL is further from stick → wants weight UP
- Feature B (e.g., rock_ori): IRL rotates less (stationary!) → wants weight DOWN

When both are weighted, the demo's high rock_ori cost makes the demo look **worse** overall, reducing `P_nopt` for ALL features — including approach. The gradient for approach shrinks, even though approach should be growing.

**Tested:** approach alone gets gradient +0.024. Approach + rock_ori + rail_lat gets gradient +0.007 (3x smaller). The "poison" features reduce the shared `P_nopt`.

**Rule:** Only combine features where the IRL trajectory has MORE of the feature than the demo. In Phase 1, only approach and traveled qualify (the stationary arm is further from the stick and hasn't traveled).

**After Phase 1:** The arm is moving aggressively. Now it has MORE JA, MORE rock_ori, MORE rail_lat than the demo. All gradients point positive. In Phase 2, ALL weights are unlocked (including approach and traveled) — they continue adjusting as the smoothness weights grow. No freezing needed.

This is not a hack — it's a consequence of the softmax probability model. The partition function treats all features as a single cost, so features with opposite gradient signs cancel each other's `P_nopt` contribution.

---

## 13. Trajectory Buffer Lag: Why Weights Overshoot

With the trajectory buffer, the gradient averages over ALL stored challengers:

```
dw = sum_i(P_i * temp * (phi_i - phi_demo))
```

**Example with traveled:**
- Iteration 0: IRL trajectory at 4% progress → `phi_diff = +0.96` → strong positive gradient → weight grows
- Iteration 1: Weight=33, trajectory at 110% (overshoots) → `phi_diff = -0.04` → slightly negative
- But the buffer still contains the 4% trajectory. The averaged gradient is `(P_4% * 0.96 + P_110% * (-0.04))` — still positive because the old entry dominates.

**Result:** The weight keeps growing (33 → 47 → 57) even though the trajectory is overshooting. The buffer entries from before the overshoot keep pushing the weight up.

**Eventually** the buffer fills with overshooting trajectories and the gradient flips negative. But with `mppi_lr=100`, the weight jumps too fast to stabilize before this happens.

**Natural fix in Phase 2:** Smoothness features (JA, JV) penalize aggressive motion. The arm can't accelerate hard enough to overshoot by 50%. The traveled weight finds a natural equilibrium.

**Alternative fixes:**
- Decay old buffer entries (exponential weighting favoring recent)
- Lower learning rate after the first accepted step
- Use a feature that explicitly penalizes overshoot (e.g., distance from goal at terminal time)

---

## 14. Practical Parameter Values

After extensive testing:

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.5 | Costs O(10-50) → `t*C ≈ 5-25` → meaningful probabilities |
| `mppi_lr` | 100 (SGD) / ~1 (Adam) | With raw SGD the gradient magnitude is often O(0.01), so a large lr is needed for the weights to change meaningfully per iteration. With Adam the effective per-coordinate step is ≈ lr, so retune accordingly. |
| `lambda` (MPPI) | 0.05 | Selective enough to differentiate samples |
| `K` (samples) | 128 | Good balance of quality vs speed |
| `H` (horizon) | 40 | Long enough to plan approach + scraping |
| `n_dial` | 3 | Sufficient convergence, diminishing returns after |
| Line search steps | [50, 20, 10, 5, 2, 1, 0.5] | Covers the range from big jumps to fine-tuning |
| Line search type | `opt` (masked) | Only checks weighted features |

---

## 15. Time-Varying Weights: Why One Weight Per Feature Isn't Enough

With a single weight per feature (`n_w=1`), the IRL gradient averages the feature difference over the entire trajectory. For monotonically accumulating features like `traveled`, this creates a masking problem: being too slow early and too fast late cancel out in the sum, so the gradient sees a moderate positive signal even when the trajectory is badly wrong in both directions.

**Example (iteration 6, `n_w=1`):**
- t=0–50: IRL at 0–58% vs demo at 0–75% → IRL is behind → positive gradient
- t=50–66: IRL at 58–109% vs demo at 75–100% → IRL overshoots → negative gradient
- Sum: early lag dominates → net positive → weight keeps increasing → more overshoot

The traveled weight grew monotonically (10 → 26 → 33) despite the trajectory overshooting by 9%, because the cumulative deficit in early timesteps masked the late surplus.

**Fix: `n_w > 1` (time-varying weights).** Each window gets its own weight and gradient:

```
dw_k = P_nopt * temp * Σ_{t ∈ window_k} (φ_irl_t − φ_demo_t)
```

With `n_w=3`:
- Window 0 (early): IRL lags → positive gradient → weight UP (move faster)
- Window 1 (mid): IRL catching up → moderate gradient
- Window 2 (late): IRL overshoots → negative gradient → weight DOWN (slow down)

**The weights don't inherently bias toward later windows.** The gradient is driven by the *difference* between IRL and demo within each window, not by the absolute feature magnitude. If the IRL matches the demo's speed profile, all windows converge to similar weights. The per-window structure only activates when there's a speed mismatch to correct.

**When to use `n_w > 1`:**
- Features that accumulate monotonically (traveled, progress)
- Long trajectories where the motion character changes over time (approach → contact → scraping)
- Phase 2 learning where smoothness in different parts of the motion matters differently

**When `n_w=1` is fine:**
- Short trajectories with uniform motion
- Features that don't accumulate (instantaneous penalties like joint limits)
- Single-feature debugging where you just want to see the gradient sign

---

*Last updated: 2026-04-20*
