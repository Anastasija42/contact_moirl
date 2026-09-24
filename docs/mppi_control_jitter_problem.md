# The MPPI control-jitter problem (and why smooth control is the enabler)

**Status:** superseded by the 2026-07-05 investigation. 2026-07-04.
Companion to `mppi_endpoint_vs_pace_problem.md` and `mppi_force_recovery_and_scaling.md`.

> **UPDATE 2026-07-05 — options tested, see `demo_consistency_and_contact_holding.md`.**
> The options below (§5) were implemented and tested:
> - **(A) post-hoc torque low-pass** (`--tau-feat-ema`): implemented, symmetric on
>   demo+rollout. **Insufficient** — it cleans `Tau`/`Eng`, but the jitter also lives
>   in `JV`/`JA` (velocity/accel), which a *torque* smoother can't touch, so the IRL
>   still fights the fine-control jitter and stalls.
> - **(C) full-horizon MPPI:** **fails on 9-DOF** — one under-optimized 256-sample
>   pass gives saturated bang-bang; block-receding's re-planning is *iterative
>   refinement*, not just feedback.
> Net: jitter-free cost-driven contact is the MPPI ceiling. The working path is the
> **gap-closer on a *feasible* demo** (CSQP-analog contact constraint), then recover
> effort/pace on that manifold with clean smooth torques. Details in the companion doc.

---

## 1. Why this blocks everything

The recovered MPPI rollout has **jittery torque** — `mean|du| ≈ 20–50` per step vs the
human demo's smooth **~1.5**. That matters because two of our cost features are
built on torque:

- `Tau = ‖u‖`  and  `Eng = τ·q̇`

Jittery `u` inflates both. So *any* quantitative claim about torque/energy effort
("the human minimizes proximal load", the effort ramp, the per-joint split) is
**contaminated by solver noise**, not measured from the human. Position and force
features average the jitter out — but the effort features do not.

**Consequence:** without smooth control we cannot trust the effort numbers, and
the effort backbone is the main thing we recover. Smooth control is therefore the
**gate**: with it, Tau/Eng are clean → the recovery is valid → we can run all
subjects and feed the morphology transfer. That's why it's worth solving before
scaling up.

---

## 2. The measurement (rollout-only, fixed weights)

To isolate the *solver's* smoothness from the IRL (which confounds it by picking
different weights), we ran a single forward rollout at the **same seed weights**,
varying only the smoothing knob (`--rollout-only`):

```
config      mean|du|   peak|du|      vs demo (~1.5)
discrete      49.7      117.0          33×          ← no smoothing
ctrl_smooth5  28.0       61.0          19×
ctrl_smooth9  26.4       57.1          18×
ctrl_smooth15 21.1       61.1          14×
spline10      17.8       45.2          12×          ← best, but still 12×
```

**Both `ctrl_smooth` AND the spline genuinely work** at fixed weights (49.7 → 21 /
17.8) — the earlier "they failed on the arm" was an **IRL weight-confound** (the
recovery picked higher-JV weights that masked the smoothing), NOT the mechanisms.
`spline10` is actually the *best* (lower sampling dim, 10 vs 45). BUT both have
**strong diminishing returns and plateau ~18–20**, i.e. **~12× the demo** even at
the most aggressive setting. Halving (or 64%-ing) is not enough.

---

## 3. What we tried, and why each falls short

| mechanism | what it smooths | result | why it's not enough |
|---|---|---|---|
| **B-spline basis** (`spline_M`) | re-parameterizes control to M knots | 49.7 → **17.8** (best) | reduces to 12× demo, then plateaus — the nominal jitter is the floor. (Nondeterminism concern moot now MJX is deterministic.) |
| **Low-pass noise** (`ctrl_smooth`, the toy's mechanism) | the *exploration noise* | 49.7 → 21, plateaus ~20 | smooths the sampled perturbations, but the residual jitter is in the **nominal control** (the MPPI weighted-mean update), which the noise filter never touches |

Both smooth the *sampling*; neither reaches the demo because the floor is the
nominal, not the samples (§4).

---

## 4. The root cause: receding horizon + nominal jitter

Both toy mechanisms target the **sampling**. But the executed control is
`nominal + smooth_noise`, and the **nominal** — the softmax-weighted average of K
rollouts, recomputed every step — is itself jittery step-to-step. Smoothing the
noise removes only part of the variance; the nominal jitter is the floor.

And it's a floor *because* the horizon is receding: only the **first step** of
each replan executes, so a plan that is smooth over its H-window contributes one
jittery sample to the executed trajectory. The toy's smoothing worked because the
toy is **full-horizon** — it executes the *entire* smooth plan. On the receding
arm, neither the spline nor the noise filter can reach the demo's smoothness.

---

## 5. Options that could actually get us there

Ranked by effort / how directly they unblock the effort features.

### (A) Post-hoc torque low-pass for the effort features only — cheapest unblock
Compute `Tau`/`Eng` from a **low-pass-filtered** torque, applied identically to
demo and rollout. The demo torque is already smooth, so this is symmetric and
directly cleans the two contaminated features **without touching the dynamics**.
- *Pro:* trivial, immediately makes the effort numbers trustworthy; symmetric so
  no phantom gradient.
- *Con:* the rollout still *looks* jittery in a torque plot (dynamics unchanged) —
  a presentation caveat, but the *features/claims* become valid. This directly
  answers "the jitter only matters for the effort features."

### (B) Control-rate cost in the SOLVER — smooth the nominal honestly
Add a fixed `‖u_t − u_{t−1}‖` penalty to the **MPPI cost** (a solver regularizer,
NOT a learned IRL feature). The toy concluded "smoothness belongs in the solver,
not as a learned cost" — this is exactly that. It penalizes the *nominal*'s
roughness, which is the actual floor.
- *Pro:* smooths the real executed control (fixes the plots too); principled per
  the toy.
- *Con:* a new solver term to tune; too strong and it fights the task.

### (C) Full-horizon MPPI — smooth by construction
Execute the whole smooth plan (no receding). Removes the first-step-only problem
entirely.
- *Con:* reopens the endpoint-vs-pace and warm-start-contamination issues
  (`mppi_endpoint_vs_pace_problem.md`); heavier.

### (D) CSQP re-solve for the effort numbers
Re-solve the recovered cost with CSQP (smooth by construction) and quote the
effort magnitudes from that. Uses the already-working smooth solver for the
reporting step.

---

## 6. Recommendation

Two-track, matching the two things "smooth control" is needed for:

- **To trust the effort features now (the blocker):** do **(A)** — low-pass the
  torque symmetrically for the `Tau`/`Eng` computation. Cheapest, directly makes
  the recovery valid, lets us run all subjects. This is the gate you flagged.
- **To also get a genuinely smooth executed controller** (nice plots, honest
  dynamics): add **(B)** — a control-rate solver cost — which is the toy's
  "smoothness in the solver" applied to the nominal, the part the noise filter
  can't reach.

Do NOT keep pushing `ctrl_smooth` higher — the sweep shows it plateaus ~20 (14×
demo); it smooths the wrong thing (noise, not the nominal). And do NOT add a
learned smoothness feature — the toy tested and rejected that (Exp 8).
