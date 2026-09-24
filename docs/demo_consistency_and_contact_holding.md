# Demo consistency & cost-driven contact-holding (MPPI)

**Status:** investigation closed with a clear verdict. 2026-07-05.
Companion to `mppi_control_jitter_problem.md`, `mppi_force_recovery_and_scaling.md`,
`contact_penetration_problem.md`.

This documents a long investigation triggered by one question: **can the MPPI
challenger hold the sliding contact and press from the *cost* alone — not from a
feedback crutch (the gap-closer)?** The answer forced us first to fix the *demo*,
and then exposed a genuine MPPI ceiling.

---

## 1. The chain of reasoning

1. The gap-closer (`fc_kp_close`) reliably holds contact (100%), but it **imposes**
   contact — a fixed-gain feedback that pulls the hand onto the rock. That converts
   MPPI into constrained-contact CSQP and throws away MPPI's appeal (discovering
   contact from the cost).
2. So we wanted contact from the **cost**. But the challenger chases the **demo**'s
   features. If the demo's own `(q, τ, f)` don't satisfy the dynamics *together*,
   we're asking the challenger to match an **infeasible target** — and only a
   crutch can fake that. **So the demo must be self-consistent first.**

---

## 2. Fixing the demo

### 2.1 The inconsistency
The demo was three separately-sourced pieces: `q` (mocap), `f` (force sensor,
26.5 N mean → 67 N peak), `τ` (computed). Diagnostics:

- **`mj_inverse`-with-contact is garbage here** (1919 N, |τ|≈752). Finite-differenced
  mocap acceleration violates the soft contact, so MuJoCo invents a huge constraint
  force. Do **not** build the demo torque that way.
- At the **static mocap posture** the model produces only **~14 N** (vs the sensor's
  26.5 N), and its force *profile* was anti-correlated (−0.98) with the sensor.

### 2.2 The resolution — induce the force *dynamically*
The arm doesn't *sit* at the posture; it **drives** the pressing stroke. Simulating
that (PD-track the mocap) induces the real force:

- Pure PD-track (faithful `N_SUBSTEPS`, no feedforward) → ~5 N, still anti-correlated.
- **Add a press feedforward scaled to the sensor** (`get_tau_from_trajectory`,
  `force_gain≈0.1` with the stiff internal PD) → **26.5 N = sensor mean**, tracks
  mocap to **0.9°**, **100 % contact**.

> **Caveat that bit us:** an early "26 N, +0.93 correlation from pure tracking" was
> an **under-integration artifact** (1 `mj_step`/frame instead of `N_SUBSTEPS=3`).
> Always use faithful substeps in these demo/replay sims.

### 2.3 It's open-loop reproducible — the key property
"What open-loop torque produces that motion and force?" — **the recorded control
itself.** Record the *exact* controls the tracking applied, replay them open-loop
(no feedback) in the same deterministic sim:

```
PASS1 PD-track   : q-RMSE vs mocap = 3.6°     force = 26.2 N (66/66)
PASS2 OPEN-LOOP  : q-RMSE vs PD-traj = 0.00°  force = 26.2 N (66/66)   (max diff 1.4e-5°)
```

The feedback is **baked into the recorded torque values**; replaying them re-executes
the identical dynamics. (The earlier "demo replay drifts to 0 N" was from replaying
the *analytic* `τ = M q̈ + h + Jᶜᵀf`, which is *not* the applied control.) A serial
arm is open-loop-unstable in general, but a `(q, τ)` pair that **exactly satisfies
the discrete dynamics** reproduces open-loop by construction — which the recorded
control does, and which also puts the demo on equal footing with the MPPI challenger
(itself a forward rollout).

**Consistent demo (`--consistent-demo --demo-force-gain=0.1`):** `q` = tracked
(≈mocap, 0.9–3°), `f` = induced 26.5 N (= sensor), `τ` = recorded control
(open-loop reproducible). This is the feasible target the recovery needs.

---

## 3. Cost-driven contact on the feasible demo — the exploration

With a feasible demo, we removed the gap-closer and tried to make contact
cost-driven. It **still collapsed** (41 % contact, IRL stalls). So the collapse was
**never the demo** — it's the MPPI challenger. Three attempts:

### 3.1 The press DOF was blind (fixed)
`press_control` adds a sampled `press_amp` along the contact normal — but it was
**gated to `dist < 0`** and used the **contact-constraint Jacobian `efc_J`**, which
vanishes when separated. So the press DOF could *modulate* force while touching but
**couldn't pull back to re-establish contact.** *"Isn't that what `approach` is
for?"* — yes: `approach` is the **incentive**, but an incentive needs an **actuator**,
and the actuator was blind exactly when needed.
**Fix:** apply the sampled `press_amp` along the **geometric normal** `J_n_geom`
(always available) when separated → the `approach` cost can now drive it, cost-driven.
Helped at the margin (28→ up), not a solution alone.

### 3.2 Tighter challenger control (option "2")
Positional precision is the real lever for a *sliding* contact. Sweep at seed:

| config | contact | `\|du\|` (demo≈1.5) |
|---|---|---|
| spline-5, n_apply-20 | 28 % | 2.8 |
| spline-12 | 57 % | 7.7 |
| spline-20 | 71 % | 8.9 |
| **spline-20, n_apply-10** | **85 %** | 13.1 |
| per-step (no spline) | 71 % | 12.2 |

More knots + more reactivity → up to **85 %** contact — but `|du|` climbs to ~13
(≈9× demo). Per-step is *not* best (spline temporal coherence helps the sliding
contact). Contact caps ~85 %; the last ~15 % is the late-stroke, travel-driven
lift-off.

### 3.3 Feature-smoothing to rescue option 2 (the torque-EMA) — **dead end**
Implemented the post-hoc torque low-pass (`mppi_control_jitter_problem.md` option A):
`--tau-feat-ema`, a causal EMA on the torque **for `Tau`/`Eng` only**, mirrored on
demo (`get_traj_features`) and rollout (`_running_phi` scan carry). Ran the full
cost-driven IRL with fine control + smoothing:

```
recovered: contact 41 % (seed was 85 %),  |du| 27.7,  "No Step Found"
```

**Why it fails:** the IRL drives the weights toward jittery, contact-losing configs
and then stalls, because **the jitter also lives in `JV`/`JA`** (velocity/accel),
which the *torque* smoother doesn't touch. The demo is smooth (low `JV`/`JA`); the
fine-control challenger is jittery (high `JV`/`JA`); the IRL cannot reconcile
"hold the contact (needs fine/jittery control)" with "match the smooth demo
(penalizes that jitter)". Smoothing `JV`/`JA` too would hide the real signal we're
recovering — not a fix.

Note this also settles two options from `mppi_control_jitter_problem.md`: **(A)
post-hoc torque low-pass is insufficient** (jitter is broader than `Tau`/`Eng`), and
**(C) full-horizon MPPI fails on 9-DOF** (one under-optimized 256-sample pass over
9 DOF × 66 steps → saturated bang-bang; block-receding's re-planning is *iterative
refinement*, not just feedback).

---

## 4. Verdict & recommendation

- **Cost-driven contact-holding is the genuine MPPI ceiling here.** The demo is
  feasible; the press DOF is un-blinded; fine control reaches only ~85 % contact and
  injects `JV`/`JA` jitter the IRL then fights. MPPI-sampled control can't match the
  demo's frame-by-frame tracking precision on a *sliding* contact.
- **Use the gap-closer on the consistent demo.** It is no longer a philosophical
  compromise: the demo is feasible (26.5 N, 0.9° tracking, open-loop reproducible),
  so the gap-closer holds the *real* demo contact — the **MPPI analog of CSQP's hard
  contact constraint**. We then recover **force-magnitude / effort / pace on that
  manifold**, with clean smooth torques (the gap-closer is silent while in contact,
  `tau_close ∝ max(gap,0)`).
- CSQP remains the precise engine; MPPI is the directional cross-check that
  independently reproduces contact + the effort/pace structure.

### Flags introduced this investigation
`--consistent-demo`, `--demo-force-gain`, `--tau-feat-ema` (kept, off by default),
`--blend-steps` (re-plan spike softener), plus the un-gated press DOF (geometric
normal when separated) and `return_qtrack` in `get_tau_from_trajectory`.
