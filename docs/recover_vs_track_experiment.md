# Experiment: Recover vs. Track — Force, Pace, and Effort

*Which motor-cost features are genuinely **recovered** (data-identified preferences)
vs. **tracked/imposed**, on the real shaving mocap. Determines the paper's claims.*

Subject: **S2**, `down_long`, 5 representative cycles, CSQP inner solver,
Gaussian basis `K=12` (time-varying `W(t)`), per-feature scaling.

---

## 1. Question

For each candidate feature, is its weight set by the **data** (recovered preference)
or by the **regularizer / an imposed target** (tracked)? Three groups:

- **Effort** — per-joint-group `Tau`, `Eng` (+ `JA`, `JV`, `JTC`, `Geo`). Always learnable.
- **Pace** — end-effector rail speed. Reformulated to `progress_vel = ṡ²` (a
  speed-magnitude penalty, **no imposed target**), then **unfrozen**.
- **Force** — pressing. Two forms: **tracking** `(f_n − f_target)²`, and **two-cost**
  `press_force = f_n²` + `press_capacity = (f_n − F_max)²` (recover the force as a
  *fraction of capacity*, `F* = F_max·w_cap/(w_p+w_cap)`).

## 2. Prerequisite fix: the demo must press honestly

The friction actuation (`friction_lib`) only injected **tangential friction**, never
the **normal press** — so the demo's back-solved contact force read ~1 N instead of the
measured ~18 N (see `press_force_transmission_problem.md`). Fixed by injecting the
normal along crocoddyl's exact `ContactModel1D` axis (`R_surface.row(2)`); transmission
went **0.39–0.70 → ~1.0**. Enabled with `--press_in_actuation`, so the demo (and rollout)
press at the measured `f_n` **through the torque**. All force runs below use it.

## 3. Runs

| run | force | pace | warmstart |
|---|---|---|---|
| tracking (warm/cold) | `(f−target)²` | frozen | demo / cold-IK |
| two-cost (warm/cold) | two-cost | frozen | demo / cold-IK |
| **reg_beta sweep** (β=0,1e-3,1e-2) | two-cost | frozen | cold |
| **ULTIMATE** | two-cost | `ṡ²` unfrozen | cold |
| **pace-iso** | tracking | `ṡ²` unfrozen | cold |

## 4. Results

### 4a. Best-iter joint RMSE (deg)

| config | q_norm |
|---|---|
| tracking + frozen task (cold) | **1.41** (best) |
| tracking + frozen (warm) | 1.51 |
| two-cost (warm) | 1.36 |
| pace-iso (tracking + ṡ²) | 2.65 |
| ULTIMATE (two-cost + ṡ² + all) | 2.50 |
| two-cost (cold) | 4.74 (stalled iter 1) |

Adding recovered features (two-cost, ṡ²) loosens the fit ~1° vs. the tracking baseline —
a modest cost, not a collapse. Warm ≈ cold on this (well-conditioned) subject.

### 4b. FORCE — `press_force` recovers, `press_capacity` does **not** (decisive)

`reg_beta` sweep (recovered weight vs. regularizer):

| β | press_force | **press_capacity** |
|---|---|---|
| 0 | 6419 | **10734** (blows up — data doesn't constrain) |
| 1e-3 | 0.37 | **0.34** |
| 1e-2 | 0.94 | **9.10** |

- **`press_capacity` swings 4 orders of magnitude with β → prior-set, not identified.**
  The force *fraction of capacity* is **not recoverable** — because `F_max` is a made-up
  ceiling far above every observed force (nobody presses near capacity → no contrast),
  and any `F_max` taken from the data makes the fraction circular (squeeze argument).
- `press_force` (the effort/magnitude penalty) **does** recover a data-driven, non-zero
  weight (moves over iterations; in `W(t)` it **peaks mid-stroke = during contact**), but
  it too is regularizer-stabilized (blows up at β=0). So force is at best **weakly**
  recoverable; its *level as a preference* is tracked/imposed.

### 4c. PACE — `ṡ²` is recoverable

- Recovered a **non-zero, data-driven weight** (largest first-basis block, `w_hat`≈0.044),
  **loaded early** in the stroke.
- The **stroke still completes** (rail RMSE ~0.001) → the hard terminal constraint
  (`Pxf`/`Vxf`) holds; no "don't-move" collapse.
- Pace is genuinely recoverable because — unlike force — it is **observable** (in the
  motion) **and controllable** (joint velocities). No `F_max`/capacity issue.

### 4d. EFFORT — robustly recovered

**Proximal `Tau` (clavicle + thoracic)** and `Eng` recover in **every** run, config-invariant
— the established "proximal load" signature. `JA` (smoothness) dominates the tracking
baseline; the recovered mass redistributes across configs (identifiability degeneracy:
several weightings fit similarly).

### 4e. Recovered `W(t)` (ULTIMATE)

Raw `W(t)`: **nearly everything ramps up over the stroke** — the *contact envelope* (cost
piles into the scraping phase). Underneath it, a sensible hand-off:
**pace early → `press_force` mid (contact) → distal effort (wrist/shoulder `Eng`) late.**
(For the paper, normalize by the envelope to show the *share* / hand-off, not the ramp.)
Plots: `recovered_weights.png`, `S2_ULTIMATE_Wt.png`.

## 5. Verdict

| quantity | observable? | controllable? | outcome |
|---|---|---|---|
| **effort organization** (proximal Tau/Eng) | yes | yes | **recovered** (robust) |
| **pace** (`ṡ²`) | yes | yes | **recovered** (data weight, early-loaded) |
| **force `press_force`** (effort on `f_n`) | via contact-aware τ | (fed forward) | weakly recovered (regularizer-stabilized), peaks mid-contact |
| **force fraction** (`press_capacity`) | no | no | **not recoverable** (prior-set; needs independent `F_max`) |

**One-line paper claim:** *pace and effort-organization are recovered; the pressing force's
magnitude/fraction is tracked — it is the one quantity that is neither observable nor
controllable, and its identifiability is regularizer-limited absent an independent
capacity measurement.*

## 6. Open / to close the force-fraction question

Recovering the force *fraction* requires an **independent per-subject max-voluntary-press
trial** (`F_max`). Without it, `press_capacity` is prior-set — proven both by the squeeze
argument and empirically by the `reg_beta` sweep.

## 6b. CONTRIBUTIONS analysis (added 2026-07-02) — corrects the raw-weight read

Raw weights are in per-feature-*scaled* space and can't be compared directly. The right
measure is each feature's **contribution** = ∫|W(t)|dt (share of the recovered cost).
Result (integrated |W| share, S2):

| feature group | with-press (pace_iso) | no-press ablation | β=1e-2 | β=1e-3 |
|---|---|---|---|---|
| **smoothness/distal effort** (JV, Geo, Eng_wrist, JA) | dominant ~50% | dominant ~65% | dominant | dominant |
| **press_force** | 15.2% | — (dropped) | 10.5% | 15.2% |
| **Eng_clavicle (proximal)** | 18.6% | 5.1% | 6% | 19% |
| **progress_vel (pace)** | ~0 (not top-9) | 0.0 | **0.3%** | **0.5%** |

Three corrected conclusions:

1. **PACE is NEGLIGIBLE, not recovered.** `progress_vel` contributes **0.3–0.5%** of the
   cost (stable across β, but tiny). The non-zero raw weight was misleading; its
   *contribution* is noise-level. → drop it / do not claim pace as a recovered preference.
2. **FORCE (`press_force`) is COLLINEAR with proximal effort, not separable.** Its
   contribution is regularizer-sensitive (15%→10% as β grows), it moves in lockstep with
   `Eng_clavicle` (19%→6%), and when dropped (ablation) its mass **redistributes into
   Eng_wrist/thoracic/Geo** rather than vanishing. So a *recovered* force weight is
   indefensible → **track it** (this is the quantitative backing for the tracking decision).
3. **The robustly recovered content is SMOOTHNESS + (distal) EFFORT** — JV, Geo,
   Eng_wrist, JA — stable across β and across the ablation. The **proximal cluster**
   (`press_force`, `Eng_clavicle`, `Tau_thoracic`) is an *entangled bundle* (force +
   proximal effort, collinear, regularizer-sensitive) and should be reported as one
   "press-effort" quantity, not separated.

**Refined verdict:** recover the **effort/smoothness organization**; **track** the force
(collinear with proximal effort → not separable); **drop** pace (negligible). Plots:
`{S2_pace_iso, S2_nopress_ablation, S2_ULTIMATE, S2_pressact_tracking_cold}_contrib.png`.

## 7. Reproduce

Launchers in `…/scratchpad/launch_*.sh`; probes `probe_*.py`; recoveries under
`analysis/special/two_cost_force/{S2_ULTIMATE, S2_pace_iso, S2_twocost_regbeta_*,
S2_pressact_*}/`. Flags: `--press_in_actuation`, `--two_cost_force --force_max`,
`--progress_vel_target_mode --target_rail_vel 0 --learn_task_only progress_vel`,
`--ls_cold_start`.
