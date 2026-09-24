# Press-force recovery — findings & method (2026-07-04)

Goal: recover the **press force** of a sustained-contact scraping (shaving) task
**from the cost** — i.e. the force should emerge from the torques the solver
explores against the rigid contact, not be injected. Testbed: 9–10 DOF human arm,
CSQP population IRL, `src/run_csqp_population_irl.py`.

This doc records what we established and the exact recipe, so the overnight sweep
(`scripts/force_sweep.sh`) can be double-checked in the morning.

---

## Key findings

1. **The "20 N stall" is `Fmax/2`, not a solver limit.**
   With the two-cost force (`press_force = f²` down, `press_capacity = (f−Fmax)²`
   up), `F* = Fmax·w_cap/(w_press+w_cap)`. Under `--feature_scale` the two press
   weights end up *effectively balanced*, so **`F*` pins at `Fmax/2` regardless of
   the raw capacity weight** (dominant_val 2→60 all gave 20 N at Fmax=40).
   Verified: Fmax=80 → demo presses **40.0 N** exactly. **So Fmax sets the press
   level, not the raw weight.**

2. **The force gradient is NOT blind.** `--check_press_grad`: analytical `df_du`
   equals finite differences (~5–6, non-zero), `df_dx` non-zero. The solver *sees*
   how torque moves the contact force. (Earlier `dtau_dx=0` hypothesis was wrong.)

3. **Reaching high press is a solver-reachability problem, not physics.**
   A cold / zero-torque warmstart *stalls* at the low plateau — the press lives in
   the **torque** (null-space of the motion), so a zero-torque seed never climbs.
   Fixes: **force continuation** (`--demo_force_continuation --ramp_fracs`) and/or a
   **light pressing warmstart** (`--demo_press_warmstart N`, baked torques pressing
   at N). Light-seed + short ramp `"0.3,0.6,1.0"` → clean 40 N demo (KKT < 1).

4. **The challenger must also reach the press region.** The IRL challenger and
   line-search probes warm-started from demo *states + zero torque* → they also
   stall at 20 N while the demo is at 40 N → phantom gap, press chased forever.
   **`--challenger_press_warmstart`** seeds challenger + probes from the demo
   *torques* (drop `--ls_cold_start`). Demo-warmstart is **not cheating**: baseline
   `q_norm = 4.05 ≠ 0` (challenger relaxes off the demo at wrong weights; climbing
   press is stiff, relaxing is easy).

5. **2×2 result (challenger {cold, presswarm} × accept {q_norm, pareto}),
   synthetic demo 40 N, known `w*_press = 0.5`:**
   - **q_norm == pareto** — identical. Acceptance base is *not* the lever;
     `opt_div` is inert for the press.
   - **Motion recovers** (presswarm: q_norm 3.77→0.85, angle RMSE **1.79°**;
     cold: 3.59° — worse, the phantom gap corrupts the weights).
   - **Press *magnitude* does not recover** — presswarm collapses press_cap (the
     challenger holds 40 N from the warmstart, press is null-space-free) with a
     boundary spike; cold chases it but never closes. **The null-space wall,
     confirmed at full 9-DOF IRL with known `w*`.**

6. **Recovered rollout vs demo:** motion matches (1.79°), **torques do not** — a
   **2–3 Nm proximal offset** (thoracic / sho_Z / clavicle) + 2.4× rougher.
   A *tighter* solve made it **worse** (roughness 0.48→0.73, KKT 359→569, press
   46.7→50.7 N — overshoot) → the roughness is a **non-convergent OCP caused by the
   boundary press_cap spike**, not under-iteration. Chain:
   `q_norm blind to torque → unconstrained press_cap boundary spike → stiff,
   non-convergent, rough, over-pressing rollout.`

7. **The unifying physics (why torque = motion + press):**
   `τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + Jᵀλ`. The first part is fixed by the
   configuration (**redundant with q_norm**); the only free part is `Jᵀλ` = the
   press. Since the **motion is recovered**, the entire torque mismatch **is** the
   press mismatch (the over-press × moment arm, dumped proximally). So:
   **get the configuration + the press right ⇒ the torques follow.** No need to
   recover torques independently.

8. **`τ_norm` acceptance (`--tau_norm_accept`).** Adds torque-trajectory RMSE
   (challenger vs demo torques, per-joint) as a **third pareto axis**
   `(q_norm, opt_div, τ_norm)`. Breaks the kinematic null-space q_norm can't see;
   equivalently, it **pins the press** (the only free part of τ given the motion).
   Smoke: τ has *real* signal (base 9.64 → 2.31 in one step), unlike inert opt_div.
   **Not a cheat** — τ is computed from the same demo (kinematics + model), it's an
   acceptance *metric* (not a warmstart, can't pin the solution), and it uses the
   demo *fully* rather than extra data.

9. **Force = measured boundary, strategy = recovered.** Two formulations:
   (A) impose the measured force as a boundary (`press_in_actuation`) and recover
   only the joint *configuration*; (B) free/emergent force (`press_normal_dual`).
   (B) is the truer model, but the magnitude is **unidentifiable** → practically
   use **(A) for the magnitude** (impose the measured value) and recover
   distribution / pace / effort. `τ_norm` and imposing the measured force pin the
   *same* thing (the press) from two sides.

---

## The recipe (what we run)

Synthetic identifiability test — demo = OCP optimum at a known `w*`, recover it:

- **Demo in the moveable region:** `--target_force 80 --force_max 80`
  (→ demo presses `Fmax/2` = 40 N), `--two_cost_force`,
  `--dominant press_capacity --dominant_val 60 --low_val 0.001`,
  `--demo_press_warmstart 12 --demo_force_continuation --ramp_fracs "0.3,0.6,1.0"`.
- **Challenger reaches the press:** `--challenger_press_warmstart` (and **NO**
  `--ls_cold_start`).
- **Acceptance:** `--ls_base pareto`, with vs without `--tau_norm_accept`.
- **Body/contact:** `--press_normal_dual --press_friction_dual --hard_rail
  --hard_rail_tol 0.012 --windowed_rail --rail_from_contact --static_stick
  --progress_vel_target_mode --target_rail_vel 0 --contact_consistent_accel
  --lock_thorax`.
- **IRL:** `--representative --n_cycles 3 --mode basis --K 12 --tau_split full
  --learn_task_only progress_vel rock_ori rail_lat --feature_scale
  --reg_beta 1e-3 --no_normalize_w --best_last --line_search_steps 12 --max_iter 15`.

Flags added this work: `--check_press_grad`, `--force_mean`,
`--demo_press_warmstart`, `--demo_force_continuation`, `--ramp_fracs`,
`--challenger_press_warmstart`, `--tau_norm_accept`.

---

## Overnight sweep (`scripts/force_sweep.sh`)

Runs **down_long** for **S1, S2, S3** (each individually) and the
**population** (all three, shared `w`), each **without** and **with**
`--tau_norm_accept` — 8 runs, sequential (no CPU contention).

Outputs under `analysis/special/force_sweep_20260704/`:
- `<label>/population_recovery.npz` — the run.
- `logs/<label>.log` — full stdout.
- `sweep.log` — start/exit timeline.
- `commands.txt` — exact command per label.

**What the report should read off each run:**
- `q_norm` convergence (motion recovery) — expect it to converge (presswarm).
- Recovered `press_capacity` **profile** `W(t)` (NOT the scalar — the scalar hides
  the boundary spike).
- With vs without `τ_norm`: does the **proximal torque offset collapse** and the
  **over-press correct** (replay `w_hat`, overlay recovered-vs-demo torques)?
- Across subjects: does S3 (harder init conditioning) behave differently from
  S2/S1? Does the population run pool cleanly?

The decisive question the sweep answers: **does `τ_norm` acceptance recover the
torque/press decomposition that kinematics alone cannot — consistently across
subjects and in population?**
