# MPPI-IRL: force recovery, task-vs-style scaling, and seeding

Design notes from the 2026-07-03 session on getting the MPPI (`KinematicMPPI_MJX`,
`tests/test_phase2.py`) side of the shaving IRL to recover a meaningful cost.
Companion to `papers/mppi_irl_methodology.md`. Cross-refs to the CSQP two-cost
work and the toy study (`src/toy_box_slide_irl.py`).

---

## 1. Why minimizing the force can't recover the force

`press_force = |f_n|/Fmax` (emergent, ref 0) has its cost **minimum at f_n = 0**.
A cost that only contains it drives the press *down* to whatever the task floor
requires (staying on the rock while scraping), not to the human's chosen level.
Empirically the MPPI challenger pressed **~11 N** while the demo pressed **~38 N**
(`[DBG press_force] demo Σ=38.6 vs chal Σ=11.1`). **Minimizing a quantity can
never recover its set-point.**

**Fix — force *capacity* term.** Add `press_capacity = (f_n − Fmax)² / Fmax²`
(min at `f_n = Fmax`). The pair
```
w₁·|f_n|/Fmax   +   w₂·(f_n − Fmax)²/Fmax²
```
has an interior optimum, so the press sits at a recoverable, non-zero level and
the challenger presses at the human's level instead of the 11 N floor.

### Two implementation traps (both hit and fixed)

1. **Demo↔rollout symmetry.** The feature MUST be computed identically on the
   demo side (`human_mppi._compute_run_feature`) and the rollout sides
   (`mppi_cpu`, `mppi_mjx_kinematic._running_phi`), reading `f_n` from the
   **contact solver** on each (demo: `_press_forces`/MuJoCo; rollout: MJX
   `efc_force`), each for its *own* trajectory. The toy (`toy_box_slide_irl.py`,
   `get_traj_features`, default `legacy_force_source=False`) proved this: reading
   the contact force on both sides makes the force feature identifiable; the
   `legacy_force_source=True` ablation reuses the demo's forces on every rollout
   → identical feature → **zero force gradient**. Same bug class as the Geo
   phantom (see §4). Verify with the ~30 s check: the demo-vs-rollout gap in the
   `Phi|opt-cur|` line must look like a *real* preference gap, not a constant.

2. **Squared, not abs — or it's collinear.** With the abs form
   `|f_n − Fmax|/Fmax = 1 − |f_n|/Fmax` on `[0,Fmax]`, capacity is *perfectly
   affine-dependent* on press_force → zero independent gradient (the symptom was
   `press_capacity` and `press_force` showing the **identical** `Phi` gap 0.433,
   and the weight never leaving 0). The **squared** `(f_n − Fmax)²/Fmax²` is a
   genuine second basis function (quadratic vs press_force's linear), so the two
   force costs are separable. Clip `f_n` to ±2·Fmax so it stays O(1) (`σ=1`,
   the natural max at `f_n=0`) — this avoids the σ≈3490 dual-blowup that killed
   the capacity term in CSQP (in MPPI `f_n` is physical Newtons, not a solver
   dual, so it's better-behaved here).

Flags: `--press-capacity-seed`, and set `--force-max ≈ 40` (the human's press),
not 80 — the capacity pulls toward Fmax, so Fmax *is* the target.

---

## 2. Getting the motion right first (contact must hold)

None of the force machinery works if the tool leaves the rock. Two upstream
settings, both from a first-solve travel sweep:

- **Travel is driven by `traveled` (reward), not `progress_vel`.** With
  `--progress-vel-sq`, `progress_vel = ṡ²` is a *pace penalty* (brake) on the
  rollout — raising `--p1_init` makes it travel *less*. The travel driver is
  `traveled = −dist/rail_len` (reward), seeded via `--traveled-seed`.
- **Sweet spot:** `--p1_init=5 --traveled-seed≈130`. Sweep (first-solve travel @
  t=60): `traveled 100→80%`, `130→~100% with rs=1`, `150/180→over-drives and
  TEARS contact (rs=0 from ~t=30)`, `200→127% torn`. **A torn contact = no force
  = both press features dead.** So `traveled≈130` is the ceiling that still holds
  contact; higher trades the whole force experiment for a couple % more travel.

---

## 3. Task-vs-style scaling (`partition_free_only`)

Convergence was slow because the big **frozen task seeds** (`traveled=150`,
`approach/surface/rail_lat/press_force = 5`) enter the MaxEnt partition cost
`C = w·φ`; a tiny mismatch ×(big seed) **saturates the softmax** (`P→0`) and
throttles the gradient on *everything*, including the style weights we want to
move. `partition_free_only=True` scores the partition over the **free
(learnable) subspace only** — the frozen task terms still shape the rollout but
no longer veto the style/force gradient. This is the "seeded task stays put, the
other weights move big" behavior. (Now set in `test_phase2.py` mo_args. The
per-feature scale — `JV σ=11.68` etc. — is a *separate* mechanism that balances
the style features against each other; leave it.)

---

## 4. The open modeling question: is force TASK or STYLE?

`partition_free_only` exposed a real reclassification. The "free subspace" is the
*learnable* set, and because `press_force`/`press_capacity` are in `--extra`
(press_force also in `never_mask`), **the force is now in the style/recovered
subspace, not the frozen task one.** So the IRL will try to *recover* the force
weights rather than treat them as a seeded boundary.

**The tension:**

| view | mechanism | pro | con |
|---|---|---|---|
| force = **TASK** (boundary) | press_force + press_capacity **frozen**, seeded to set ~40 N | matches paper ("force is a task-driven boundary condition") + toy ("magnitude not identifiable → seed it"); clean | force isn't "recovered" |
| force = **STYLE** (preference) | force weights **learnable** | tests whether force *can* be recovered | toy says magnitude is only *directionally* identifiable → weights drift, anchored only by Fmax; not meaningful |

**Recommendation (consistent with the toy + paper): keep force as TASK.** Freeze
`press_force` and `press_capacity` (seed them so the challenger presses at the
human's level), let `partition_free_only` exclude them, and **recover only the
effort backbone** (Eng/Tau/JV/JA/Geo — the identifiable part). The capacity
term's real job is then to make the *seeded* press **sit at the right level**,
not to be a recovered preference. This keeps the story: *effort recovered, force
seeded-to-level, collinearity made explicit.*

Implementation to do this: a "frozen-seed" path so `press_force`/`press_capacity`
are seeded and shape the rollout but have `mask=0` (out of the learnable
gradient). Alternative, if we deliberately want to *test* recovering force as
style: leave them learnable and report the caveat that the magnitude is
seed-anchored.

---

## 5. What's robustly recovered (the actual result)

Across the S2/S3 runs the recovered **effort backbone** is consistent and
readable — the same shape CSQP finds:

- **JV (joint-velocity smoothness) dominates**, ramping low→high through the
  stroke and releasing near the end.
- **JA + Eng_elbow/shoulder/clavicle + Geo** form the secondary tiers, same
  ramp-then-release.
- **Tau terms stay near seed** — torque isn't the recovered lever here; velocity/
  energy smoothness is.

Caveat on comparing `W(t)` to CSQP: MPPI is **receding-horizon** (replans each
step, `H=45`, indexing the *global* `W(t)` by absolute time `t_+h`). So `W(t)`
is the *same object* as CSQP's (weight on `φ(t)`), but the recovered *values*
carry a myopia confound that concentrates in the late windows and shrinks as
`H→T`. Compare **shapes/rankings** ("both solvers agree JV/effort ramps late"),
not raw curves — this is the paper's "directional cross-check" framing.

---

## 6. Fixes this session that made MPPI-IRL work at all

Recorded here so the pipeline state is clear:

1. **Geo phantom (the stall cause).** `Geo = √(vᵀMv)` used the full mass matrix
   incl. the `1e10` armature on locked DOFs; the demo carried a real velocity on
   the locked thoracic, the rollout held it at 0 → `1e10·v²` inflated *only* the
   demo's Geo → constant ~130 phantom gap independent of travel, swamping the
   gradient. Fix: zero locked-DOF velocities before the kinetic-energy compute.
   `|g|` jumped 0.058→0.937.
2. **Base reseed for MPPI.** MJX contact is nondeterministic, so the stored base
   q_norm was a lucky-low draw the noisy probes couldn't beat → "No Step Found."
   Reseed the base (fire for any model list, incl. the single-model MPPI case) so
   base and probes are measured the same way. (WARN `stored != reseed` is
   expected for MPPI — it's the MJX noise, informational.)
3. **q_norm hygiene (shared with CSQP).** Excludes pinned DOFs (rock always,
   locked trunk when `--lock-thorax`); CSQP also got `reset_rho/reset_y=True` for
   determinism. See `project_csqp_qnorm_determinism`.

New flags added to `test_phase2.py`: `--traveled-seed`, `--press-capacity-seed`,
`--max-iter`, `--lock-thorax`; parity flags `--press-emergent`,
`--progress-vel-sq`, `--force-max`. `press_capacity` appended to `KEYS_RUN`
(index 22, so no index shift).
