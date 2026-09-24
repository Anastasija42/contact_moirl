# Getting the MPPI to discover the whole down-stroke

**Goal:** find an MPPI-IRL setup (`--pd-track`, S2 `down_long`, cycle 5) where the challenger
**reproduces the full down-stroke** (≈100% rail travel, sustained contact, realistic torques) AND the
outer IRL loop can still **recover the cost** (stable line search, weights that converge). Right now we
get one or the other, never both. This doc records why, and the concrete setups worth trying.

Date: 2026-07-09.

---

## 1. What "full discovery" has to reproduce (the demo)

The demo is the recorded human motion, IK'd to the 9-DOF arm, made into a consistent (q, τ, f) triple:

- **Travel:** 0 → 0.264 m along the rail = **100%** over T=66 steps (dt≈8.6 ms).
- **Required torques** (`us_demo` peak |τ| per joint, Nm) — all **within** the anatomical caps:

  | joint    | demo need | cap  | headroom |
  |----------|-----------|------|----------|
  | sho_Z    | **71.9**  | 92   | yes |
  | sho_X    | 23.9      | 71   | yes |
  | sho_Y    | 23.1      | 52   | yes |
  | elb_Z    | 20.2      | 77   | yes |
  | wri_Z    | 16.3      | 100  | yes |
  | clav_X   | 14.2      | 100  | yes |
  | elb_Y    | 4.4       | 15   | yes |
  | wri_X    | 3.2       | 100  | yes |

  **⇒ The motion is physically feasible. There is NO actuator-cap ceiling.** A discoverable setup
  must exist. (thorax is locked; its 190 Nm reading is the lock artifact, ignore.)

- **Force:** measured ATI profile, ~12–62 N over the stroke (demo_rec values in the logs).

So the challenger must command ~20–72 Nm on the proximal joints to accelerate the limb through the
stroke. That single fact is the crux of everything below.

---

## 2. Root-cause chain (why the arm under-travels)

1. **The pd-track controller has no inertial feedforward.** It applies
   `u = τ_grav + Kp·(q_ref − q) − Kd·q̇` — gravity compensation **plus pure position feedback**. It only
   reacts to *accumulated* position error and never *commands* the large inertial torque the motion needs.
   At the natural gains it produces **|u|≈6.5 Nm peak** where the demo needs up to **72** → the arm lags →
   travel floors at **~38%**. (The demo itself is ~92% feedforward, ~8% PD — our challenger supplies ~0%
   feedforward.)

2. **Wide noise + frequent replanning is a *workaround* for the missing feedforward.** With
   `--noise-scale=2.5 --n-apply=8` + the lifted traveling warmstart, some sampled references *lead* the arm
   far enough that the large `q_ref − q` error drives a large PD torque (|u|→72). In a single rollout that
   reached **74.7% travel**. So the arm *can* be pushed through — but only by manufacturing big tracking
   errors through exploration, not by commanding the torque.

3. **That same wide noise makes the rollout a high-variance random function of the weights.** At noise 2.5
   the *same* weight setting produces travel anywhere from **7% to 105%** between line-search probes
   (observed: LS step 2 → 105% travel, q_norm 19.9; LS step 7 → 7.6%). Combined with MJX contact
   nondeterminism (the `stored q != reseed` warning), the IRL objective `q_norm(w)` is no longer a stable
   function — the line search compares a fixed baseline against wildly-varying fresh draws → **No Step Found**.

4. **The line-search mask bug (fixed) was a separate, real blocker.** `_ls_feats` used only `EXTRA` (the
   effort backbone), so the completion driver (`traveled`/`progress_vel`, which live in `phase1_feats`) was
   invisible to the acceptance merit `opt_div`. Fixed to `EXTRA + phase1_feats`
   (`tests/test_phase2.py`). After the fix, `traveled` climbs for the first time (5.0 → 6.6 at noise 1.5).

**The tension in one line:** travel needs wide exploration (high noise) because the controller can't command
torque; wide exploration makes the objective too noisy for the IRL to optimize. Feedforward is the thing
that would let us drop the noise.

---

## 3. Configs tried this session

| config | optimizer | travel | \|u\| peak | IRL outcome |
|---|---|---|---|---|
| noise 1, single-shot, no warmstart | LBFGS | ~33% | ~6.5 Nm | q_norm 3.63; `traveled` frozen (pre opt_div fix) |
| noise 1, warmstart, single-shot | LBFGS | ~38% | ~6.5 | MPPI re-optimizes travel away |
| **noise 2.5, replan 8, warmstart, traveled=50** (ceiling test, fixed weights) | — | **74.7%** | **72** | not an IRL run; one lucky low-variance draw |
| noise 1.5, replan 8, warmstart | LBFGS | ~38% | ~20–27 | **`traveled` climbs 5→6.6**, effort backbone recovers (JV→4.7, Geo→2.2), then No Step Found |
| noise 2.5, replan 8, warmstart | SGD | fails | — | **No Step Found iter 1**, weights stay at init; probes swing 7–105% |

**Read:** the machinery is correct (opt_div fix works, effort backbone recovers, `traveled` climbs). The
only missing piece is a **reliable** full-travel rollout that is a **low-variance function of the weights**.

---

## 4. Setups worth trying (ranked) to get full-stroke discovery

### A. Add inertial feedforward to pd-track  ← most promising
Change the controller to **computed-torque / inverse-dynamics feedforward**:
`u = τ_grav + M(q)·q̈_ref + C·q̇_ref + Kp·(q_ref − q) − Kd·q̇`, where `q̈_ref` comes from the reference
(warmstart demo path, or its spline). Then the controller **commands** the ~72 Nm directly from the
reference instead of needing a huge tracking error to generate it. Consequences:
- Travel becomes reliable at **low noise** (no need for leading-error exploration).
- Low noise ⇒ low rollout variance ⇒ the line search works ⇒ IRL recovers the cost.
- Symmetry: the demo is already ~92% feedforward, so demo and challenger effort features become
  computed the same way (fixes a phantom-gradient source too).
This directly removes the reason we needed noise 2.5. **Try first.** (Feedforward can be the warmstart demo
accelerations; the IRL still shapes the residual + the cost.)

### B. Deterministic CPU MPPI rollout for the IRL
`mppi_cpu.py` is reproducible (MJX contact is the nondeterminism source — see determinism notes). A
reproducible objective makes the line search viable even at higher noise (the `stored q == reseed` case).
Slower per solve, but removes the variance wall. Good fallback / cross-check if A is fiddly.

### C. Reduce MPPI rollout variance directly
Lower the MPPI temperature (λ) and/or raise annealing passes (`--n-dial` up, more `plan_iters`) so the
inner MPPI *converges* to a tight optimum at each weight setting → travel becomes a near-deterministic
function of the weights → a middle noise (~2.0) becomes optimizable. Cheapest to try; uncertain payoff
(n_dial 2→8 didn't move travel in the free-torque case, but this is about variance, not travel).

### D. A warmstart that HOLDS (feedforward the warmstart torques)
Today the lifted traveling warmstart seeds only the *position* reference and the MPPI re-optimizes travel
away at seed weights. Seeding the **feedforward torque** (demo `us`) as the nominal — with PD only
correcting — keeps the arm on the traveling path so the sampler explores *around* a full-stroke motion
instead of rediscovering it. (Overlaps with A; A is the principled version.)

### E. Paired / common-random-number line search
Make the baseline and every probe use the **same** noise draw so rollout variance cancels and only the
weight-change effect remains. `--fixed-samples` intends this but MJX contact breaks it — so E only works
on top of B (deterministic rollout).

### F. Two-phase: discover motion, then recover cost
Phase 1: high noise + forced `traveled` to *find* a full-travel motion; freeze that as the reference.
Phase 2: low noise, PD-track the frozen motion, recover the effort/force cost around it. Splits the
travel problem from the identification problem. Pragmatic if A–C stall.

---

## 5. What NOT to conclude
- **Not** an actuator/anatomical ceiling — every joint has cap headroom (§1).
- **Not** "the demo is unrealizable" — it's a feasible recorded motion.
- **Not** an optimizer choice (SGD vs LBFGS) — both fail at noise 2.5 for the same reason (objective
  variance). SGD failed *harder* (No Step Found iter 1). The fix is variance, not the optimizer.
- **Not** the feature choice (`traveled` vs `progress_vel`) — the floor is the controller + rollout variance.

## 6. UPDATE — the partition experiment: it's convergence depth, not feedforward

`--sample-partition` (added to `tests/test_phase2.py`) samples the 512 MPPI rollouts at the seed weights
and prints per-sample rail travel % vs softmax weight. **Result at noise 1.5, traveled=5, single-shot H=66:**

```
plan_iters=1 :  best 134%  median 68%  → plan travels 27.1%
   [20,40)% : 42 samples, 100.0% softmax weight, cost 91.1   ← ALL weight on the cheap short bin
   [80,400)%: 158 samples,  0.0% weight,          cost 99.6  ← long ones exist, get nothing
plan_iters=6 :  best 151%  median 87%  → plan travels 75.3%
   [60,80)% : 140 samples, 99.8% weight                       ← weight migrated onto traveling samples
```

**Key findings that reframe the whole problem:**
1. **The long trajectories already exist at noise 1.5** — the *median* sample travels 68%, 158/512 travel
   80–134%. No feedforward and no noise 2.5 are needed to *produce* travel. (My §2/§4-A framing that the
   controller "can't command torque" was wrong: plenty of samples command it.)
2. **The softmax down-weights them.** At λ=0.05 (near-greedy) the update dumps 100% of its weight on the
   cheapest 20–40%-travel bin because long samples cost more effort → the plan collapses to 27%. This is
   exactly the "they exist but aren't chosen" hypothesis.
3. **More convergence passes fix it.** Iterating the from-start plan 6× ratchets the nominal onto the
   traveling samples → **75% travel at the SAME seed weights, at low noise 1.5.** So the lever is **MPPI
   convergence depth (`n_dial` / plan passes)**, and it works at a noise low enough that the IRL objective
   stays smooth and optimizable.
4. Likely why the IRL runs floored at 38%: `n_dial=4` + receding replanning (`n_apply=8`) didn't converge
   the plan as far as 6 from-start passes.

**Revised ranking of §4:** the top move is now **C (raise convergence: `n_dial`↑, single-shot) at low
noise**, NOT A (feedforward). Feedforward/high-noise are unnecessary — the traveling samples are present;
we just need enough softmax passes to select them, at a noise the line search can handle.

## 7. Immediate next step
Re-run the IRL at **noise 1.5, single-shot (`--horizon=66 --n-apply=66`), `--n-dial=8`**, opt_div fix +
warmstart, LBFGS. Success signature: the `LS baseline` rollout travels ~75% (not 38%), q_norm drops, and
`traveled` + the effort backbone converge. Optionally sweep `n_dial` (6/8/12) and temperature λ (lower =
sharper selection of the traveling samples). Command in the session notes / `experiments/pdtrack_ndial8.log`.
