# Design: the rollout must *compute* the press, the demo must *impose* it

*How the pressing force enters the CSQP inner solve, so that the force is
**recovered** (challenger solves for it) rather than **imposed** (fed forward on
both sides). Fixes the flaw where `--press_in_actuation` pinned the force on the
rollout too, defeating the whole point of the CSQP contact dual.*

---

## 1. The flaw

`--press_in_actuation` puts the normal press into the actuation as a feedforward,
`τ_applied = u + J_cᵀ f_n`, so the `ContactModel1D` dual comes out pinned at the
**measured** `f_n`. That is correct for the **demo** — a fixed, measured
trajectory whose internal press is otherwise underdetermined (back-solves to
~1 N). But the feedforward lives in `create_solver`
(`human_crocoddyl.py:934`), i.e. in the **running models the OCP actually
solves**. So the *challenger* also presses at `f_n` **by imposition**, not by
choice.

Consequences:
- The rollout can't *disagree* with the demo on force → **no force contrast** in
  the IRL gradient.
- The `press_force` feature is **near-inert on the rollout side**
  (φ_rollout ≈ f_n² always) → the demo−rollout difference that would move the
  weight is fake.
- **The force verdict (collinear / weakly identified) was measured under force
  imposed on both sides** — not a clean test.

This contradicts the paper's own claim that CSQP *"carries the contact reaction
as an explicit dual variable and recovers the force weight."* It can't, if we
pin the dual before the solver runs.

## 2. Why the two sides are genuinely asymmetric

The press is a **no-motion internal force**: a family of postures reproduces the
same motion with different contact forces. So the force is **not** determined by
the kinematics alone.

- **Demo** — a *fixed* trajectory. There is no optimisation to pick the press, so
  it must be **injected** (the measured `f_n`). Evaluated on clean models the
  demo's dual is ≈ 0 (see `consistent_us`, `run_csqp_population_irl.py:78`:
  `contact_aware_demo=True` keeps `u_RNEA`, and it is *the feedforward* that the
  contact reacts against to give λ ≈ f_n — remove it and the demo stops
  pressing).
- **Rollout** — the OCP *solves*. Given a `press_force` cost, the solver chooses
  torques `u`; the `ContactModel1D` **dual λ** is the reaction to those torques.
  The force is **computed**, and it is *allowed to be wrong* — which is the only
  way it can be *recovered*.

> **Note (why the force still varies — the point that was wrong before).** "No
> motion ⇒ force fixed" is only true for a *fixed* trajectory. The OCP varies
> motion **and** torque jointly; to press with λ it must supply `τ_press = J_cᵀλ`,
> so as the cost weights change the torques change and **λ changes with them**.
> The force is genuinely torque-coupled and has a real gradient. Whether it is
> *separable* from proximal effort (`press_force ∝ λ²` vs. proximal
> `Tau ∝ (J_cᵀλ)²` move together) is the identifiability question — and *only the
> corrected run answers it*, we do not pre-judge.

## 3. The design

### 3.1 Solve side — free the normal

`create_solver` builds the running models **without** the normal feedforward. The
`press_force` cost (`= ½‖λ‖²`, read from the contact dual) is what drives the
press. The dual is now free.

`--press_in_actuation`'s **normal injection becomes demo-only.**

### 3.2 Friction — Option B (consistent with the dual)

The actuation also injects **tangential friction** `f_t = μ·f_n·(v/|v|)`. Since
the normal `f_n` is now the *dual* (variable), the friction must use the **same**
dual `f_n`, so normal and tangential are one consistent contact wrench.

The circularity — the actuation runs *before* the dynamics produce the dual — is
resolved by a **lagged dual**: the friction at step `t` uses `λ` from step `t−1`
(seeded at step 0 with the profile). Forces are smooth across nodes, so the lag
is a sub-step approximation; optionally a 1–2 pass fixed-point per node tightens
it. This keeps `μ·f_n` and the normal `f_n` referring to the *same* force instead
of a fixed estimate.

### 3.3 Feature side — no special-casing needed (the simplification)

Originally scoped as a `contact_lambda` branch in `get_traj_features` (demo
injects, rollout reads the dual). **That was over-designed.** Once the model is
clean (§3.1–3.2), the demo is just *the rollout with `u` loaded instead of
solved* — **same model, same `get_traj_features`, no branch.** The only thing that
differs is how the demo's `u` is computed:

```
u_demo = u_RNEA − τ_bias − J_cᵀ f_n           # bake_press (consistent_us)
```
where `τ_bias` cancels the friction feedforward and `J_c` is the model's EXACT
1-D contact row `(R_surface · J_lin)[2]`. Then, for BOTH demo and rollout, the
model computes the same way:

```
τ_act(t) = u(t) + J_cᵀ f_fric(t)   (actuation.calc)   press_force(t) = ½‖λ(t)‖² (dual)
```

- **Demo**: `u` baked → the contact dual replays at `f_n`; `τ_act = u_demo +
  J_cᵀf_fric = u_RNEA − J_cᵀf_n` (friction cancels).
- **Rollout**: `u` solved → the dual is free; `τ_act = u + J_cᵀf_fric`.

Symmetric **by construction** — the press enters `u` once, no double-count, no
demo/rollout formula split. Anti-`project_mppi_demo_feature_symmetry` by design.
Verified (S2): demo dual replays 19.83 N (0.94 of measured 21), features finite.

## 4. Sign / bookkeeping (so we don't flip a transpose again)

- Contact frame convention follows the existing `ContactModel1D` / `friction_lib`
  fix (`R_surface.row(2)` normal; `project_press_actuation_transmission`).
- Contact-aware muscle torque is `u + J_cᵀλ` in **this codebase's sign**
  (λ = press the body exerts); magnitude `|τ_press| = |J_cᵀλ|`, so pressing harder
  ⇒ more muscle torque (`project_irl_phi_contact_aware_tau`).
- `press_force` uses the **normal** component only (transverse is friction, not
  press; `project_normal_force_approximation`).

## 5. Implementation checklist

1. `human_crocoddyl.py :create_solver (~934)` — drop the 5-arg normal-injecting
   actuation from the solve; keep the (now dual-consistent) friction actuation.
   Add a `self._press_demo_only` / demo-eval switch.
2. Friction actuation (`friction_lib`) — accept a **lagged dual** `f_n` per step
   instead of the profile (Option B). Seed step 0 with the profile.
3. `get_traj_features (~956)` — compute `τ_act = u + J_cᵀλ` and `press_force =
   ½‖λ‖²` with `λ` = injected profile (demo) or dual (rollout). Add a
   `contact_lambda=None` arg: not-None ⇒ demo (use it); None ⇒ rollout (read the
   dual). Route `contact_lambda=profile` at the demo-feature call.
4. Demo `u` stays `consistent_us(contact_aware_demo=True)` = `u_RNEA` (no bake-in).

## 6. Test BEFORE any IRL run (probe)

A probe on S2/down_long, **not** a full recovery, checking:

1. **Demo unchanged** — demo `press_force` feature ≈ measured `f_n²`
   (~18 N) as before; demo `τ_act` (Tau/Eng) includes the press.
2. **Rollout is free** — sweep the `press_force` weight and confirm the
   rollout's **dual λ responds** (higher weight ⇒ different press). If λ is
   constant across weights, the normal is still pinned somewhere — stop.
3. **No double-count** — demo `τ_act` with the new path == demo `τ_act` under the
   old feedforward (same number), i.e. the press enters exactly once.
4. **Friction consistency (B)** — `μ·f_n` at each node uses the same `f_n` (lagged
   dual) as the normal; check `f_t/f_n ≈ μ·(v/|v|)` at the solution.

Only when 1–4 pass do we run the corrected S2 force recovery.

## 7. What we expect (and how it feeds the papers)

The corrected test is the *clean* version of the force-identifiability question.
Expectation (not a verdict): the rollout's press now varies with the cost, but
`press_force ∝ λ²` still tends to co-move with proximal `Tau ∝ (J_cᵀλ)²` because
the torque that produces the press *is* proximal effort. If it stays collinear,
that is **rank deficiency of `J_ω` from linearly-dependent costs** (Colombel
et al., *On the Reliability of IOC*, ICRA 2022) — now demonstrated with the
rollout genuinely free, not with the force imposed on both sides. The
task-velocity feature `ṡ²` (tool linear speed, = box-lifting Φ6, Sabbah/Mehrdad
et al., Humanoids 2025) and effort/smoothness organisation are unaffected.

Either way, this run — not the imposed-on-both one — is what belongs in the
paper's identifiability section.

---

## 8. Probe results (2026-07-02, S2/down_long, loose-tol CSQP)

Ran the §6 make-or-break before building the feature side
(`scratchpad/probe_rollout_dual.py`, `probe_press_leverage.py`). Only
`create_solver` was changed (normal freed in the solve); the friction still used
the profile seed (lagged-dual not yet wired), and the demo-side feature symmetry
is not yet built — so these are *solve-side* results.

**8.1 The normal is no longer imposed** — sweep `press_force` (tracking form),
read the dual mean|λ| over the contact window:

| scheme | λ @ w=0.01 | λ @ w=2.0 | spread |
|---|---|---|---|
| OLD `press_in_actuation` (imposed) | 20.74 | 21.04 | **0.30 N** (pinned) |
| NEW `press_normal_dual` (free) | 19.75 | 21.03 | **1.28 N** (responds) |

**8.2 The press is FULLY controllable** — penalty form `press_force = f²`
(`press_capacity` = 0), the dual collapses top-to-bottom:

| `w_press` (penalty) | 0.0 | 0.01 | 0.1 | 1.0 | 100 |
|---|---|---|---|---|---|
| mean\|λ\| (N) | **16.2** | 4.5 | 0.5 | 0.05 | 0.06 |

100% drop → the press is *freely chosen by the cost*, **not** motion-floored.
(The "modest 1.28 N" in 8.1 was the *tracking* cost pulling λ to a fixed target,
not the true leverage — the penalty sweep is the honest measure.)

**8.3 The key finding — the demo's press ≈ the effort-induced natural press.**
At **zero** press cost the effort-minimising motion already presses **~16 N**,
right next to the measured demo (~18–20 N). So the human's press level is
essentially *what the arm does to scrape under effort minimisation*; a `f²`
penalty only *reduces* it. Consequences for recovery:

- The recovered `press_force` weight sits **near zero / at the natural operating
  point** — the "how hard" is carried by the **effort + motion** costs, with
  `press_force` a fine-tuning rider.
- The small demo excess (~18–20 vs. natural ~16) is exactly what the two-cost's
  `(f−Fmax)²` *upward* pull supplies. Whether that gap is a preference or model
  slack is the recover-vs-track question — now testable **with the force free**,
  not imposed on both sides.

**8.4 Friction-frame bug (found + fixed).** The demo replay only recovered ~77% of
the normal *not* because of a transmission loss but because the **4-arg friction
is built in the LOCAL x–y plane**, and the surface is tilted ~**45°** (LOCAL-z vs.
`R_surface.row(2)`). So `μ·f_n` leaked **~3–4 N onto the normal axis** and the 1-D
dual read 16 instead of 21. Surface-frame friction leaks **0.00 N** (measured).
Fix: added an `inject_normal` bool to the 5-arg `friction_lib` actuation
(`f_surf.z = 0` when false); `press_normal_dual` now builds the **surface-frame
friction with `inject_normal=False`** — tangential only, no normal, no leak.
Verified: normal recovery **16.11 → 19.22 N** (0.91; matches the friction-*off*
19.83). This also **reconciles the "transmission ≈1.0"**: that was measured
friction-*off* (`probe_exact_jc` set `mu=0` *before* `set_force_target_profile`,
which rebuilds the actuation via `create_solver`). Residual ~9% (19.2 vs. 21 at
the contact mean) is numerical-`ddq` replay + the ~1 N motion baseline; it tightens
in the real solve.

**Status:** §6 checks 1–2 (demo unchanged / rollout responds) pass at the solve
level; friction consistency (§6.4) now correct (surface-frame, no leak). Still to
implement: symmetric `get_traj_features` (`contact_lambda`: demo injects the normal
via the exact 1-D `J_c`, rollout reads its dual), demo routing, and the Option-B
lagged-dual update (`_update_friction_from_duals`; C++ `target_fn` + `inject_normal`
setters already built).
