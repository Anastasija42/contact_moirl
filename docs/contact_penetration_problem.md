# Contact Penetration vs. Force-from-Torque — Design Note

**Date:** 2026-07-03
**Status:** ⚠️ SUPERSEDED root cause (see §0). The penetration is real but is NOT the
demo-gap cause. Kept as a record of the investigation.
**Related:** `rollout_force_dual_design.md`, `press_force_transmission_problem.md`.

---

## 0. CORRECTION (verified 2026-07-03)

Sections 1–7 attribute the baked-demo dynamics gap to the recorded stick penetrating the
rigid contact. **Direct measurement disproves this:**

- On the raw demo (0-iter solve), `get_contact_forces()` reads **~1 N** (the contact
  *constraint reaction*), NOT the ~1000 N I had reported. The **1000 N was a solve
  artifact** — the wild, unregularized `us` the feasibility *re-optimization* invents in
  the null space. It was never in the demo and never the press (the press, 17.7 N, is
  injected through the actuation, not the constraint).
- The demo gap is **~entirely velocity defect** (pos ~3e-3, vel ~0.4/node), and that
  defect is **~95 % TANGENTIAL** (contact-normal component is 3–7 %). So it lives in the
  free/along-surface joint directions — **not** the contact normal.
- **Surface-projecting the penetration out does NOT reduce the gap** (it slightly raises
  the normal part). Penetration ≠ gap.

**Actual root cause (narrowed by elimination):** the demo gap is a ~0.4/node VELOCITY
defect, ~95 % TANGENTIAL. Systematically eliminated:
- ❌ penetration / contact normal — surface projection (`--project_kinematics`) does not
  reduce it; defect is tangential, not normal.
- ❌ the "1000 N" — a solve artifact (wild re-optimized `us`); real demo contact reaction
  is ~1 N; the press (17.7 N) rides the actuation.
- ❌ armature — off (default 0).
- ❌ actuation mapping — `actuation(us) = u_rnea` holds to ~0.003.
- ❌ central-vs-forward differencing — `--demo_euler_consistent` (forward-diff v,a) leaves
  the defect unchanged.

**CONFIRMED root cause:** `consistent_us` bakes `u = RNEA` with **contact-free** inverse
dynamics, but the OCP integrates the **contact-constrained** forward dynamics (ContactModel1D
forces `a_normal = 0`). The recorded motion has a real normal acceleration residual
`a_err = Jc·q̈ + J̇c·v` = **0.4–0.67 m/s² mean, up to 1.75 m/s² max** (≫ 0.1 threshold), so
the rigid contact fights it every node; `M⁻¹` smears the correction across joints → reads
tangential.

**FIX — Path A (`--contact_consistent_accel`, implemented):** project the recorded accel
through the contact before RNEA, `q̈* = q̈ − Jc⁺·a_err`, using the exact ContactModel1D
normal row (`R_t·J_LOCAL`, row 2). **Result: demo velocity defect 0.40 → 0.16 (halved).**
`--demo_euler_consistent` (forward-diff) adds nothing on top. The residual ~0.16 is a
smaller second-order source (contact Baumgarte gains / FD noise in constrained dirs).

**Path B mop-up (`--project_bake_ureg`, tried):** feasibility projection stage-1 (q,v only)
reaches gap ~1e-6 but with wild null-space `us`; adding a stage-2 `‖u − u_bake‖²` from that
warmstart PULLS THE FORCE toward measured (wu=1e-1 → 12–17 N, within ±5 %) but the **gap
reopens to 11–92 and the CSQP filter line search STALLS** (KKT 60–150). Same wall as every
force-pinning attempt: `gap≈1e-6` and `force±5%` are not simultaneously reachable through
projection+u-reg — a solver-stall limitation, not a formulation error. Would need press/u
continuation or a different solver to close both.

**Shippable state:** Path A alone. The baked demo (`--contact_consistent_accel`) has the
gap halved and carries the ~17.7 N press by construction (baked into torque); good enough
for the IRL (which consumes demo features, not a perfect rollout).

None of `--project_kinematics`, `--contact_normal_track`, or the spring idea address this.

---

## 1. Symptom

When we build a *demonstration* from the recorded scraping motion (`--use_recorded`,
`press_normal_dual`, `consistent_us` baking the measured press into the torques), the
demo trajectory is **not a valid rollout of the model**: the multiple-shooting dynamics
defect (`gap_norm`) is **~5–14** (growing toward the end of the stroke), not ~0.

Consequences:
- A feasibility projection that tracks the recorded `q` finds a feasible trajectory
  (gap → 1e-5) but with **unphysical loads**: contact reaction **~740–1290 N** and joint
  torque spikes **~400 Nm**.
- Trying to additionally pin the contact force to the **measured ~17.8 N** *reopens* the
  gap and stalls the solver. Recorded-`q` + measured-force + gap=0 **cannot all hold**.

## 2. Root cause: the recorded stick penetrates the rigid rock

Measured per node, the recorded contact point sits **11–28 mm *below* the surface**
(into-surface velocity ~0.13 m/s), systematically (never random jitter → real
compliance, not mocap noise):

| cycle | penetration | into-surface vel |
|------:|------------:|-----------------:|
| 5     | −13.0 mm    | 0.13 m/s         |
| 17    | −14.7 mm    | 0.13 m/s         |
| 4     | −11.0 mm    | 0.13 m/s         |
| 7     | −14.8 mm    | 0.09 m/s         |
| 10    | −27.6 mm    | 0.22 m/s         |

We model the stick⟷rock contact as a **rigid** `ContactModel1D` (zero penetration).
The real contact is compliant — soft tissue, tool flex, rock give — so the rigid model,
handed a trajectory that drives 1–3 cm *into* it, reacts with a **huge normal force
(~1000 N)** to stop the motion it can't allow. That reaction is the "explosion": it is
the contact's Lagrange multiplier, **not** the press (the press itself is the ~17.8 N
transmitted through the actuation). Reading `get_contact_forces()[:,2]` as the press was
a measurement error on our side.

## 3. What we ruled out

- **Feasibility projection (track `q,v` only).** A feasible trajectory matching the
  recorded motion exists (gap → 1e-5, q drift ~0.01°), but the force/torque are the
  unphysical null-space values (the freed dual picks ~1000 N).
- **Pin the force to measured.** 17.8 N is *off* the feasible manifold at the recorded
  `q`; forcing it reopens the gap (back to 5–28) and the filter line search stalls.
- **`--contact_normal_track` (move `xref[t]` to ride the penetration).** Fixed the
  *conditioning* — stage-1 projection converges in **5 iterations instead of 300** — but
  did **not** drop the reaction. `ContactModel1D.__init__` exposes only `xref` (a scalar
  *position* for Baumgarte); there is **no velocity/acceleration reference**. So moving
  the frame shifts only the position target; the contact still drives normal velocity → 0
  and, at the constraint level, `a_normal = 0`, both in **absolute** terms. It keeps
  fighting the absolute 0.13 m/s even though the frame moved.
- **`--contact_normal_slack` (soften the Baumgarte gains).** At `slack=0.99` (gains ≈ 0)
  the reaction is *still* 550–950 N. So the residual is the **acceleration-level
  constraint** (`a_normal = 0`), not the gains — nothing on the rigid 1D contact can make
  it accept the recorded into-surface motion.

## 4. Why a compliant spring is the WRONG fix

The obvious "make the contact compliant" idea — normal force = `k · penetration` (a
penalty spring) — **does not match the mechanism we want**. Our model of pressing is:

> the stick holds essentially the **same position** on the rock; the human presses
> *harder* by applying **more joint torque**, which the contact Jacobian transmits into
> **more normal force** — at the same position.

A spring couples force to **penetration**: to press harder the stick must sink deeper, so
force becomes a function of *position*, not torque. That is the opposite of what we want
and would make "effort" show up as displacement instead of joint load. Rejected.

## 5. Reconciliation with the transmission result

This is *not* a regression of the transmission work (`press_in_actuation`,
`R_surface.row(2)` axis, ratio ≈ 1.0). That result established the **force-generation**
channel: a commanded joint torque produces the intended normal force through the exact
1-D contact Jacobian. It is correct and still holds.

The dynamics gap is a **separate, purely kinematic** problem: the recorded *position*
penetrates the rigid contact. Two orthogonal axes:

| axis | question | status |
|------|----------|--------|
| **force generation** | can joint torque produce the normal force? | ✅ yes (transmission ≈ 1.0) |
| **contact kinematics** | does the recorded contact point lie on the surface? | ❌ no, 1–3 cm inside |

The rigid contact is in fact the *right* model for "force from torque at fixed position":
the arm pushes into a rigid surface, the surface reacts, net motion is zero, and the
reaction equals the transmitted press. The only defect is that the *recorded* position
sits inside the surface, so the rigid reaction is dominated by penetration-fighting
instead of the intended press.

## 6. Proposed fix: project the demo kinematics onto the surface

Keep the rigid contact and the force-from-torque mechanism; **remove the penetration from
the demo kinematics** before use:

- Per node, adjust `q` (IK on the contact frame, as `project_start_configuration` already
  does for `q0`) so the stick tip lies **on** the rail/surface — a ~1–3 cm correction at
  the contact, within mocap/compliance noise, leaving the rest of the posture ~unchanged.
- Then the rigid `ContactModel1D` is satisfied (no penetration to fight → no 1000 N
  reaction, no dynamics gap), and the normal force comes from **joint torque via
  transmission** (`press_in_actuation`), scaling with effort at the (projected) position —
  exactly the intended model.

This differs from every attempt in §3 because those all kept the penetrating `q` and tried
to make the *contact* absorb it. Here we make the *kinematics* consistent with the rigid
contact instead.

## 7. Open questions

- How far does surface-projection move each joint? (Expect small; verify the arm posture,
  and the biomech features Tau/Eng, barely change.)
- After projection, does the demo close the gap **and** does the transmitted force land
  near the measured ~17.8 N — i.e. is the 1–3 cm the whole story, or is there residual
  model/inertia inconsistency underneath?
- Cycle 10 shows +8.8 mm *lift* as well as −27 mm penetration — projection must handle
  brief separation (or fall back to the contact-phase window there).

## 8. Tooling added during this investigation

In `run_csqp_population_irl.py` (all additive, default-off):
`--project_demo` (feasibility projection), `--replay_npz` (roll out recovered weights),
`--sqp_iter` (was hardcoded 50 in `mo_args`; the init-baseline vs LS-probe mismatch is the
`[WARN != reseed]`; filter-LS naturally stalls ~56 iters), `--effort_limits` (was gated off
and unexposed), `--contact_normal_track` / `--contact_normal_slack` (both shown insufficient
above — kept for the record, not the fix).
