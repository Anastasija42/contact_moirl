---
layout: default
title: The Physical Model
---

# The Physical Model

Everything on this site is driven by one physical model of a **seated human
scraping a stone tool along a wooden stick**. This page documents that model
exactly as the code builds it — the body, the dynamics, the contact, and the
force — with the modelling assumptions stated plainly at the end.

There are in fact **two model stacks** sharing one loader (`HumanBase`):

- a **Pinocchio / Crocoddyl** stack — rigid-body dynamics with an analytic
  contact, used by the CSQP optimal-control solver (the population IRL runs);
- a **MuJoCo** stack — full contact simulation, used by the MPPI sampling solver.

They agree on the *body* but differ fundamentally in *contact* and *force*
(see [The two solvers](#the-two-solvers-side-by-side)). Unless noted, the
description below is the **CSQP / Pinocchio** stack, which produces the results
in the [methodology paper](method).

---

## The body: a 9-DOF seated upper limb

The source model is a **whole seated human** — pelvis, both legs, full spine,
both arms, head — authored per subject as a scaled, marker-registered URDF
(Pinocchio) and a matching MuJoCo XML. For the optimal-control problem this is
**reduced to the nine right-side chain joints** that actually drive the scrape;
everything else is welded at its recorded posture.

| # | Joint | Group | Type |
|:-:|-------|-------|------|
| 0 | `middle_thoracic_X` | thoracic (trunk lean) | revolute |
| 1 | `right_clavicle_joint_X` | clavicle | revolute |
| 2 | `right_shoulder_Z` | shoulder | revolute |
| 3 | `right_shoulder_X` | shoulder | revolute |
| 4 | `right_shoulder_Y` | shoulder | revolute |
| 5 | `right_elbow_Z` | elbow | revolute |
| 6 | `right_elbow_Y` | elbow | revolute |
| 7 | `right_wrist_Z` | wrist | revolute |
| 8 | `right_wrist_X` | wrist | revolute |

So $n_q = n_v = n_u = 9$: nine positions, nine velocities, nine joint-torque
inputs. The reduction is a Pinocchio `buildReducedModel` (locked joints welded at
$q_0$); the MuJoCo twin keeps nine `<motor>` actuators and welds every other
joint with an `<equality>` constraint.

- **Seated / pinned base.** The pelvis is fixed in the world — there is no
  floating base in the reduced problem (the free-flyer root is among the welded
  joints, pinned at seat height and rotated 90° about X).
- **The rock is welded to the hand.** The tool is a fixed (0-DOF)
  `rock_fixed_joint` on the hand body, so it inherits the wrist's pose — the
  subject grips it rigidly. Rock mass ≈ 0.25 kg.

### `--lock_thorax` does *not* freeze the trunk

A subtle but important point: locking the thorax does **not** clamp it to a
constant angle. Instead the thoracic DOF is made to **track the recorded trunk
path node-by-node**, within a ±`lock_thorax_tol` (default $10^{-3}$ rad) state
constraint. The trunk lean is a weakly-identified redundant DOF, so we treat it
as a **postural input taken from the data** rather than a joint whose cost we
recover — and it is excluded from every posture-error metric.

### Masses, inertias, and what is subject-specific

Segment masses (kg) come from the authored model — e.g. thorax ≈ 12.0, upper arm
≈ 1.8, forearm ≈ 1.28, hand ≈ 0.45, rock ≈ 0.25 — with per-segment diagonal
inertias.

> **Honest caveat — dynamics are subject-*generic*.** The per-subject URDFs are
> scaled and marker-registered, so **link lengths (kinematics) are
> subject-specific** (e.g. S2's upper arm 0.361 m, forearm 0.250 m). But on
> the CSQP/Pinocchio path the segment **inertias are overwritten from a single
> reference human** unless `keep_urdf_inertias` is set. So the *mass/inertia
> dynamics are shared across subjects; only the geometry differs.* The
> **cross-morphology study is the exception** — it deliberately keeps each
> taxon's own inertias, which is the whole point of that experiment.

**Torque limits** are taken from the model (`ctrlrange`, N·m): thoracic ±190,
clavicle ±100, shoulder ±92/71/52, elbow ±77/15, wrist ±100/100. They are
imposed as hard box constraints only when effort limits are enabled.

---

## Dynamics

- **Rigid-body dynamics** via Pinocchio (RNEA for inverse, ABA / constrained
  forward dynamics for the OCP). State $x=(q,v)\in\mathbb{R}^{18}$; control
  $u\in\mathbb{R}^{9}$ = commanded joint torques.
- **Integration**: symplectic (semi-implicit) **Euler**,
  `IntegratedActionModelEuler`, at timestep $\Delta t \approx 8.3$ ms — the
  control-frame period, set per demo cycle from the mocap clock. The terminal
  node uses $\Delta t = 0$.
- **Gravity**: $g = 9.81\ \mathrm{m/s^2}$, Pinocchio default (the seated base is
  rotated so gravity acts correctly on the upright trunk).
- **No passive joint damping, friction, or springs** — the URDF carries no
  `<dynamics>` tags. Joint **armature** is available but defaults to 0 (when set,
  it is added uniformly to the mass matrix $M(q)$).
- **Solver**: `mim_solvers.SolverCSQP` — a constrained SQP with a filter line
  search (termination tol $10^{-5}$, up to 1000 QP iterations). Constraints: box
  state limits always; optional hard effort / rail / self-collision / strict-force
  bands; and a terminal end-position constraint (±2 cm).

The control $u$ is **not** the muscle-equivalent torque — it passes through the
friction-augmented actuation $\tau_{\text{act}} = u + J_c^\top f_{\text{fric}}$
before the effort features read it (see [Method](method#friction-augmented-actuation-the-central-modeling-device)).

---

## Contact: a moving 1-D rail

The scrape *is* the sliding, so the contact point migrates along the stick over
the stroke. The model captures this as a **rigid one-dimensional holonomic
contact that follows the recorded stick**:

- **`ContactModel1D` ("rail_sliding")** constrains **only the surface-normal
  direction** (the contact frame's $z$-axis). The two in-plane axes are free, so
  the tool **slides along the rail** while its normal penetration is pinned.
- **Geometry.** The "rail" is a straight line from `p_start` to `p_end`,
  PCA-fit to the recorded tool-frame trajectory. The surface frame is
  $\big[-\text{along-stick},\ \text{cross},\ \text{radial normal}\big]$; by
  default the **normal is held fixed** through the stroke and only the sliding
  axis rotates to follow the tangent. (An older "normal-follows-tangent" mode
  that swung the normal ~90° is retained only for reference.)
- **Penetration handling.** The recorded stick penetrates the rigid surface by
  ~1–3 cm (real soft-tissue / tool compliance), which a rigid pin cannot
  represent. Two remedies: `project_kinematics_to_surface` — a minimal-disturbance
  IK that lifts penetrating frames back onto the surface and projects out the
  normal velocity — or `contact_normal_track`, which lets the pin ride the
  recorded offset. The [contact-penetration note](contact_penetration_problem)
  covers this in full.

The MuJoCo/MPPI stack uses a **completely different, soft contact** — a penalty
`<pair>` between a rock sphere and the stick cylinder (radius 2 cm, half-length
45 cm) with an **elliptic** friction cone so the normal force reads in Newtons.

---

## Force: a recovered normal dual and a friction cone

Force is the crux of the task, so the model is careful about where it comes from.

**The normal (press) force is the contact reaction — the `ContactModel1D` dual
$\lambda$** (its normal component). By default (`press_normal_dual`) the solve
does **not** inject a prescribed press; the dual is **free**, driven only by the
`press_force` cost, so the trajectory has to **produce its press through the same
joint torques it moves with**. This is why pressing is identifiable at all.

Two contact features shape it (each a CSQP $\tfrac12\lVert r\rVert^2$ residual):

$$\text{press\_force} = \tfrac12\,(f_n - f_{\text{target}}(t))^2,
\qquad
\text{press\_capacity} = \tfrac12\,(f_n - F_{\max})^2 .$$

`press_force` pulls the normal toward the measured per-step target;
`press_capacity` pulls it toward the body's **capacity** $F_{\max}$ (the largest
press it would exert). Their joint optimum is a *fraction* of capacity,

$$f_n^\star = \frac{w_{\text{cap}}}{w_{\text{press}}+w_{\text{cap}}}\;F_{\max},$$

and because **neither term vanishes there, both weights are identifiable** — that
is the point of carrying two contact features rather than one. The three
[force treatments](method#force-treatments-free-imposed-tracked) differ precisely
in how this force is constrained (Free keeps the dual free; Imposed pins it to
the measurement in a slack band; Tracked drops `press_capacity`).

**Friction cone.** The tangential (scraping) reaction is Coulomb at the cone
boundary,

$$f_{\text{fric}} = -\,\mu\, f_n\, \frac{v_c}{\lVert v_c\rVert},
\qquad \mu = 0.25,$$

opposing the sliding velocity $v_c$, with the **same** $f_n$ the normal produces
(the friction reads the back-solved dual, refreshed after each solve). Friction
enters the joint torque as $J_c^\top f_{\text{fric}}$, so the effort of holding
the press against the slide appears in the recovered cost rather than being
absorbed silently.

- **`press_in_effort`** additionally charges the normal press to the effort
  *feature* — $\tau_{\text{press}} = J_{c,n}^\top f_n$ is added to what Tau/Eng
  read (not to the dynamics) — so a harder press costs proximal muscle torque and
  becomes visible to the IRL, while the dual stays free.
- **Demo force target.** Mocap gives no joint torques, so the demonstrated press
  is reconstructed: by default it is the **emergent** contact force (forward-sim
  the demo torques through MuJoCo and read the rock↔stick reaction /
  `qfrc_constraint`); optionally the mocap force-**sensor** value. On the CSQP
  side the demo `press_force` is the back-solved contact dual, so demo and
  challenger go through identical bookkeeping.

> **Which force numbers are actually used.** The generic driver defaults are
> $\mu=0.3$, target 60 N, $F_{\max}=80$ N — but the **population runs are anchored
> to the measured per-subject forces**: `--avg_force --force_ref_fill` feeds the
> recorded ramping profile as the target, and `--force_max_map S3:47,S2:28,S1:14`
> sets each subject's $F_{\max}$ to their own measured press (the S3/S2/S1 = 47/28/14 N
> force levels). The 60/80 N defaults are *not* what the reported experiments run.

---

## The two solvers, side by side

| | **CSQP** (Crocoddyl / Pinocchio) | **MPPI** (MuJoCo) |
|---|---|---|
| Dynamics | Pinocchio rigid-body (ABA / constrained) | MuJoCo `mj_step` full simulation |
| Contact | rigid **1-D holonomic** pin along the normal | **soft penalty** `<pair>`, elliptic cone |
| Normal force | **free contact dual** $\lambda$ (recovered) | **emergent** `mj_contactForce` (read) |
| Integration | symplectic Euler at $\Delta t$ | native MuJoCo step × ~4 substeps |
| Optimizer | gradient-based CSQP | sampling (softmax over rollouts) |
| Contact schedule | fixed from data (contact window) | discovered natively |
| Feature scale | $\tfrac12\lVert r\rVert^2$ | normalized $\sqrt{\,}$-L2 (O(1) for softmax) |

Because the two use different contact models **and** different feature
conventions, raw feature magnitudes are **not** comparable across solvers — the
code applies per-feature normalization before any cross-solver comparison.

---

## Units and timestep

- **Control timestep** $\Delta t \approx 8.3$ ms (8.6 ms actual per cycle),
  data-driven from the mocap frame rate.
- **MuJoCo simulation timestep** 2 ms → about **4 substeps** per control frame.
  A clock fix was needed here: the rollout
  originally advanced one 2 ms sim step per 8.6 ms control frame, replaying a
  0.57 s scrape ~4× too fast so the tool ran out of stroke at ~66 % travel; the
  substep count $N = \Delta t_{\text{ctrl}} / \Delta t_{\text{sim}}$ fixes it.
- **Units**: torque in **N·m**, force in **N**, angles in **radians internally**
  (reports convert to **degrees**), gravity 9.81 m/s². The CSQP friction
  coefficient is **μ = 0.25** (set in `build_model`); the recorded transverse/axial
  force ratio implies an *empirical* μ ≈ 0.3, so the model uses a slightly
  conservative value.

---

## Modelling assumptions and limitations

Stated plainly, because the physics only means something if its boundaries are
clear:

1. **Friction is sliding-only and reverses over a ~1 mm/s band.**
   $f_{\text{fric}} = -\mu f_n\, v_c/\sqrt{v_c^2+\varepsilon^2}$ ($\varepsilon=10^{-3}$ m/s)
   saturates at $\pm\mu f_n$ — kinetic friction with no stiction — and **flips sign
   within about one control frame at a stroke reversal**, injecting a torque step
   there (the reversal transient the demo-torque pipeline has to clip). A smoother /
   regularised stick–slip law would remove it at the source. *(Future work.)*
2. **No passive joint impedance.** The model carries no passive joint stiffness or
   damping (ligaments, muscle passive tension), so *all* resistance is charged to
   *active* torque — biasing the recovered effort (`Tau`/`Eng`) high, most at
   end-range postures. Representing it faithfully would require a full
   **musculoskeletal (Hill-type) actuator model**, which is out of scope here.
   *(Limitation / future work.)*
3. **Subject dynamics are generic** (shared reference inertias); only link
   lengths are subject-specific on the CSQP path. The morphology study is the
   deliberate exception.
4. **Two contact models.** Rigid 1-D holonomic (CSQP) vs soft penalty (MuJoCo) —
   the same scrape, resolved two ways; they are validated against each other, not
   assumed identical.
5. **Rigid contact vs recorded penetration.** The raw mocap penetrates the rigid
   surface by ~1–3 cm; the projection / normal-track workarounds are required for
   the rigid pin to be consistent with the data.
6. **Force magnitude at coarse timesteps reads low** in the MuJoCo deploy — a
   known artifact of the sim-step / substep tradeoff, which is why the reported
   press is anchored to the measured profile, not read off a coarse rollout.
7. **The rock is a rigid weld** — grip compliance and in-hand tool motion are not
   modelled.

---

[Home](.) | [Method](method) | [Identifiability](identifiability) | [Cross-Morphology](morphology) | [Code](code)
