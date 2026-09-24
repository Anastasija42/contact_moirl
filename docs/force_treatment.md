# Force treatment: OCP, MPPI, and transfer to a real robot

## Principle

> **Update.** This page's original stance — *force is a boundary condition, never
> recovered* — is now just **one of three force treatments** (the *Imposed* one).
> The current population runs compare **Free / Imposed / Tracked** force; see
> [Method → Force treatments](method#force-treatments-free-imposed-tracked). Under
> the default (*Free*) treatment the normal force **is** recovered, as a free contact
> dual back-solved from the rollout's own torques.

Historically we treated force as a **boundary condition, not a recovered objective**.
The measured 27_02 sensor force is the ground truth (S2 down_long: mean ~26.5 N).
In the **Imposed** treatment we recover **pace + effort** (the biomechanical cost)
and hold the contact force to the measurement (within a 15 % slack band). The
**Free** treatment instead leaves the normal as a free dual and recovers it from the
dynamics; the **Tracked** treatment keeps a soft force objective but drops the
friction-cone capacity term.

The two inner solvers below handle the imposed/read force in different ways.

---

## 1. OCP (analytical) — force is PLANNED / IMPOSED

- **Solver:** CSQP / Crocoddyl with a 1D normal contact (`ContactModel1D`,
  `press_normal_dual`).
- **Force model:** the contact force is an explicit quantity in the optimization — an
  analytic normal spring `F = K·Δ` on the end-effector deflection. This is the same
  visco-elastic soft-contact used by Kleff et al. (*Force Feedback in MPC: A Soft
  Contact Approach*, hal-04572399), `λ = −K·Δp − B·ṗ` — a different parametrization of
  the same model ("basically the same thing").
- **Anchoring:** the press target is the **recorded 27_02 force profile**. The OCP plans
  the trajectory *and* the press so the modeled force realizes the measurement.
- **Role:** the clean, smooth, precise force **model**. Used for the recovery experiment,
  where force is an imposed boundary condition and the recovered cost is pace + effort.

## 2. MPPI (emergent) — force is READ, not planned

- **Solver:** `KinematicMPPI_MJX` with **emergent MuJoCo contact**.
- **Geometry:** primitive rock (sphere) so MJX collides sphere↔capsule. (Mesh↔capsule is
  unsupported in MJX — that was the old flicker.) This gives a smooth, continuous
  emergent contact force across the stroke.
- **How force is obtained:** we do **not** plan a force schedule. The controller commands
  the motion and a press (press channel / impedance target); the normal force **emerges**
  from the contact solver and is **read** from `efc_force` / the force sensor.
- **Role:** the physically-honest, deployment-faithful path. The force is a *read*
  quantity, exactly like a sensor reading — never a feedforward trajectory.

---

## 3. Transfer to a real robot

The control law is the **same** in sim and on hardware. Only the *source of the reaction
force* changes: the contact solver (sim) becomes the physical contact + F/T sensor (robot).

| | OCP (analytical) | MPPI (emergent) |
|---|---|---|
| Force in sim | planned / imposed (spring model, anchored to recording) | emergent from MuJoCo contact, read from `efc_force` |
| On the robot | impedance / soft-contact term run as **MPC** (receding horizon, sensor feedback) — Kleff-style | **read the F/T sensor**, regulate to hold the target |
| What transfers | the planned force becomes an impedance command `F = K·Δx` with chosen stiffness | MuJoCo emergent force ⟷ real sensor reading (1:1 role swap) |

- **OCP → robot:** deploy the same visco-elastic / impedance force term as an MPC
  (10-step plan, 1-step execute) with sensor feedback. The planned force is realized by
  commanding an impedance `F = K·Δx`; the outer loop trims `Δx` against the sensor to hold
  the target. This is exactly Kleff's soft-contact MPC.
- **MPPI → robot:** the emergent MuJoCo force is the **simulator's stand-in for the F/T
  sensor**. On hardware you command the impedance, read the sensor, and regulate — the
  press is *read and regulated*, never planned open-loop, in both sim and deploy.
- **Ground truth stays the recording.** The measured 27_02 force is the target in the
  recovery experiment; MuJoCo is never relied upon to *supply* the magnitude — it is the
  physical stand-in for reality. At deploy time the target comes from the task spec and
  the real sensor closes the loop.

---

## One-line summary

- **OCP:** force is a **planned** boundary condition (analytic spring, anchored to the
  recording) → on the robot, an impedance/soft-contact **MPC**.
- **MPPI:** force is **read** from emergent contact (never planned) → on the robot, the
  same read from the **F/T sensor**, regulated to the target.
- **Recovered in both:** pace + effort. **Imposed/read in both:** the force.
