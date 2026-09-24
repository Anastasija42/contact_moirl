# Keeping the End Effector on the Rail Without Reintroducing a Kinematic Latch

*Design + empirical notes for the MO-IRL stone-tool shaving controller (13_02 rig, MJX MPPI).*

## Context

A robot arm holds a tool (the "rock") and scrapes it along a stick. **MO-IRL** learns the
human's cost weights from a recorded demo: the challenger controller is deployed, its
trajectory features are compared to the demo's, and the feature weights are adjusted (LBFGS)
until the challenger reproduces the demo. This concerns the **MPPI** solver
(`src/mppi_mjx_kinematic.py`, driver `tests/test_phase2.py`).

We want the sampler to explore, per rollout, **pace**, **force**, and **configuration**,
while the tool **stays in contact** with the stick — so the IRL can recover meaningful weights.

## The core idea: **constrain** the end effector, don't **transport** it

The previous **kinematic latch** (`latch_project`) did **both**:

1. It kept the tool on the rail (good — a constraint).
2. It **advanced** the tool along the rail by explicitly updating its position each timestep (bad).

The second property is exactly what breaks MPPI. Since the tangential motion is **prescribed**,
every rollout follows essentially the same path → the trajectory features are nearly identical
→ the optimization landscape flattens (the cost is ~0 for every in-contact sample) → the
sampler has nothing to rank. The **latch is the only thing moving the rock**; the MPPI just
jitters around a prescription it can't influence.

The new controller should **only constrain the tool to remain on the rail**, while allowing the
**motion along the rail to emerge from the sampled control inputs and the system dynamics.**

## Decompose the rail into three directions

At every point on the rail define an orthogonal frame:

- **Tangential** `t̂` — along the rail.
- **Normal** `n̂` — into the stick.
- **Lateral** `l̂` — perpendicular to the rail, within the surface.

Each direction has a different control objective:

| Direction | Objective | Controller |
|---|---|---|
| Tangential | Move along the rail | **Velocity servo** (sampled by MPPI) |
| Normal | Maintain contact | **Impedance** controller |
| Lateral | Stay centered on the rail | **PD / impedance** controller |

### Tangential — MPPI controls this

Rather than prescribing the end-effector position, MPPI samples a desired scraping speed
`v_target`. The controller generates a tangential force

```
F_t = Kv · (v_target − ṡ) · t̂
```

where `ṡ` is the current velocity along the rail and `Kv` the velocity gain. Crucially this
**does not prescribe position** — only a desired speed. The resulting motion depends on the
robot dynamics, friction, contact, and configuration. (A velocity servo, *not* a constant
force: a constant push has no equilibrium and the tool accelerates / flies off — see below.)

### Normal — keep contact

Maintain the desired contact pressure with impedance:

```
F_n = ( Kp·(g + δ) − Kd·v_n ) · n̂
```

`g` = current surface gap, `δ` = desired penetration, `v_n` = normal velocity. This only
ensures the tool stays pressed against the surface (presses harder as the gap opens).

### Lateral — stay on the rail

Let `e_l` be the lateral offset from the rail centerline:

```
F_l = ( −Kp·e_l − Kd·v_l ) · l̂
```

Continuously corrects sideways drift while leaving the tangential direction unrestricted.

### Total task-space force → joint torques

```
F = F_t + F_n + F_l
τ = Jᵀ F
```

(A null-space posture controller can be added later without disturbing the contact task.)

## Why this is fundamentally different from the latch

The latch prescribed the tangential **position**, tangential **velocity**, and arm
**configuration** — the robot was *transported* along a predefined trajectory. Here:

- the **normal** controller keeps the tool pressed against the surface,
- the **lateral** controller keeps it centered on the rail,
- the **tangential** controller only specifies a desired *velocity*.

The robot is **constrained to remain on the rail, but not transported along it.** Motion
emerges from the interaction between the sampled controller, contact dynamics, friction, and
robot kinematics.

### Intuition — a train

The rails prevent sideways motion; gravity keeps the train on the tracks; the engine decides
how fast it moves. **The rails do not transport the train.** Likewise, the lateral and normal
controllers constrain the tool to the rail; the tangential controller decides how fast it goes.

## Why this preserves MPPI and MO-IRL

MPPI should optimize high-level task objectives, not replay a trajectory. Instead of sampling
joint torques or prescribing end-effector positions, MPPI samples

```
u = ( v_target ,  δ ,  q_null )
```

- `v_target` — desired scraping speed (**pace**),
- `δ` — desired contact pressure (**force**),
- `q_null` — arm (or later whole-body) posture (**configuration**).

Because the tangential motion is no longer prescribed, different samples produce genuinely
different trajectories → the feature values (progress velocity, contact force, energy, posture,
…) differ across rollouts → MPPI can rank them and MO-IRL can recover meaningful weights.

---

## Empirical grounding — the slide test (`SLIDETEST=1`)

Before trusting the design, `SLIDETEST` seats the tool lightly (3 mm) and commands **only** the
three-term force law above, stepping the **real dynamics** (no kinematic override). Findings:

**Dynamic sliding works at μ=0.25.** (Friction is already low — the rock–stick pair is patched
to `--mu=0.25`.) The old "a pressed tool won't move" was an artifact of **pressing too deep**
(deep penetration → huge N → high friction) *and* kinematically overriding the motion so the
dynamics never got to push.

| Setting | advance | gap (contact) | drift | note |
|---|---|---|---|---|
| constant push, light press | 16.4 cm | −3 → **+37 mm** | 6.6 mm | slides but **flies off** |
| + closed-loop press | 16.8 cm | −3 → +9 mm | 2.4 mm | less lift-off, still escaping |
| + **velocity servo** | 2.2 cm | −3 → **0 mm** ✓ | 8.8 mm | **pressed + steady**, but slow |
| push hard + hold hard | 5.5 cm | ≈ 0 ✓ | 18.7 mm | faster pressed, but drifts |
| + KP_LAT 800→3000 | 5.5 cm | ≈ 0 ✓ | 12.7 mm | drift = diminishing returns |

So a **fast, pressed, on-line scrape by real forces** is achievable — validating the design.

**Knobs** (env vars, `SLIDETEST=1`): `V_TARGET` · `KV` · `F_MAX` · `KP_N` / `FN_MAX` · `KP_LAT`.

---

## Implementation status — `rail_force` mode (what we built, what broke, what we have now)

The three-term law is now a real control mode, `rail_force`, in **both** the MPPI rollout
(`step_fn`) and the receding-horizon deploy (`solve`), gated by `--rail-force`. When it's on the
kinematic `latch_project` is **disabled** — there is no transport override anywhere. MPPI samples
`u = (v_target, δ, q_null)` through the existing pace / press / config channels
(`--rail-walk-pace`, the press channel, `--rail-walk-post-kp`).

```
τ = Jᵀ( F_t + F_n + F_l ) + null-space posture
F_t = clip(Kv·(v_target − ṡ), ±F_max)              # tangential velocity servo (sampled pace)
F_n = clip(Kpn·(gap+δ) − Kdn·v_n + F_press, 0, cap) # normal impedance (sampled press)
F_l = −Kplat·e_l − Kdlat·v_l                        # lateral hold to the rail line
```

### What the open-loop rollouts showed (good)

Rendered 256 rollouts (`rail_rollouts_off.png`). Versus the old kinematic latch: every rollout
now travels **forward** (0→22 cm, no −40 cm backward transport artifacts), pace **regulates toward
v_rail** (the velocity servo working), force is **bounded** (0–50 N, not 300 N spikes), and — the
whole point — the **cost un-flattened**: a dense low-cost cluster sits right on the demo point
(~9–11 cm, ~11 N). MPPI finally has a landscape to rank. **Caveat:** that plot is *open-loop*
(one converged plan rolled the full horizon, no replanning); the deploy is receding-horizon and
should be tighter still.

### What went wrong in the first closed-loop deploy (the slam/bounce)

First full IRL deploy: the run was healthy (finite torques, 66 steps, IRL line-searching) but the
tool **slammed and bounced** instead of seating:

- The tool **starts ~2.7 cm off** the surface (approach), and the stiff normal spring computed
  `F_n = Kpn·gap = 8000 × 0.027 ≈ 216 N` (saturating `fnmax`) **while still 2.7 cm away** →
  the tool was flung at the stick at **2.3 m/s**, bounced off, separated, slammed again.
- The gap oscillated `+0.02 / −0.005 m` and the contact force spiked **50 ↔ 220 N** instead of
  holding steady. `Kdn=30` against `Kpn=8000` was also far under critical (`≈179`), so contact
  **rang** rather than settling.
- SLIDETEST never exposed this because it **pre-seats** the tool at 3 mm — gap ≈ 0 always, so
  `F_n` stayed moderate. In deploy the tool has to *approach*, and the stiff spring turned the
  approach into an impact.
- Reach still capped at **~33–37 % of the rail (~8 cm)** — the fixed-base wall below.

### The fix we have now (gentle-approach gate)

Position-dependent cap on the normal press, applied identically in rollout and deploy:

```
cap = F_nmax   if gap < rf_near   (1.5 cm — inside the contact band → full impedance holds)
      rf_fapp  otherwise          (25 N — during approach → SEATS, does not slam)
```

Plus `Kdn 30 → 120` (≈critical for `Kpn=8000`) so contact settles instead of ringing. New knobs
`rf_near`, `rf_fapp` (and all `rf_*` gains) are env-overridable via the driver. **Status:** pushed
to `origin/portable`; awaiting the re-run to confirm the gap holds near 0 and the 50↔220 N spikes
are gone. The reach cap is expected to remain — that's the torso, not the controller.

## The remaining wall: the fixed base (the torso is locked)

Every residual — short reach (~7 cm), slow sweep (creeps under force; friction is only ~1.5 N,
so it's the *arm*), and lateral drift (~38 N of Jacobian coupling throwing it off-line) — traces
to **one cause**: the MJX model **locks 27 DOFs** (legs, **spine/torso**, other arm, neck), so
the arm has a **fixed shoulder**.

The human scrapes the full rail by using the torso — held **upright and stable**, but **leaning
/ shifting to give the arm base motion, reach, and leverage**. We locked exactly those DOFs.

## Open question — "PD to the rail after each rollout, like the demo?"

The demo uses a PD-to-rail (marker/IK) path — but the demo is a **replay**; we already know its
trajectory. Doing the same on the **challenger** is **kinematic transport again** → prescribes
the motion → flattens the cost → replays a path instead of learning a behavior. That's why it's
not the answer. The systematic alternative is the **three-term force law above**: constrain to
the rail with forces (normal impedance + lateral hold), let the tangential motion be a
**dynamic, sampled push** — the arm is *pushed*, not *transported*.

## Next experiment

**Unlock the torso** — controlled **upright + stable** (actuated to stay upright, free to lean)
— on the GPU box (no nan there), and re-run the `SLIDETEST` three-term law. Hypothesis: with
base motion the advance jumps, the sweep speeds up, and the lateral drift drops, because the arm
can lean into the push and follow the line like a body does. If so, the dynamic controller +
stabilized-but-mobile torso is the real shaving controller, and the kinematic latch /
`latch_vrail` go away.
