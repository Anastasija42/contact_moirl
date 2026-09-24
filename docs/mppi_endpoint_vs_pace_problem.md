# The endpoint-vs-pace problem in MPPI-IRL (and why the moving horizon fights it)

**Status:** open methodological problem. Written 2026-07-03.
Companion to `docs/mppi_force_recovery_and_scaling.md` and the CSQP methodology notes.

---

## 1. What we want

We are recovering a motor cost from the shaving demo. For that recovery to be
honest, the completion of the task and the *pace* of the motion must be treated
differently:

- **Endpoint = task requirement → IMPOSE it.** The stroke must reach the end of
  the rail. This is not a preference; it's the task. We are allowed to hand the
  model "you must finish."
- **Pace = preference → RECOVER it.** *How fast* the hand moves along the way,
  and how that speed is distributed over the stroke, is part of the cost we are
  trying to infer. We must **not** tell the model the pace — it has to fall out
  of the recovered weights. Prescribing the pace is cheating: the recovered cost
  would then just be echoing an input we supplied.

So the target formulation is: **endpoint hard-imposed, interior trajectory
(including pace) free and cost-shaped.** Exactly what CSQP already does.

---

## 2. How CSQP gets this for free

CSQP is a **full-horizon** trajectory optimizer: it sees the entire stroke
`t = 0 … T` in one problem. It imposes the endpoint as a **hard terminal
constraint** (`Pxf`/`Vxf` on `x(T)`), and leaves the whole interior trajectory
free, shaped only by the running cost `∑ₜ w·φ(t)`.

Consequence: the **pace is genuinely recovered.** Whatever velocity profile
minimizes the running cost *subject to arriving at the endpoint at T* is what the
solver produces. Change the weights → the pace changes. So the IRL can identify
the pace-related weights from the demo, because in the forward problem the pace
is a free consequence of those weights, not an input.

The key enabler is **global visibility**: the terminal constraint at `T` is in
scope at every interior step, because the optimizer holds the whole horizon at
once.

---

## 3. Why receding-horizon MPPI cannot do the same

MPPI as we run it is **receding-horizon**. At each execution step `t` it plans
only `H` steps ahead (`H = 45` or `30`, with `T = 66`), samples `K` control
sequences over that window, weights them by `exp(−cost/λ)`, executes the first
step, then shifts the window forward and re-plans.

The cost it optimizes over the window covers absolute times `t … t+H`. A cost
defined **only at the task endpoint `T`** therefore enters the planning problem
**only when `t + H ≥ T`**, i.e. only during the last `H` steps. For every step
`t < T − H` the planner **literally cannot see the goal**:

```
  H = 45 :  goal T=66 invisible for t < 21   (first 32% of the stroke)
  H = 30 :  goal T=66 invisible for t < 36   (first 55% of the stroke)
```

During that blind window there is no gradient pulling toward completion. So a
pure "impose only the endpoint" formulation leaves the myopic planner with **no
reason to make progress** for the first third-to-half of the stroke. It stalls.

This is not a tuning issue — it is structural. "Endpoint-imposed, pace-free" is a
**global** property of the trajectory (it couples `t=0` behavior to a constraint
at `t=T`). Receding-horizon control is **local by construction**. The moving
horizon and the property we want are fundamentally in tension.

---

## 4. The two surrogates we tried, and why each violates the goal

Because the endpoint alone is invisible, people add a **per-step** progress
signal so the planner always has something local to chase. There are two, and
**both break the recovery**:

1. **Greedy distance reward** `traveled = −dist/rail_len` (what is currently
   active). Rewards being-further-along at every step.
   - *Ill-defined:* its pull depends on `H` (the sampler only integrates reward
     over the visible window), so the same weight **sprints at H=45 and stalls at
     H=30** — a cost whose meaning changes with a solver hyperparameter.
   - *Contaminates pace:* it directly rewards *going fast/far early*, which is a
     pace bias we are injecting. The pace is no longer free.
   - Empirically: H=45 → sprints and tears contact at ~t30; H=30 → gentle but
     only 58% travel. Neither is the demo.

2. **Schedule tracking** `(s(t) − s_ref(t))²` toward the demo's own progress
   profile.
   - *Well-defined and horizon-robust* (local waypoint always in view), and it
     does impose finishing (the schedule ends at full travel).
   - **But it prescribes the pace outright** — `s_ref(t)` *is* the demo's timing.
     Tracking it means we hand the model the exact velocity profile and then
     "recover" it. That is the cheating we are trying to avoid.

So: greedy = ill-defined *and* pace-contaminating; tracking = well-defined but
pace-prescribing. Neither delivers "endpoint imposed, pace recovered."

---

## 5. The core statement of the problem

> **Recovering the pace requires the endpoint to be imposed *globally* (a
> terminal constraint visible from every interior step, with the interior free).
> Receding-horizon MPPI only ever sees a local `H`-step window, so it cannot make
> a purely-terminal endpoint constraint visible early enough to drive the stroke
> without also supplying a per-step progress surrogate — and every such surrogate
> either is horizon-dependent (greedy) or prescribes the very pace we want to
> recover (tracking).**

The moving horizon buys speed and on-line realism, but it is precisely what
prevents the CSQP-style "endpoint-only, pace-free" recovery.

---

## 6. Options

Ranked by how well they preserve honest pace recovery.

### (A) Make MPPI full-horizon (`H = T`, single shot) — recommended for the recovery experiment
Set the planning horizon to the whole stroke and plan **once** (no receding).
Then MPPI is a **sampling-based full-trajectory optimizer**: the terminal
endpoint cost/constraint at `T` is visible at every step, the interior is free,
and **pace is recovered exactly as in CSQP**. No `traveled` feature at all — just
a terminal endpoint term + the running cost we're inferring.
- *Pro:* matches CSQP structure; pace genuinely free; removes the ill-defined
  `traveled` feature and the horizon-dependence entirely; strengthens the
  CSQP↔MPPI cross-check instead of confounding it.
- *Con:* loses the speed/receding benefit; sampling over a full `T=66` horizon is
  harder (higher-variance rollouts, may need more samples / better proposals);
  we give up the "on-line receding controller" framing.
- *Note:* this directly answers the earlier "why moving horizon at all, why not
  optimize the whole thing?" — for pace recovery you **must** optimize the whole
  thing.

### (B) Keep receding horizon, accept pace is not cleanly recoverable
Run receding-horizon with a per-step surrogate, and **do not claim to recover the
pace**. Report pace as a task-side/prescribed quantity and only recover the
effort/style backbone. Honest, but concedes the pace question.

### (C) Middle grounds (partial, likely not enough)
- **Large `H` (close to `T`) + soft terminal endpoint cost:** as `H → T` the goal
  is visible for most of the stroke and the surrogate can be weakened. Reduces but
  does not remove the contamination; really just approaches option (A).
- **Phase-indexed / time-to-go terminal:** weight a terminal endpoint term by
  remaining time so it "pulls" earlier. Still needs the goal inside the window to
  have any gradient — same visibility wall.

---

## 7. Recommendation / open question

For the **pace-recovery** claim in the paper, option **(A) full-horizon MPPI** is
the only formulation that matches CSQP's "endpoint imposed, pace free" and keeps
the recovery honest. The `traveled` feature — greedy *or* tracking — should be
dropped from that experiment and replaced by a **terminal endpoint term only**.

The moving horizon should be reserved for a *separate* claim (an on-line,
myopic controller), where we explicitly accept that pace is shaped by the horizon
and is not being recovered.

**Open items before committing:**
1. Confirm sampling MPPI is stable/identifiable at `H = T = 66` (variance, sample
   count, proposal quality) — the practical cost of option (A).
2. Decide whether the paper's MPPI story is "CSQP-parity recovery" (needs A) or
   "receding-horizon controller" (accepts B) — they need different formulations
   and should not be conflated in the `W(t)` comparison.
