# MPPI force implementations and their flags

Every approach we tried for producing / handling the **normal press force** in the MPPI
inner solver is in the repo, toggled by CLI flags on the driver.

**Where it lives**
- `tests/test_phase2.py` — the driver; all flags below are parsed here (search the flag
  string to find its variable).
- `src/mppi_mjx_kinematic.py` — the MJX rollout: where the press torque, the impedance
  spring, the emergent-contact read, and the task-space / PD control laws are applied.
- `src/final_models/human_mppi.py` — the model builder (`create_pinned_model`: primitive
  rock geom, contact params) and the demo-side torque (`get_tau_from_trajectory`: how the
  demonstration's press is reconstructed and, optionally, closed-loop regulated).

Two independent choices combine: **(A) how the challenger produces the normal force**, and
**(B) how the demonstration's force is supplied** so demo and challenger are measured with
the same ruler. Pick one from each group.

---

## A. How the challenger produces the normal press force

These are mutually-exclusive modes (roughly in the order we tried them). Default is the
analytic virtual-target spring.

### 1. Virtual-target impedance spring (analytic, default) — `--virtual-target`
The press is an analytic normal spring toward a target `Δz` below the surface:
`F = Kp·(gap + Δz) − Kd·vₙ`, applied as `Jₙᵀ·F`. Smooth and controllable by construction;
this is the "planned / imposed" force (the impedance model). The press force *feature* is
read from this analytic SDF spring, not from MuJoCo contact.

| flag | meaning | default |
|---|---|---|
| `--virtual-target` | enable the analytic Δz spring | on by default in most recipes |
| `--vt-kp=` | spring stiffness Kp (N/m) | 2500–2600 |
| `--vt-kd=` | damping; `<0` = auto critical damping | -1 (auto) |
| `--vt-dz-demo=` | fixed Δz target depth for the demo | — |
| `--vt-dz-sigma=` | sampling spread on Δz (exploration) | — |
| `--vt-dz-from-force` | per-step `Δz(t)=F_meas(t)/Kp` so the demo force **tracks** the measured profile | off |
| `--symmetric-demo-vt` | demo uses the SAME spring as the challenger (feature symmetry) | recommended on |

### 2. Emergent MuJoCo contact — `--emergent-force`
Honest simulator-produced press: a **sphere** rock collision primitive + real MJX contact +
a sampled press channel; the analytic spring is OFF. The rollout reads `efc_force` (the sim
"sensor"). Convenience switch that sets `--primitive-rock` + `press_control` +
`virtual-target off`. The demo keeps the **measured** load-cell force as its target (replaying
the demo through the sphere gives garbage penetration force), so the IRL drives the emergent
force toward the measured profile.

| flag | meaning | default |
|---|---|---|
| `--emergent-force` | primitive rock + real contact + press channel, spring off | off |
| `--primitive-rock` | swap the rock collision geom mesh→sphere (needed for MJX contact) | off |
| `--rock-radius=` | sphere radius (curvature) | 0.03 |
| `--rock-penetration=` | protrude the surface past the site → baseline press | 0.0 |

> Why the sphere: MJX cannot collide capsule/cylinder ↔ **mesh**, so a mesh rock flickers
> (contact drops out ~half the frames). The sphere is MJX-supported and gives continuous
> contact.

### 3. Confined-motion emergent — `--confine-motion`
Emergent force, but project the sampled motion **tangential** to the surface when grazing, so
exploration slides along the workpiece and only the press channel + MuJoCo contact set the
normal force. Restores the surface confinement that `--virtual-target` has but plain
`--emergent-force` dropped → smoother emergent force while still reading efc (with friction).
Use together with `--emergent-force`.

### 4. Depth impedance — `--depth-impedance`
Normal impedance to a **target penetration depth**: holds the tool at `DEPTH_TARGET` so
contact sustains (no fly-off, no over-press) and MuJoCo gives a steady real force.
Knobs (constants in the driver, edit or add `=` parse if you need them live):
`DEPTH_KN=3000` (N/m normal stiffness), `DEPTH_TARGET=-0.0015` m (into the stick).

### 5. Hybrid force / position — `--force-press` and `--force-control`
Direct force control of the normal press (command `F_des` toward the stick, stable on stiff
contact) + PD-track only the tangential sweep (reference confined tangential). The fix for
tool peel-off.

| flag | meaning | default |
|---|---|---|
| `--force-press` | force-control normal (`PRESS_FORCE_DES` N) + PD tangential | off |
| `--force-control` | task-space force-control variant | off |
| `--fc-kp=`, `--fc-kd=`, `--fc-kv=` | force-control gains | — |
| `--fc-sign=` | normal sign convention (use `-1` in the working config) | — |
| `--fc-kp-close=`, `--fc-sign-close=` | gains while closing the gap (approach) | — |
| `--target-rail-vel=` | sweep speed along the rail | 0.13 |

### 6. Task-space hybrid control — `--task-space-control`
`Jₙᵀ·F_des` (normal force) + rail **velocity-servo** (tangential) + null-space posture.
Truthful sustained scrape on the 9-DOF, demo-free-capable. The rail is velocity-servoed,
**not** position-tracked (position-tracking the rail destabilizes).

| flag | meaning | default |
|---|---|---|
| `--task-space-control` | enable the decomposition | off |
| `--ts-fdes=` | commanded normal press force (N) | 30 |
| `--ts-vrail=` | sweep speed (m/s); 0 = demo pace | 0 |
| `--ts-post-kp=` | null-space posture sampling gain (0 = off); config diversity for IRL | 0 |
| `--ts-post-kd=` | null-space damping | 6 |

### 7. Position-tracking (PD) with a free normal — `--pd-track`
The challenger PD-tracks a sampled position reference (`q0 + U_k`) so `approach` shapes a
kinematic path onto the surface and PD holds it there (no drift-off). Combine with:

| flag | meaning | default |
|---|---|---|
| `--pd-track` | PD-track the sampled position reference | off |
| `--pd-free-normal` | project the normal out of the PD torque so the press channel owns the force | off |
| `--pd-feedforward` | add inertial `M·q̈_ref` (computed torque); demo is ~92% feedforward | off |
| `--pd-gain-scale=` | scale the (stiff) demo-matched PD gains; `<1` softens to stop slam | 1.0 |

### The press channel (used by emergent / PD-free-normal modes) — `--press-control`
A dedicated scalar press effort along `Jₙᵀ`, sampled and learned.

| flag | meaning |
|---|---|
| `--press-control` | enable the dedicated press channel |
| `--press-seed=` | initial press magnitude |
| `--press-sigma=` | sampling spread on the press |
| `--press-max=` | clamp on commanded press |
| `--press-ema=` | EMA smoothing on the applied press |
| `--press-ref=` | reference press for the feature |
| `--press-capacity-seed=` | press-capacity seed weight |

---

## B. How the demonstration's force is supplied

The demo has no recorded torque, so its press must be reconstructed the SAME way the
challenger produces force (feature symmetry), or anchored to the real sensor.

| flag | meaning |
|---|---|
| `--demo-sensor-force` | demo force := measured ATI sensor profile (not the graze penetration read); challenger tracks it (force = boundary condition) |
| `--demo-force-gain=` | scale the demo's force feedforward (`force_gain`, default 0.25) |
| `--consistent-demo` | recorded control open-loop-reproduces the tracked demo trajectory (consistent, byte-recordable) |
| `--contact-aware-tau` | reconstruct demo torque WITH the contact term `Jcᵀ·λ_measured` so Tau/Eng carry the pressing effort |
| `--dump-demo-force` | print the demo force profile and exit (diagnostic) |
| `--replay-forces` | replay the recorded controls and report the resulting force |

**Closed-loop force regulation (demo side, `get_tau_from_trajectory`).** New additive kwargs
(off by default) that read the emergent sensor and PI-trim the commanded press to hold the
target profile (the deployment read-and-regulate loop):
`force_regulate=True, force_reg_kp=0.4, force_reg_ki=0.15` (+ `self._force_reg_meas_ema`).
Not wired to a CLI flag yet; call it directly or add a flag if you want it in a run.

---

## Contact / physics and feature knobs (shared)

| flag | meaning | working value |
|---|---|---|
| `--contact-solref=` | MuJoCo contact stiffness/damping (softer = smoother force) | soft |
| `--force-ema=` | EMA smoothing on the read force | — |
| `--force-max=` | clamp on the force feature | — |
| `--press-emergent` | CSQP `f_n²` parity for the MPPI `press_force` feature | off |
| `--noise-scale=` | sampling noise; note it perturbs the **position reference** in pd-track, not forward travel | 1 |

**Determinism (required for the line search on GPU/MJX):** export
`XLA_FLAGS="--xla_gpu_deterministic_ops=true"` so rollouts are bit-reproducible and a descent
step is distinguishable from sampling noise.

---

## Recommended recipes

Run from the repo root in the `unified_env` conda env.

**Analytic impedance (planned force, the default / most stable):**
```
python tests/test_phase2.py <phase1_weights> \
  --virtual-target --symmetric-demo-vt --vt-kp=2600 --vt-dz-from-force \
  --pd-track --pd-feedforward --demo-sensor-force
```

**Emergent, smooth (read force from MuJoCo, surface-confined):**
```
XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
python tests/test_phase2.py <phase1_weights> \
  --emergent-force --confine-motion --primitive-rock --rock-radius=0.03 \
  --contact-solref="0.02 1" --demo-sensor-force
```

**Demo-free task-space scrape (no demonstration trajectory needed):**
```
python tests/test_phase2.py \
  --task-space-control --ts-fdes=30 --ts-vrail=0.13 --ts-post-kp=... --ts-post-kd=6
```

Quick single rollout to inspect force before a long IRL run: add `--rollout-only`
(and `--p1_init=1` where applicable). Diagnostics: `--dump-demo-force`, `--sample-partition`.
