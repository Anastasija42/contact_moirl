# The Press-Force Transmission Problem

*Why the shaving demo shows ~1–6 N of press when the sensor measured 10–40 N, and
why injecting the force into the joint torques only partially reproduces it.*

---

## 1. Setup and notation

A 9-DOF arm holds a stick that presses on a rock. Per timestep:

- $q, v, a$ — joint position, velocity, acceleration.
- $M(q)$ — joint-space mass matrix; $h(q,v)$ — nonlinear terms (Coriolis + gravity).
- contact frame at `contact_frame_id`; $J$ — its geometric Jacobian.
- $f_n$ — the **normal press** the hand applies to the rock (ATI sensor: S1 ≈ 9.9 N,
  S2 ≈ 17.7 N, S3 ≈ 39 N).
- $\lambda$ — the contact reaction force (a Lagrange multiplier / dual variable).
- $u$ — the joint control; $\tau$ — the net actuation torque.

The `press_force` cost / feature is a function of $\lambda$ (the back-solved contact force).

---

## 2. The symptom

Feeding the recorded demo $(q,v,a)$ through the model, the back-solved contact normal
force is **~1–2 N for every subject**, regardless of the 4× spread in the measured force:

| subject | measured $f_n$ | demo back-solved $\lambda$ |
|---|---|---|
| S1 | 9.9 N | 0.39 N |
| S2 | 17.7 N | 1.18 N |
| S3 | 39.0 N | 1.95 N |

So the demo's press feature collapses to "not pressing" for everyone
($\text{press\_force}=(\lambda/F_{\max})^2\approx 0.0001\!-\!0.0015$), and the
cross-subject force difference — the thing we would want to recover — is **invisible**.

---

## 3. Why the force is underdetermined (the physics)

Shaving is a **quasi-static / isometric press**: the hand pushes with $f_n$, the rigid
rock pushes back with $f_n$, and the net motion is ~0. Inverse dynamics is a map
*motion → torque*:
$$\tau_{\text{RNEA}} = M(q)\,a + h(q,v).$$
Any internal contact force $\lambda$ satisfies $M a + h = \tau + J^\top\lambda$ with the
**same** motion, because a no-motion internal force lives in the **null space** of the
motion→force map. Given only kinematics, the model therefore assigns the *minimal*
$\lambda$ (~1 N), never the true press. **The force is not observable from motion; it
must be measured and imposed.** The actuation model $\tau = u + J^\top\lambda$ has
$\lambda$ as a **free variable** — pinned by a *cost* in a rollout, but nothing pins it
for a fixed demo.

---

## 4. The actuation model — and the bug

The stick uses a friction-augmented actuation, `ActuationModelFriction`
(`friction_lib/friction_model.cpp`), intended to be $\tau = u + J^\top f_{\text{fric}}$
with $f_{\text{fric}}$ the contact wrench. The original `calc` set only the **tangential
friction**:
$$
f_{\text{fric}} =
\begin{bmatrix} -\mu f_n\,\dfrac{v_x}{\lVert v_t\rVert} \\[4pt]
                -\mu f_n\,\dfrac{v_y}{\lVert v_t\rVert} \\[4pt]
                \mathbf{0} \end{bmatrix}
\quad\Longleftarrow\ \text{the normal component } f_{\text{fric},z} \text{ was never set.}
$$
So **the normal press $f_n$ was never put into $\tau$** — only its friction consequence
$\mu f_n$ was. That is the concrete reason the demo reads ~1 N: nothing commands the
normal force into the dynamics, so the back-solve returns the motion-minimal reaction.

(The friction *magnitude* $\mu f_n$, $\mu\approx0.3$, is a measured force ratio and is
unaffected; only its **direction** was approximated — applied in the plain LOCAL tangent
rather than the surface tangent plane, i.e. the observed 13–17° off-axis is the surface
tilt.)

---

## 5. The fix, and the KKT math for the reaction

Add the normal press to the actuation, $f_{\text{fric},z} = -f_n$ (along the contact
normal). The contact forward dynamics is the KKT system

$$
\begin{aligned}
M a + h &= \tau + J_c^\top \lambda, \\
J_c\,a + \dot J_c v &= 0 \qquad (\text{contact acceleration} = 0),
\end{aligned}
$$

where $J_c$ is the **contact** Jacobian (for `ContactModel1D`, the 1-D projection of the
frame Jacobian onto the constraint axis). Eliminating $a$:

$$
\boxed{\;\lambda = -\bigl(J_c M^{-1} J_c^\top\bigr)^{-1}\Bigl[J_c M^{-1}(\tau - h) + \dot J_c v\Bigr].\;}
$$

The actuation injects the force through **its own** Jacobian $J_a$
(`computeFrameJacobian(LOCAL)`), i.e. $\tau = u + J_a^\top f$ with $f = f_n\,\hat n$.
Splitting:

$$
\lambda = \underbrace{-\bigl(J_c M^{-1} J_c^\top\bigr)^{-1}\bigl[J_c M^{-1}(u-h)+\dot J_c v\bigr]}_{\lambda_{\text{motion}}\ (\approx\,1\text{ N})}
\;\;\underbrace{-\;\bigl(J_c M^{-1} J_c^\top\bigr)^{-1}\bigl(J_c M^{-1} J_a^\top\bigr)\,f}_{\text{injected part}}.
$$

Write the operational-space inverse inertia $\Lambda^{-1} = J M^{-1} J^\top$. **If the
actuation and contact share the same operator** ($J_a = J_c = \hat n^\top J$, aligned to
the constraint axis $\hat n$), the injected part collapses cleanly:

$$
-\bigl(\hat n^\top\Lambda^{-1}\hat n\bigr)^{-1}\bigl(\hat n^\top\Lambda^{-1}\hat n\bigr)\,f_n
= -f_n .
$$

**The inverse inertia cancels exactly — transmission is 100%, independent of posture.**
A correctly-added, aligned, pure normal force *must* reproduce $\lambda = f_n$.

---

## 6. What we measured — it is **not** 100%

Injecting $f_n$ and reading the reaction (`press_in_actuation`, S2, mid-stroke):

| condition | transmission $\lambda / f_n$ |
|---|---|
| surface-normal direction $R_t$, with friction | 0.42 |
| aligned to contact axis (LOCAL-z), $\mu=0$ (pure normal) | 0.59 |
| aligned, $\mu=0$, **rigid contact** (Baumgarte gains $[0,0]$) | 0.61 |

and it is **linear** but **subject-dependent**:

| subject | $k{=}1$ | $k{=}2$ | $k{=}3$ | slope |
|---|---|---|---|---|
| S1 | 3.4 | 7.2 | 11.0 | **0.39** |
| S2 | 7.5 | 16.2 | 24.9 | **0.49** |
| S3 | 26.9 | 54.2 | 81.5 | **0.70** |

Ruled out, one by one:
- **Direction** — aligning the injection exactly to the contact's reaction axis (LOCAL-z,
  measured reaction direction $[0,0,1]$) changed nothing (0.59, same as the tilted $R_t$).
- **Friction** — $\mu=0$ (pure normal) only moved 0.42 → 0.59.
- **Contact stiffness** — rigid gains $[0,0]$ only moved 0.59 → 0.61.

So even an **ideal** case (aligned, pure, rigid) transmits ~0.61, not 1.0.

---

## 7. Diagnosis: a flipped-transpose (wrong operator), now **FIXED**

By §5 the only way transmission $\ne 1$ for an aligned rigid pure normal is $J_a \ne J_c$.
Reading crocoddyl's `ContactModel1D::calc`, the contact Jacobian is

$$
J_c = \bigl(R_{\text{axis}}\,J_{\text{linear}}^{\text{LOCAL}}\bigr).\text{row}(2),
$$

i.e. the surface frame is $R_{\text{axis}}\cdot(\text{LOCAL})$ and the reaction axis is
$R_{\text{axis}}.\text{row}(2)$. Our actuation injected the normal along
$R_{\text{axis}}.\text{col}(2)$ (i.e. $R_{\text{axis}}^\top$) **and** applied a 6-D force
through `computeFrameJacobian`. **Two transposes were flipped.** Injecting through
crocoddyl's *exact* $J_c$ (apply $\tau \mathrel{+}= J_c^\top f_n$) gives transmission
**≈ 1.0 for every subject** (S3 0.99, S1 1.14, S2 1.15 — the small excess is the
~1 N motion reaction $\lambda_{\text{motion}}$), vs 0.39–0.70 before.

**Fix** (`friction_lib/friction_model.cpp`, 5-arg path): surface transform
$v_{\text{surf}} = R_{\text{surface}}\,v_{\text{local}}$ and map the wrench back with
$R_{\text{surface}}^\top$ (previously the two were swapped). So the normal lands on
$R_{\text{surface}}.\text{row}(2)$ = the contact's actual reaction axis, and the
torque→reaction loop **closes**: the demo presses at $\approx f_n$, subject-independent.

**Corrected conclusion:** the torque *can* set the contact force — the earlier
"fundamental, posture-dependent 0.39–0.70" was our operator being wrong, not physics.

---

## 8. Consequences

1. **`press_in_actuation` now works** (flag-gated). With it on, `contact_aware_demo`
   makes the demo press at the measured $f_n$ *through the dynamics* — the physically
   consistent route. No feature hack needed for the demo to press correctly.
2. **`--demo_force_measured`** (feature-level injection) remains available as a
   dynamics-free alternative, but is no longer the *only* way to get an honest demo.
3. **The press force is still measured/imposed, not recovered from a single demo** (§3:
   the null-space argument is unchanged — the *magnitude* $f_n$ has to come from the
   sensor). What changed is that we can now feed that magnitude in *consistently* at the
   torque level and have $\lambda$ reproduce it.
4. **Recovery of the force** is still only possible as a *cross-subject preference* — a
   fraction of capacity $F^\* = F_{\max}\,w_{\text{cap}}/(w_p + w_{\text{cap}})$ — and
   only with **per-subject capacity $F_{\max}$**. With a shared $F_{\max}$ the "fraction"
   just re-encodes the measured force → equivalent to tracking.

---

## 9. Reproduce

```
# demo back-solved force + press features, all subjects
python scratchpad/probe_scale_subjects.py
# transmission slope per subject
python scratchpad/probe_linearity_subjects.py
# direction / friction / gains isolation (S2)
python scratchpad/probe_alignment.py
```

Code: `friction_lib/friction_model.cpp` (5-arg constructor = press-in-actuation, off by
default), `src/final_models/human_crocoddyl.py` (`press_in_actuation` flag),
`src/MO_IRL.py` (`--demo_force_measured` injection).
