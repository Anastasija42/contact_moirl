# Why the contact-aware CSQP solve is slow (and what actually fixes it)

**TL;DR** — Adding the pressing force to the joint torque did **not** make computing `τ`
slow (that's a one-line add). It made `τ` **state-dependent**, which turned a *free*
(zero) cost-derivative into a real one — and that derivative is currently **brute-forced by
finite difference** (~30 recomputations per node), which the SQP solver pays for millions of
times per run. The `τ` *value* is cheap; the `τ` *Jacobian* is the whole cost.

> **Status — RESOLVED (both phases).** All three contact-aware residuals now use
> analytic Jacobians. **Phase 1** (Tau/Eng, §8) and **Phase 2** (JTC, §9) together
> cut the single-subject 1-cycle/2-iter solve **159s → 95s (~40%, 1.67×)**, q_norm
> 3.855 → 3.791 (the exact-gradient value). The `friction_lib` C++ module links
> pinocchio directly, so the second-order RNEA that was the Phase-2 blocker
> (`ComputeRNEASecondOrderDerivatives`, absent from the Python bindings) is called
> straight from C++. See §8–§9.

---

## 1. Symptom

A single-subject, 1-cycle, 2-iteration IRL run takes **~159 s** with the contact-aware cost.
It was fast before the press-baking. `cProfile` of one run (`_evaluate_at_alpha` = the line
search / OCP solves) shows **77 % of the time in the solves**, and inside them the
**contact-aware residuals dominate via finite-difference Jacobians**:

| function (`src/utils_model_residuals.py`) | calls | tottime |
|---|---|---|
| `constrained_accel_1d` (JTC's contact fwd-dyn) | 1.1 M | 18.8 s |
| `_resid` (Eng = `q̇·τ`) | 6.5 M | 18.6 s |
| `_r`  (Tau = `τ[idx]`) | 6.5 M | 15.8 s |
| `jtc_id_residual` (RNEA-deriv torque rate) | 1.3 M | 11.4 s |

`eps_abs` is **not** the cause (tested `1e-10` vs `1e-5` → identical time *and* q_norm).

## 2. Root cause: `τ` became state-dependent

CSQP is an SQP method. At **every node, every iteration** it needs `∂(cost)/∂x`, which needs
`∂τ/∂x`. That is the expensive object — not `τ`.

- **Before (kinematic, `τ = u`)**: `τ` depends only on the control, **not the state**, so
  `∂τ/∂q = ∂τ/∂v = 0` — for free. Residual Jacobians were closed-form and instant.
- **Now (contact-aware, `τ = u + Jᶜ(q)ᵀ·f_fric(q,v)`)**: the *value* is still a simple sum,
  but `τ` now depends on the state through the contact Jacobian `Jᶜ(q)` (geometry) and the
  friction `f_fric(q,v)` (velocity/cone direction). So **`∂τ/∂x ≠ 0`**, and it is a genuinely
  hard derivative (the Jacobian's config-derivative + the friction-cone velocity-derivative).

| | `τ` value | `τ` derivative | solve |
|---|---|---|---|
| before | `u` | analytic, **= 0** | fast |
| now | `u + Jᶜᵀf` | zeroed → **finite-diff (~30×/node)** | slow |

## 3. Why it's brute-forced

`friction_lib`'s `ActuationModelFriction::calcDiff` **zeroes** the analytic `dtau_dx`
(`friction_model.cpp` — `data->dtau_dx.setZero()`), because the closed form is non-trivial.
So the contact-aware cost residuals recover `∂τ/∂x` by **numerical differentiation**:
perturb each of the ~30 state/control dims and recompute the residual each time.

- **Tau / Eng** (`ResidualModelGroupTau`, `ResidualModelEnergy`): ~30 `actuation.calc`
  recomputations per `calcDiff`.
- **JTC** (`ResidualModelTorqueChangeContactAware`): **worst case** — each of the ~30
  perturbations re-solves the **contact-constrained forward dynamics** (`pin.forwardDynamics`
  inside `constrained_accel_1d`) plus `computeRNEADerivatives`, because JTC's residual
  `r = dtau_dq·v + dtau_dv·a` needs the constrained acceleration `a` recomputed at every
  perturbed state.

**Scale:** ~30 perturbations × ~66 nodes × ~25 SQP iters × ~12 cold OCP solves (per 2 IRL
iters, `--ls_cold_start` with 6 line-search probes) ≈ **millions of recomputations**. Note
the "2 iterations" is really ~20 000 node-solves.

## 4. What was tried — C++ finite-difference ports

The idea: move the numdiff loop into C++ to strip the Python glue. Added to `friction_lib`:

- **`calcTauJacobian`** — `∂τ/∂x` by finite difference **in C++** (fast arithmetic, no Python
  per-perturbation). Wired into Tau/Eng. **Result: 159 → 119 s (−25 %).**
- **`jtc_contact_calcDiff`** — the whole JTC `_r_fd` numdiff loop (`actuation.calc` +
  `constrained_accel_1d` + `jtc_id_residual`) in C++. Verified **byte-identical (6e-9)** vs the
  Python numdiff. **Result: 159 → 152 s (−5 %).**

### Two hard findings from this

1. **JTC is `forwardDynamics`-bound, not glue-bound.** Its C++ port is byte-identical but only
   saved ~5 %, because each perturbation still runs a full pinocchio contact-dynamics solve —
   identical work in C or Python. Moving the *loop* to C++ can't remove that.
2. **Byte-identical values ≠ identical run.** Even the 6e-9-accurate JTC C++ path drifted
   q_norm `3.855 → 3.821`, and the Tau C++ path drifted it to `4.75`. The CSQP carries `rho/y`
   solver state (see `project_csqp_qnorm_determinism`), making the IRL sensitive enough that
   **any** C++ path nudges which iterate the pareto filter keeps. So there is no *free,
   reproducible* speed-up from the finite-diff ports.

**Ceiling of the finite-diff ports: ~30 % (Eng/Tau ~25 % + JTC ~5 %), with q_norm drift — not 2×.**

## 5. The real fix (not yet done)

Give the solver the **analytic `∂τ/∂x`** instead of finite-differencing it:

- **Tau / Eng** then need **no perturbation loop** — `∂τ/∂x` closed-form, then chain-rule
  (`q̇·∂τ/∂x` for Eng). Requires the friction map's analytic derivative: `∂(Jᶜᵀf_fric)/∂q`
  (Jacobian config-derivative, e.g. via RNEA-with-external-force) + `∂f_fric/∂(q,v)` (the
  friction-cone softening derivative).
- **JTC** then needs **one** `forwardDynamics`, not thirty — using the analytic
  contact-dynamics derivative `∂a/∂x` (`pin.computeConstraintDynamicsDerivatives`, which *is*
  exposed) **plus** the second-order RNEA `∂(dtau_dq)/∂x`. **That second-order RNEA is NOT
  exposed in pinocchio 3.8's Python bindings** — it exists in the C++ library and would need a
  pybind wrapper (the "expose second-order from C++" task).

Only the analytic path removes the ~30× — that is the only route to a real ~2×.

## 6. Current state

- **Baseline restored**: Eng/Tau reverted to Python numdiff → q_norm **3.855** (matches the
  32-cell matrix). JTC is currently wired to the C++ path (5 %, small drift) — **revert to
  numdiff for full byte-identity** if reproducing the matrix.
- **C++ building blocks committed** in `friction_lib` (`calcTauJacobian`,
  `jtc_contact_calcDiff`) as the foundation for the analytic-derivative work.
- **Import-order gotcha fixed**: `friction_lib` registers `ActuationDataFriction` with base
  `crocoddyl::ActuationDataAbstractTpl`, so **crocoddyl must be imported first** (added
  `import crocoddyl` before the `friction_lib` import in `human_crocoddyl.py`; the residual's
  friction_lib handle is now a lazy import).

## 7. Key files

- `src/final_models/human_crocoddyl.py` — model; `contact_aware_cost` / `press_in_effort`.
- `src/utils_model_residuals.py` — contact-aware residuals + their finite-diff `calcDiff`:
  `ResidualModelGroupTau` (`τ[idx]`), `ResidualModelEnergy` (`q̇·τ`),
  `ResidualModelTorqueChangeContactAware` (`dtau_dq·v + dtau_dv·a`), plus `constrained_accel_1d`
  and `jtc_id_residual`.
- `friction_lib/friction_model.cpp` — `ActuationModelFriction` (`calc`: `τ = u + Jᶜᵀf_fric`;
  `calcDiff`: `dtau_dx.setZero()` ← the analytic gap), `calcDiff_analytic` (the fix),
  `calcTauJacobian`, `jtc_contact_calcDiff`.

## 8. Resolution — analytic Tau/Eng (Phase 1 done)

**Implemented.** `ActuationModelFriction::calcDiff_analytic` computes `∂τ/∂x` in closed form:
- `dtau_dv = Jᶜ,linᵀ (Rᵀ D R) Jᶜ,lin` — friction-cone velocity derivative `D = ∂f_surf/∂v_surf`
  chained through the linear frame Jacobian.
- `dtau_dq = Jᶜ,linᵀ (Rᵀ D R) ∂v_loc/∂q  +  (∂Jᶜᵀ/∂q) f_fric` — `∂v_loc/∂q` from
  `getFrameVelocityDerivatives`; the Jacobian-variation term from `computeRNEADerivatives`
  with an external force (two calls, gravity cancels).

**Verified** against `calcTauJacobian` and pinocchio's own manifold FD: max abs error ~3e-5
across the contact window (finite-difference truncation precision). Tau (`r=τ[idx]`) and Eng
(`r=q̇·τ`) now read this analytic `dtau_dx`/`dtau_du` (`utils_model_residuals.py`) instead of
their per-node numdiff loop.

**Speed / recovery.** S2 down_long, 1 cycle, 2 IRL iters: **159s → 107s (~1/3)**, q_norm
3.855→3.752 (the analytic is the *exact* gradient; the numdiff carried eps-truncation error).

**The bug that blocked it.** An earlier attempt read `dtau_dq` 4.5× too large — the analytic
applied a fixed injected normal (`inject_normal_` not gated on `use_press_`), but the recovery
actuation is `use_press_=False`: pure tangential friction with the normal supplied by the free
contact dual, so the calc never injects it. Gating the normal on `use_press_` closed the gap —
and confirmed the normal-force default should be the *actual recovered dual* everywhere
(`press_normal_dual`/`press_friction_dual` now default True).

**Phase 1 speed.** With analytic Tau/Eng and JTC still on the C++ FD port: **159s → 107s**.

## 9. Resolution — analytic JTC (Phase 2 done)

JTC's residual is `r = g(q,v,a)` with `g = dtau_dq·v + dtau_dv·a` (torque rate) and `a` the
1-D **contact-constrained** acceleration, so `dr/dx = dg/dx|_a + dg/da · da/dx_total`. Both hard
pieces are now analytic (`ActuationModelFriction::jtc_analytic_calcDiff`, C++):

- **`da/dx`** via the KKT sensitivity — ONE `forwardDynamics`, no per-perturbation solve:
  - `da/dτ = M⁻¹ − M⁻¹Jᶜᵀ(1/K)JᶜM⁻¹` (constrained inverse inertia; gives `da/du` and the
    friction τ-path `da/dq += da/dτ·∂τ/∂q` from Phase 1).
  - `[M Jᶜᵀ; Jᶜ 0][da/dq; −dλ/dq] = [−dtau_dq + ∂Jᶜᵀ/∂q·λ ; −∂(Jᶜa)/∂q − ∂γ/∂q]`, and the
    `dv` analogue — the RHS uses the **first-order** RNEA `dtau_dq/dtau_dv` (the identity
    `∂M/∂q·a + ∂h/∂q ≡ dtau_dq` collapses the second-order terms here) plus cheap contact-
    geometry finite-diffs.
- **`dg`** via the **second-order RNEA** (`ComputeRNEASecondOrderDerivatives`), contracted with
  `v` and `a` — this removes the ~2·nv `computeRNEADerivatives` of the JTC numdiff.

Verified vs the C++ FD to solver precision (~3e-5). **Result: 107s → 95s** (JTC ~30s → ~18s;
the `forwardDynamics` loop is gone, the residual second-order RNEA + KKT is the new floor).
**Total 159s → 95s (~40%, 1.67×), q_norm 3.791.** The Python reference for the same assembly
is `_jtc_analytic` in `utils_model_residuals.py` (a native-Python version measured *slower*,
111s — the per-node Python overhead exceeds the `forwardDynamics` it saves, which is why the
assembly lives in C++). Key files: `friction_lib/friction_model.cpp`
(`jtc_analytic_calcDiff`, `jtc_dg`), `src/utils_model_residuals.py`
(`ResidualModelTorqueChangeContactAware.calcDiff`).
