---
layout: default
title: Code
---

# Code map

Everything on this site is reproducible from the
**`tool_handling` repository**.
This page maps the repo onto the methods and experiments described in the rest of
the site, so you can jump from a result to the code that produced it.

The library lives in `src/`
(installed editable, imported as top-level modules).
`experiments/`
holds standalone drivers,
`results/` the
figure producers,
`morphologies_study/`
the cross-species application,
`papers/` the two
LaTeX papers, and
`docs/` this site.

## Getting started

Run from the repo root inside the `unified_env` conda environment (Python 3.10):

```bash
pip install -e .
python tests/test_models_load.py        # smoke-test the models load
python src/run_moirl_batch.py           # primary IRL batch driver
```

---

## 1. Core IRL / MO-IRL machinery

The outer loop, feature library, and weight optimization → [Method](method).

| File | Role |
|------|------|
| `src/MO_IRL.py` | Main multi-objective IRL solver; windowed + Gaussian-basis time-varying weights $W(t)$, exact or MPPI gradients, multi-subject models |
| `src/IRL.py` | Base IRL solver, log-likelihood gradient, `make_gradient_mask` |
| `src/IRL_utils.py` | `LogLikelihood` and IRL helper objects |
| `src/irl_utils_setup.py` | IRL setup helpers incl. `load_smoothed_force_profile` |
| `src/Optimization_utils.py` | `Optimizer` wrapper (line search, L-BFGS, step schedules) |
| `src/cost_features.py` | Crocoddyl residual models defining the feature library |
| `src/moirl_config.py` | Combo + method definitions consumed by the batch driver |
| `src/irl_phases.py` | Clean two-phase CPU-MPPI IRL (task then style weights) |
| `src/irl_viz.py` | Weight-evolution / feature-signal / convergence plots |

## 2. Inner solver A — Crocoddyl / CSQP optimal control

The analytical contact-aware OCP → [Method — Crocoddyl forward](method#1-crocoddyl-forward).

| File | Role |
|------|------|
| `src/final_models/human_crocoddyl.py` | `HumanCrocoddyl`: production 9-DOF contact-aware CSQP solver |
| `src/final_models/human_base.py` | `HumanBase`: shared logic for the model classes (incl. `get_traj_features`) |
| `src/model_ocp.py` | `HumanOCP`: two-phase OCP (contact + free follow-through) |
| `src/model_stick_forces_1D.py` and siblings | Crocoddyl model builders (arm + stick + rock contact, force residuals) |
| `friction_lib/` | Compiled `ActuationModelFriction` ($\tau = u + J_c^\top f_\text{fric}$) |

## 3. Inner solver B — MPPI (CPU + MJX / MuJoCo)

The sampling solver with native contact → [Method — MuJoCo + MPPI](method#2-mujoco--mppi-physics-simulation).

| File | Role |
|------|------|
| `src/mppi_cpu.py` | CPU-only **deterministic** `KinematicMPPI`; canonical `KEYS_RUN` feature ordering |
| `src/mppi_mjx_kinematic.py` | `KinematicMPPI_MJX`: MJX/JAX port, numerically matched to the CPU version |
| `src/MPPI_MJX.py` | `MPPIMJXController`: GPU MPPI via MuJoCo MJX (full dynamics) |
| `src/mppi_twophase.py` | Two-phase MJX MPPI (Jacobian-shaped noise + commanded rail velocity) |
| `src/final_models/human_mppi.py` | `HumanMPPI`: MPPI wrapper (auto-selects MJX when available) |

## 4. Human arm model: scaling, pinned models, anthropometry, IK

Mocap → model pipeline → [IK & Marker Registration](ik_analysis).

| File / dir | Role |
|------|------|
| `src/ik.py` | QP-based inverse kinematics from mocap markers |
| `src/utils_get_trajectories.py` | Load/scale subject models, extract joint trajectories |
| `src/utils_slice_trajectories.py` | Slice motion into per-cycle trajectories |
| `human_model/` | Scaled per-subject + species URDFs, MuJoCo XMLs, tool + limb meshes |
| `config/` | Marker maps, `subject_anthropometry.json`, per-date scaled URDF/XML model sets |
| `experiments/run_ik_trajectories.py` · `batch_run_ik.py` | Single / all-subject IK jobs |
| `experiments/calibrate_scale_models.py` · `calibrate_from_stance.py` | Subject calibration |

## 5. Cross-morphology / species study

The morphology paper's experiments → [Cross-Morphology](morphology). All under
`morphologies_study/`:

| File | Role |
|------|------|
| `generate_species_urdf.py` | Generate per-species URDFs from the human base + `PARAMS` table |
| `run_species_forward.py` | **Experiment A** — hold human cost $w^\star$ fixed, re-solve on each body |
| `run_species_inverse.py` | **Experiment B** — recover a per-taxon hypothesis cost, compare to $w^\star$ |
| `plot_paper_figures.py` | Publication figures (divergence, cost re-partition, reach/force) |
| `render_forward_videos.py` · `view_species_meshcat.py` | Videos / animations (source of the [Gallery](gallery) clips) |

Generated species assets:
`human_model/urdf/generated/`.

## 6. Toy box-slide IRL testbed

The known-ground-truth validation problem → [Identifiability — box-slide](identifiability#box-slide-toy-problem).

| File | Role |
|------|------|
| `src/toy_box_slide_irl.py` | Minimal MPPI MO-IRL testbed (push + press DOFs, known $w^\star$, two basins) |
| `src/toy_box_slide_csqp.py` | CSQP/Crocoddyl counterpart (exact-gradient IRL) |
| `src/toy_feature_analysis.py` | Toy feature-sensitivity / identifiability analysis |
| `experiments/run_toy_ablation.py` · `run_all.py` | Box-slide ablation drivers (the `--no-press` ablation lives here) |

## 7. Experiment entry points

Drivers a newcomer would actually run (most useful first):

| Script | What it does |
|--------|--------------|
| `src/run_moirl_batch.py` | **Primary batch driver** — all combos × methods → `analysis/moirl/` |
| `src/run_csqp_population_irl.py` | Population IRL: each subject on their own body, one shared $w(t)$ |
| `src/run_csqp_synthetic_irl.py` | Recover a known $w^\star$ on the CSQP human model |
| `src/run_csqp_identifiability.py` | KKT / feature-sensitivity identifiability on the 9-DOF model |
| `experiments/run_mppi_irl.py` | MPPI MO-IRL pipeline for the scraping task (`--use-gpu`) |
| `experiments/run_mppi_twophase_S3.py` | Reproducible all-feature two-phase MPPI IRL on S3 |
| `experiments/estimate_friction_mu.py` · `estimate_contact_phase.py` | Friction-$\mu$ and contact-window estimation |

More ablation/diagnostic drivers live throughout
`experiments/`.

## 8. Analysis, results, figures

`results/` holds the
figure producers (`irl_analysis.py`, `csqp_traintest.py`, `plot_cost_contrib.py`,
`run_csqp_recovery_plots.py`, a `streamlit.py` interactive viewer, …);
`analysis/` holds the
generated outputs (`analysis/moirl/`, `analysis/special/`, `analysis/posterior/`,
`contact_windows.json`, `force_scale.json`);
`tests/` holds smoke /
consistency scripts.

## 9. Papers and docs

| Path | Contents |
|------|----------|
| `papers/methodology_paper.tex` | **Methodology paper** — MO-IRL with force feedback in contact-rich manipulation |
| `papers/morphology_paper.tex` | **Cross-morphology paper** — Paleolithic tool use via IRL |
| `documentation/` | Method notes `01_…`–`08_basis_weights.md`, warm-start blending |
| `docs/` | This GitHub Pages site + its figure-generation scripts |

---

[Home](.) | [Method](method) | [Identifiability](identifiability) | [Cross-Morphology](morphology) | [Results](results) | [Gallery](gallery)
