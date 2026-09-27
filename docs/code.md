---
layout: default
title: Code
---

# Code

The released repository is
[**Anastasija42/contact_moirl**](https://github.com/Anastasija42/contact_moirl).
Its README covers installation; this page says what is in it and what each
entry point reproduces.

## Three entry points

| Command | What it reproduces |
|---|---|
| `./experiments/run_recovery.sh <stroke> <A/B/C>` | The gradient-based (CSQP) cost recovery for S1, S2, S3 and the pooled cost, under the free / imposed / tracked force treatment. |
| `./experiments/run_harness.sh <s1/s2/s3>` | The sampling recovery: an operational-space rollout with the virtual press impedance, and MO-IRL on the reduced arm. |
| `./morphologies_study/run_transfer.sh <stroke>` | The cross-morphology transfer: the recovered human cost re-solved on the seven bodies, which is what the [viewer](species_inspector.html) shows. |

Each entry point carries the published configuration as its own defaults and
prints the values it injects, so a bare run is the reference run.

## Layout

| Path | Contents |
|---|---|
| `src/MO_IRL.py`, `src/IRL.py`, `src/IRL_utils.py`, `src/Optimization_utils.py` | The MO-IRL outer loop: the contrastive update, the feature mask, the time-varying basis weights, the L-BFGS-B step, the line search and the Pareto acceptance. |
| `src/cost_features.py`, `src/utils_model_residuals.py` | The feature library as Crocoddyl residuals, including the contact-aware effort residuals and their derivatives. |
| `src/final_models/human_crocoddyl.py`, `src/final_models/human_base.py` | The gradient-based inner solver: the constrained OCP with the moving 1-D rail contact, the friction-augmented actuation and the three force treatments. |
| `friction_lib/` | The C++ actuation model $\tau = u + J_c^\top f_{\text{fric}}$ and its Jacobians; built locally. |
| `src/run_csqp_population_irl.py`, `src/csqp_cli.py`, `src/demo_prep.py` | The recovery driver: demonstration preparation, per-subject and population recovery, and replay from the neutral initialisation. |
| `experiments/mppi_impedance_harness.py`, `src/mppi_cpu.py`, `src/prune_arm_model.py` | The sampling inner solver: a deterministic CPU sampler over MuJoCo, the pruned arm model and the virtual impedance press. |
| `experiments/plot_gradient_rundown.py`, `experiments/plot_recovery_vs_demo.py`, `experiments/plot_friction_estimate.py` | The recovery figures, and the normal-force approximation with the effective friction coefficient taken from the force-sensor slices. |
| `morphologies_study/generate_species_urdf.py` | The seven bodies, generated from the human base and the parameter table. |
| `morphologies_study/run_species_forward.py`, `build_start_postures.py` | The transfer: the fixed cost re-solved on each body, with the task placement and the start-posture rules. |
| `morphologies_study/plot_paper_figures.py`, `plot_joint_trajectories.py`, `plot_cost_panels.py`, `plot_species_embedding.py`, `render_species_mesh_video.py` | The morphology figures and the textured-mesh renders. |
| `data/` | The recovered cost per stroke (`weights/`), the start postures, the sampler's setup dump and demonstration, the contact windows and the force scales. |
| `trajectories_from_mocap/` | The joint-angle trajectories fitted to the recordings, per subject, session and stroke, with the per-cycle force-sensor slices. |
| `config/`, `human_model/` | The per-subject scaled models, and the base human model with the generated species URDFs and meshes. |
| `docs/` | This site: this page and the viewer. |

## Not in the release

The motion-capture fitting pipeline, the toy studies and identifiability
ablations, and the exploratory tooling around the transfer (the earlier
sampling pipeline, the meshcat viewers, the species-inverse experiment) are
kept in the working repository. The trajectories the release ships are that
pipeline's output. The recordings themselves are not released, and participants
appear only as S1, S2 and S3.
