# Contact-rich MO-IRL

Minimal-Observation Inverse Reinforcement Learning for a stone-tool shaving task:
recover the motor cost behind a sustained-contact demonstration, then carry that
cost across hominin upper-limb morphologies.

**Recovery.** From a few recorded shaving cycles, recover the weights of a
biomechanical cost (effort, smoothness, pace, tool attitude) with two independent
inner solvers — a constrained optimal-control solver (CSQP / Crocoddyl) and a
sampling MPPI harness (MuJoCo). The contact force is handled as a boundary
condition on the solver, not as a free variable the cost has to explain.

**Transfer.** Hold the recovered human cost fixed and re-solve the same task on
seven upper limbs — modern human, Neanderthal, *H. naledi*, *A. sediba*,
*A. prometheus*, chimpanzee and bonobo — so that any difference in behaviour is
attributable to morphology alone.

The three human demonstrators are anonymized as **S1 / S2 / S3**, ordered by how
hard they pressed (14 N / 28 N / 47 N).

## Setup

Three steps, once. Python 3.10 in a conda environment called `unified_env`:

```bash
conda env create -f environment.yml
conda activate unified_env
pip install -e .          # the modules under src/ import as top-level names
```

The friction model is a small C++ extension, compiled locally rather than
shipped. It is needed by the CSQP recovery and by the transfer (the MPPI harness
does not use it). Point CMake at the environment's interpreter — otherwise it can
pick up a system Python and fail to find headers:

```bash
cd friction_lib
cmake -S . -B build -DPython3_EXECUTABLE="$(which python)"
cmake --build build
cd ..
```

`environment.yml` is a full export of the environment the results were produced
in (Linux x86-64). If it does not solve on your platform, the packages that matter
are `crocoddyl`, `pinocchio`, `eigenpy`, `mujoco`, `numpy`, `scipy` and
`matplotlib`.

Run everything from the repository root: data and model paths resolve from there.

## Reproducing the results

Each entry point carries the published configuration as its own defaults. The
commands below only take what genuinely varies between runs — the stroke, the
force treatment, the subject — and write their outputs under `runs/`.

### 1. Cost recovery with the analytical solver

```bash
./experiments/run_recovery.sh down_long B    # stroke: down_long | up_long
./experiments/run_recovery.sh up_long   B    # treatment: A free | B imposed | C tracked
```

The treatments are the three ways the contact force enters: **A** lets the press
emerge as a preference, **B** pins it to the measured profile (±15 %), **C**
tracks it as an explicit cost term. Each run recovers S1, S2 and S3 individually
and then the shared pooled cost, into `runs/recovery_<stroke>_<treatment>/`. To
look at one:

```bash
python experiments/plot_gradient_rundown.py <run>/population_recovery.npz out.png "title"
python experiments/plot_recovery_vs_demo.py  <run>/rollout_recovered.npz out_prefix "title"
```

### 2. Cost recovery with the sampling solver

The MPPI impedance harness: an operational-space rollout with MO-IRL on the
reduced 9-DOF arm, in NumPy and MuJoCo, with no compile step.

```bash
./experiments/run_harness.sh s2              # subject: s1 | s2 | s3
```

The calibrated press configuration is the harness's own default. Its knobs are
environment variables, so any one of them can be overridden in place, e.g.
`PEN_KP=8000 ./experiments/run_harness.sh s2`. A short sanity run is
`STEPS=40 IRL_ITERS=1 python experiments/mppi_impedance_harness.py`.

### 3. Cross-morphology transfer

```bash
./morphologies_study/run_transfer.sh down_long
./morphologies_study/run_transfer.sh up_long
```

`--task` selects the per-stroke recipe — the recovered weights, geometry subject,
cycle, press target and contact regime — and the driver prints the values it
injects, so a run is always explicit about what it used. The recovered human cost
for each stroke ships in `data/weights/`, so the transfer runs without redoing
the recovery. Budget a few minutes per body.

Figures from a finished run (each takes the run directory, which holds one
`<species>__<stroke>/forward.npz` per body):

```bash
python morphologies_study/plot_paper_figures.py     --indir runs/transfer_down_long --task down_long
python morphologies_study/plot_joint_trajectories.py --indir runs/transfer_down_long --task down_long
python morphologies_study/plot_cost_panels.py       --plain runs/transfer_down_long --task down_long
python morphologies_study/render_species_mesh_video.py --task down_long \
    --indir runs/transfer_down_long --weights data/weights/recovered_down_long.npz --format gif
```

## Layout

| Path | What it holds |
|------|---------------|
| `src/` | The library. Solvers (`MO_IRL`, `IRL`, `mppi_cpu`, `mppi_mjx_kinematic`, `MPPI_MJX`), the contact-aware models (`final_models/`, `prune_arm_model`), the feature library (`cost_features`, `utils_model_residuals`), and the CSQP driver: pipeline (`run_csqp_population_irl`), options (`csqp_cli`), demonstration preparation (`demo_prep`). |
| `experiments/` | Cost recovery: the two run scripts, the MPPI impedance harness and the recovery figure producers. |
| `morphologies_study/` | The cross-morphology study, self-contained: species model generation, the forward transfer, its run script and its figures, including the textured-mesh renderer. |
| `data/` | Inputs the default runs need: the recovered human cost per stroke (`weights/`), per-body start postures, the harness's setup dump and demonstration, contact-window and force-scale configs. |
| `trajectories_from_mocap/` | The input data: IK joint-angle trajectories per subject (`S1`–`S3`), session (`13_02`, `27_02`, `27_02_sensor`) and stroke (`{up,down}_{short,long}`). |
| `config/` | Scaled per-subject models, marker mappings, anthropometry, and the pinned and pruned MuJoCo models. |
| `human_model/` | The base human URDF, meshes and tool geometry, plus the generated species URDFs and meshes. |
| `friction_lib/` | The C++ friction model (`friction_model.cpp`, `CMakeLists.txt`); see Setup. |
| `docs/` | The project site (GitHub Pages), built from this directory. |
