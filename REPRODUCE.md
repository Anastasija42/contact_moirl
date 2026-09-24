# Reproducing the results

Every command below is run from the repository root with the `unified_env` environment
active. Set `REPO_ROOT` to the checkout if you run the transfer scripts from elsewhere.

```bash
conda env create -f envs/unified_env.yml     # first time only
conda activate unified_env
export REPO_ROOT=$(pwd)
```

`friction_lib` is a C++ contact model and must be built before anything else runs.
Point `FRICTION_LIB_BUILD` at the build directory if it is not `friction_lib/build`.

---

## Part 1: recovering the cost from the demonstrations

Two challenger generators are compared. Both recover a time-varying weight field `W(t)`
over the same feature set from the same demonstrations.

### The sampler

```bash
cd analysis/contact_point
bash scripts/chain_real_A.sh                 # subject S2
bash scripts/chain_real_A_S13_os.sh          # subjects S1 and S3
NOWAIT=1 bash scripts/chain_real_A_pooled_os.sh   # pooled across subjects
```

Configured by environment, not arguments. `scripts/env_ctrlA.sh` sets the controller and
`scripts/env_sigmaA.sh` the frozen feature scale. `PY_CPU` and `PY_GPU` must be set.
Prefer the `_os` variants: the plain pooled chain applies one subject's feature scale to
everyone.

### The gradient-based reference (CSQP)

```bash
OUT=csqp_S2/runs_b ONLY=b bash csqp_S2/launch_csqp_S2.sh
```

Flags are pinned in `csqp_S2/flags.sh`. Two corrections matter and are on by default:
`CSQP_FRICTION_DX=1` (the friction actuation returned a zero state derivative, which left
the dynamics Jacobian 41 to 55 percent wrong and the solves unconverged) and the
acceptance gap gate `CSQP_GAP_MAX`.

---

## Part 2: transferring the cost across body plans

The recovered human cost is held fixed and re-solved on each of seven body plans. The
contact force is **not** prescribed: it is carried by the recovered press preferences,
evaluated against a per-body capacity scaled from arm mass, and emerges from the solve.

```bash
# all seven bodies, one stroke
bash scripts/transfer/run_set_A.sh down_long  out/transfer_down  data/weights/recovered_down_long.npz
bash scripts/transfer/run_set_A.sh up_long    out/transfer_up    data/weights/recovered_up_long.npz
```

`run_set_A.sh <stroke> <outdir> <weights>` runs `human`, `chimp`, `bonobo`,
`australopithecus_prometheus`, `australopithecus_sediba`, `homo_naledi`,
`homo_neanderthal` in sequence, 200 iterations each, writing
`<outdir>/<body>/<body>__<stroke>/forward.npz` plus a per-body `run.log`. It passes
`--force_strict_slack_frac 0` so no force band is imposed, and sets the capacity
reference per stroke (28 N down, 47 N up for the geometry subject, other bodies scaled by
`(arm mass / human arm mass)^(2/3)`).

A single body:

```bash
bash scripts/transfer/run_one_fmax.sh down_long chimp out/transfer_down data/weights/recovered_down_long.npz
```

### The neutral-cost control

Re-solves every body under a cost that weights all features equally, at the mean weight
magnitude of the recovered vector so the overall scale is comparable:

```bash
bash scripts/transfer/run_set_A.sh down_long out/transfer_neutral data/weights/neutral_down_long.npz
```

Read this control qualitatively. The neutral solves do not converge to the same tolerance
as the recovered ones, so the size of the difference partly reflects the solve.

---

## Part 3: figures

Run from the repository root against a transfer output directory. The plotting scripts
expect a flat layout, one directory per body:

```bash
mkdir -p out/flat/down_long
for sp in human chimp bonobo australopithecus_prometheus australopithecus_sediba \
          homo_naledi homo_neanderthal; do
  ln -sfn $PWD/out/transfer_down/$sp/${sp}__down_long out/flat/down_long/${sp}__down_long
done
```

| Figure | Command |
|---|---|
| Per-joint divergence, cost re-partition | `python morphologies_study/plot_paper_figures.py --indir out/flat/down_long --task down_long --outdir figures --figs R2 R4` |
| Joint trajectories | `python morphologies_study/plot_joint_trajectories.py --indir out/flat/down_long --task down_long --out figures/R3_joint_traj.png` |
| Joint signature | `python morphologies_study/plot_joint_signature.py --indir out/flat/down_long --task down_long --out figures/R7_joint_signature.png` |
| Species embedding | `python morphologies_study/plot_species_embedding.py --indir out/flat/down_long --tasks down_long --out figures` |
| Recovered cost `W(t)` | `python analysis/special/plot_morpho_weights.py --down data/weights/recovered_down_long.npz --up data/weights/recovered_up_long.npz --out figures/weights_Wt.png` |

`plot_paper_figures.py` no longer draws a KKT-over-cost panel. That quantity is a solver
convergence residual, not a biomechanical one, and it read high exactly where the solve
was hardest rather than where the body was most strained.

---

## What the transfer produces

Each `forward.npz` holds the solved trajectory `xs`, the controls `us`, the contact force
`f_contact`, the feature values `phi`, the realised cost contributions `cost_contrib`,
and the solver's `kkt` residual.

Divergence between two bodies is the per-joint RMSE over time on the eight arm joints
(index 1 through 8, excluding the thoracic degree of freedom which tracks the
demonstration), combined as a quadratic mean over joints.

Realised cost contribution is `|w . phi|` per feature. Report contributions rather than
weight shares: the features are spread-normalised, so a large weight on a seeded task
term can buy almost no realised cost.
