# Recovered objectives behind the reported results

Every cell of the paper's human table, as the weights the recovery returned, so the numbers can be
reproduced rather than taken on trust. Participants appear only as S1, S2 and S3 throughout, including
inside the `.npz` files.

## The table

Joint RMSE is train / held-out over the eight active arm joints, in degrees; force RMSE is on held-out
cycles, in newtons. A dash means the training deploy was not repeated at the reported budget.

### Down stroke

| | S1 | S2 | S3 |
|---|---|---|---|
| CSQP | 6.88 / 7.28, 4.0 N | 5.76 / 6.55, 7.9 N | 6.64 / 8.38, 24.3 N |
| Sampler-A | 2.11 / 2.74, 5.4 N | 1.32 / 1.40, 3.6 N | 2.77 / 2.95, 12.0 N |
| Sampler-B | – / 3.23, 4.4 N | – / 1.56, 4.5 N | – / 2.24, 8.3 N |

### Up stroke

| | S1 | S2 | S3 |
|---|---|---|---|
| CSQP | 2.67 / 2.62, 9.0 N | 10.49 / 8.02, 5.1 N | 6.63 / 6.23, 36.1 N |
| Sampler-A | 4.83 / 4.87, 8.5 N | 3.64 / 3.69, 1.5 N | 4.01 / 9.35, 26.9 N |
| Sampler-B | 1.87 / 1.28, 2.9 N | 4.03 / 4.17, 4.5 N | – / 2.95, 5.3 N |

### Pooled, one objective for all three subjects, down stroke

| S1 | S2 | S3 |
|---|---|---|
| – / 3.70, 3.1 N | – / 1.84, 2.8 N | – / 2.64, 2.5 N |

## What is in here

`sampler/` holds the schedule-free (Sampler-B) recoveries as `<subject>_<stroke>_<seed>[_v<budget>]_weights.npz`,
with `w_rec` the recovered weight vector, `keys` the feature names in the same order, `scale` the frozen
feature scale it was fitted in, and `th` the log-parameters. Files with a `_v<budget>` suffix are the
validation sweep: the same run deployed at budgets 2 to 12, which is how the reported stopping budget is
chosen. Files without it are the run itself.

`csqp_projected/` holds the constrained reference, one `<subject>_<stroke>_recovery.npz` per cell.

Two things to know before comparing them. The weights live in the scaled feature space of the `scale`
array shipped beside them; multiplying raw features by them is a units error of up to three orders of
magnitude. And a weight is not a preference: read cost contributions, the products `w_i * phi_i`, since
weight placed on a feature whose value is identically zero costs nothing and still shows up as a large
coefficient.

## Reproducing a cell

The three entry points in the repository root README regenerate these from the shipped trajectories. The
sampler cells use the validation protocol described in the paper: five training cycles, three held out,
two for choosing the stopping budget, and the held-out cycles enter no selection decision.

## Caveat on the CSQP cells

The constrained reference is fitted to feasibility-projected demonstrations, because several recordings
are not reproducible by the constrained model, and at a three-iteration budget. Those numbers are a floor
rather than converged estimates, and the S3 down and S2 up rollouts are not dynamically consistent
(gaps 4.1 and 3.1), so their kinematics should not be compared with the other cells.
