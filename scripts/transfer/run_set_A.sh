#!/usr/bin/env bash
# Treatment-A pilot: band-free transfer (--force_strict_slack_frac 0), per-body capacity from FMAX_REF,
# A pooled weights. run_set_A.sh <stroke> <tag> <weights>
set -uo pipefail
cd ${REPO_ROOT:-$(pwd)}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate unified_env
STROKE=$1; TAG=$2; W=$3
case $STROKE in down_long) REF=28;; up_long) REF=47;; esac
mkdir -p runs_dx/$TAG; : > runs_dx/$TAG/bodies.done
for SP in human chimp bonobo australopithecus_prometheus australopithecus_sediba homo_naledi homo_neanderthal; do
  OUT=runs_dx/$TAG/$SP; mkdir -p $OUT
  PYTHONPATH=${REPO_ROOT:-$(pwd)}/src OMP_NUM_THREADS=2 FMAX_REF=$REF nice -n 5 python -u runs_dx/fwd_fmax.py \
    --task $STROKE --species $SP --max_iter 200 --weights "$W" \
    --force_strict_slack_frac 0 --outdir $OUT > $OUT/run.log 2>&1
  echo "$SP rc=$? $(date +%T)" >> runs_dx/$TAG/bodies.done
done
echo "ALL BODIES DONE $(date +%T)" >> runs_dx/$TAG/bodies.done
