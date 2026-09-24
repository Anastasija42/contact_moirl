#!/usr/bin/env bash
# One body, one stroke, OSS recipe + friction fix + PER-BODY press capacity (fwd_fmax.py; FMAX_REF = the geometry subject's fitted
# capacity: down S2 28 N, up S3 47 N; other bodies scaled by arm mass^(2/3)), 200 iterations.   run_one_fmax.sh <stroke> <species> <tag> <weights>
set -uo pipefail
cd ${REPO_ROOT:-$(pwd)}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate unified_env
STROKE=$1; SP=$2; TAG=$3; W=$4; OUT=runs_dx/$TAG/$SP; mkdir -p $OUT
case $STROKE in down_long) REF=28;; up_long) REF=47;; esac
PYTHONPATH=${REPO_ROOT:-$(pwd)}/src OMP_NUM_THREADS=2 FMAX_REF=$REF nice -n 5 python -u runs_dx/fwd_fmax.py \
  --task $STROKE --species $SP --max_iter 200 --weights "$W" --outdir $OUT > $OUT/run.log 2>&1
echo "rc=$? $(date +%T)" >> $OUT/run.log
