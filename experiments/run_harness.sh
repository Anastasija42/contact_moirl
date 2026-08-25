#!/usr/bin/env bash
# MPPI impedance harness: op-space rollout + MO-IRL on the reduced 9-DOF arm.
#
#   ./experiments/run_harness.sh [subject] [out]
#     subject  s1 | s2 | s3   (default s2)
#
# The calibrated press configuration is the harness's own defaults, so nothing
# is set here beyond the subject and where to write. Override any single knob
# from the environment, e.g.  PEN_KP=8000 ./experiments/run_harness.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate unified_env

SUBJ="${1:-s2}"
OUT="${2:-runs/harness_${SUBJ}}"

SUBJECT="$SUBJ" WRITE_ANIM="${OUT}.npz" \
  python -u experiments/mppi_impedance_harness.py > "${OUT}.log" 2>&1

echo "=== harness $SUBJ -> ${OUT}.npz ==="
grep -oE 'deployed q_norm vs demo = [0-9.]+' "${OUT}.log" | tail -1 || tail -3 "${OUT}.log"
