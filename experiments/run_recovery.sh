#!/usr/bin/env bash
# CSQP MO-IRL cost recovery from the shaving demonstrations.
#
#   ./experiments/run_recovery.sh <stroke> <treatment>
#     stroke     down_long | up_long          (default down_long)
#     treatment  A | B | C                    (default B)
#
# A  free      press recovered as a preference (two-cost form)
# B  imposed   press pinned to the measured profile within +-15%
# C  tracked   press carried as an explicit cost weight, no capacity term
#
# The shared regime lives in the driver's defaults; only what differs between
# strokes and treatments is passed here.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate unified_env

STROKE="${1:-down_long}"
TREAT="${2:-B}"
OUT="runs/recovery_${STROKE}_${TREAT}"
mkdir -p "$OUT"

case "$STROKE" in
  down_long) STROKE_ARGS=(--static_stick --max_iter 30) ;;
  up_long)   STROKE_ARGS=(--demo_smooth_q 7 --demo_edge_hold 4 --max_iter 4) ;;
  *) echo "unknown stroke: $STROKE" >&2; exit 2 ;;
esac

case "$TREAT" in
  A) TREAT_ARGS=(--two_cost_force) ;;
  B) TREAT_ARGS=(--two_cost_force --force_strict_slack_frac 0.15) ;;
  # Tracking needs the longer budget, so it re-states --max_iter after the
  # stroke's value; argparse keeps the last one.
  C) TREAT_ARGS=(--no_capacity --max_iter 30) ;;
  *) echo "unknown treatment: $TREAT" >&2; exit 2 ;;
esac

run() {  # name subjects fmax
  local name="$1" subj="$2" fmax="$3"
  echo "=== [$name] $STROKE/$TREAT fmax=${fmax}N $(date +%H:%M) ==="
  python -u src/run_csqp_population_irl.py \
    --subjects "$subj" --task "$STROKE" \
    "${STROKE_ARGS[@]}" "${TREAT_ARGS[@]}" \
    --force_max "$fmax" --target_force "$fmax" \
    --outdir "$OUT/$name" > "$OUT/$name.log" 2>&1

  # Re-solve from a cold IK warmstart under the recovered weights. Writes
  # rollout_recovered.npz beside them, which is what the comparison plots read.
  python -u src/run_csqp_population_irl.py \
    --subjects "$subj" --task "$STROKE" \
    "${STROKE_ARGS[@]}" "${TREAT_ARGS[@]}" \
    --force_max "$fmax" --target_force "$fmax" \
    --replay_cold --replay_npz "$OUT/$name/population_recovery.npz" \
    --outdir "$OUT/$name/replay" >> "$OUT/$name.log" 2>&1
}

# Per-subject capacities (N), then the shared pooled cost at the ceiling.
run s1  S1       14
run s2  S2       28
run s3  S3       47
run pop S2,S3,S1 47

echo "=== done $STROKE/$TREAT -> $OUT $(date +%H:%M) ==="
