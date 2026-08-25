#!/usr/bin/env bash
# Transfer the recovered human cost onto the seven hominin upper-limb models.
#
#   ./experiments/run_transfer.sh <stroke> [outdir]
#     stroke  down_long | up_long   (default down_long)
#
# The transfer regime, the per-stroke weight file, start postures, geometry
# subject, cycle and press target are all driver defaults selected by --task,
# so a bare run reproduces the published result for that stroke.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate unified_env

STROKE="${1:-down_long}"
OUT="${2:-runs/transfer_${STROKE}}"
mkdir -p "$OUT"

python -u morphologies_study/run_species_forward.py \
  --task "$STROKE" --outdir "$OUT" > "$OUT/run_all.log" 2>&1

echo "=== transfer $STROKE -> $OUT $(date +%H:%M) ==="
grep -E '^\[wstar\] from' "$OUT/run_all.log" || true
