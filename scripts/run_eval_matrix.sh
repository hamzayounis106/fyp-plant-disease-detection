#!/usr/bin/env bash
# Run 4 evals per model: {plant_village, plantdoc} x {raw, segmented}
# Usage: bash scripts/run_eval_matrix.sh
set -e
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
RAW_CSV="artifacts/metadata/unified_metadata_resized.csv"
SEG_CSV="artifacts/metadata/unified_test_segmented.csv"
CMAP="artifacts/metadata/unified_class_map.json"

CKPT_NAME="${1:-best_calibrated.pt}"
for MODEL in cnn efficientnet; do
  CKPT="artifacts/checkpoints/${MODEL}/${CKPT_NAME}"
  if [ ! -f "$CKPT" ]; then echo "skip $MODEL: $CKPT missing"; continue; fi
  echo "=== $MODEL ==="
  for SRC in plant_village plantdoc; do
    echo "--- $SRC raw ---"
    "$PY" evaluate.py --metadata-csv "$RAW_CSV" --image-column image_path_resized \
      --checkpoint "$CKPT" --class-map "$CMAP" --source-filter "$SRC" \
      --output-json "artifacts/checkpoints/${MODEL}/eval_${SRC}_raw.json" \
      --num-workers 0
    echo "--- $SRC segmented ---"
    "$PY" evaluate.py --metadata-csv "artifacts/metadata/unified_test_segmented_ok.csv" --image-column processed_path \
      --checkpoint "$CKPT" --class-map "$CMAP" --source-filter "$SRC" \
      --output-json "artifacts/checkpoints/${MODEL}/eval_${SRC}_segmented.json" \
      --num-workers 0
  done
done
echo "=== eval matrix complete ==="
