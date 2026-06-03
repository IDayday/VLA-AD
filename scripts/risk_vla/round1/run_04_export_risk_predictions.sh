#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
CACHE_PATH="${CACHE_PATH:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
CHECKPOINT="${CHECKPOINT:-${ROUND_DIR}/R0_risk_head_diagnostic/train/best.ckpt}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROUND_DIR}/risk_predictions/risk_predictions.csv}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cpu}"
SYNTHETIC_SMOKE="${SYNTHETIC_SMOKE:-0}"

CMD=(python scripts/risk_vla/export_risk_vla_predictions.py --checkpoint "${CHECKPOINT}" --cache-path "${CACHE_PATH}" --split navval --output-csv "${OUTPUT_CSV}" --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}" --device "${DEVICE}")
if [[ "${SYNTHETIC_SMOKE}" == "1" ]]; then
  CMD+=(--synthetic-smoke)
fi

echo "[run_04] cache path: ${CACHE_PATH}"
echo "[run_04] checkpoint: ${CHECKPOINT}"
echo "[run_04] output csv: ${OUTPUT_CSV}"
printf '[run_04] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_04] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to export predictions."
fi
