#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-${ROUND_DIR}/risk_predictions/risk_predictions.csv}"
LABELS_JSONL="${LABELS_JSONL:-${ROUND_DIR}/risk_labels/risk_labels.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROUND_DIR}/risk_diagnostics}"

CMD=(python scripts/risk_vla/aggregate_risk_vla_diagnostics.py --predictions-csv "${PREDICTIONS_CSV}" --labels-jsonl "${LABELS_JSONL}" --output-dir "${OUTPUT_DIR}")

echo "[run_05] predictions csv: ${PREDICTIONS_CSV}"
echo "[run_05] labels jsonl: ${LABELS_JSONL}"
echo "[run_05] output dir: ${OUTPUT_DIR}"
printf '[run_05] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_05] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to aggregate diagnostics."
fi
