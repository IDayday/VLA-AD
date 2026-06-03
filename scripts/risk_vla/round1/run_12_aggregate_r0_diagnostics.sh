#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
R0_DIR="${R0_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
SPLIT="${SPLIT:-navval}"
PREDICTIONS_CSV="${PREDICTIONS_CSV:-${R0_DIR}/predictions_${SPLIT}.csv}"
LABELS_JSONL="${LABELS_JSONL:-${ROUND_DIR}/risk_labels/risk_labels.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${R0_DIR}/diagnostics_${SPLIT}}"

CMD=(python scripts/risk_vla/aggregate_risk_vla_diagnostics.py --predictions-csv "${PREDICTIONS_CSV}" --labels-jsonl "${LABELS_JSONL}" --output-dir "${OUTPUT_DIR}")

echo "[run_12] predictions: ${PREDICTIONS_CSV}"
echo "[run_12] labels: ${LABELS_JSONL}"
echo "[run_12] output dir: ${OUTPUT_DIR}"
printf '[run_12] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_12] EXECUTE=0; no aggregation run."
fi
