#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
INPUT_CSV="${INPUT_CSV:-${ROUND_DIR}/matched_pdm/matched_pdm.csv}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ROUND_DIR}/risk_labels/risk_labels.jsonl}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROUND_DIR}/risk_labels/risk_label_summary.csv}"
OUTPUT_MD="${OUTPUT_MD:-${ROUND_DIR}/risk_labels/risk_label_report.md}"

CMD=(python scripts/risk_vla/build_risk_labels_from_pdm.py --input-csv "${INPUT_CSV}" --output-jsonl "${OUTPUT_JSONL}" --output-csv "${OUTPUT_CSV}" --output-md "${OUTPUT_MD}" --schema both)

echo "[run_01] input csv: ${INPUT_CSV}"
echo "[run_01] output jsonl: ${OUTPUT_JSONL}"
echo "[run_01] output csv: ${OUTPUT_CSV}"
echo "[run_01] output md: ${OUTPUT_MD}"
printf '[run_01] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_01] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to run."
fi
