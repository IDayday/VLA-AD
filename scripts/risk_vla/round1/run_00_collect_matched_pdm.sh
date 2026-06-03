#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
PDM_INPUTS="${PDM_INPUTS:-A0_base=${ROUND_DIR}/inputs/a0_pdm.csv B3_direct_bit=${ROUND_DIR}/inputs/b3_direct_bit_pdm.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROUND_DIR}/matched_pdm/matched_pdm.csv}"
OUTPUT_MD="${OUTPUT_MD:-${ROUND_DIR}/matched_pdm/matched_pdm.md}"

read -r -a INPUT_ARGS <<< "${PDM_INPUTS}"
CMD=(python scripts/risk_vla/collect_matched_pdm_results.py --inputs "${INPUT_ARGS[@]}" --output-csv "${OUTPUT_CSV}" --output-md "${OUTPUT_MD}")

echo "[run_00] inputs: ${PDM_INPUTS}"
echo "[run_00] output csv: ${OUTPUT_CSV}"
echo "[run_00] output md: ${OUTPUT_MD}"
printf '[run_00] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_00] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to run."
fi
