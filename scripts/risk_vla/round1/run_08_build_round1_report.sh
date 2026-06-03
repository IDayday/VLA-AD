#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
MATCHED_PDM_CSV="${MATCHED_PDM_CSV:-${ROUND_DIR}/matched_pdm/matched_pdm.csv}"
RISK_DIAGNOSTICS_CSV="${RISK_DIAGNOSTICS_CSV:-${ROUND_DIR}/risk_diagnostics/risk_prediction_metrics.csv}"
TRANSITION_CSV="${TRANSITION_CSV:-${ROUND_DIR}/transitions/transition_metrics.csv}"
STRATEGY_ACTIVATION_CSV="${STRATEGY_ACTIVATION_CSV:-${ROUND_DIR}/risk_diagnostics/strategy_activation_by_subset.csv}"
OUTPUT_MD="${OUTPUT_MD:-${ROUND_DIR}/BIT_RISK_VLA_ROUND1_REPORT.md}"

CMD=(python scripts/risk_vla/build_round1_report.py --matched-pdm-csv "${MATCHED_PDM_CSV}" --risk-diagnostics-csv "${RISK_DIAGNOSTICS_CSV}" --transition-csv "${TRANSITION_CSV}" --strategy-activation-csv "${STRATEGY_ACTIVATION_CSV}" --output-md "${OUTPUT_MD}")

echo "[run_08] output md: ${OUTPUT_MD}"
printf '[run_08] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_08] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to build report."
fi
