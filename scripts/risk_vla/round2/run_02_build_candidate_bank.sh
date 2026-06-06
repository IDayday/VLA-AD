#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DRY_RUN="${DRY_RUN:-1}"
K="${K:-4}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/candidate_bank"
if [[ -z "${BASE_PREDICTIONS_JSONL:-}" ]]; then
  echo "Missing BASE_PREDICTIONS_JSONL. Candidate bank export needs explicit prediction inputs." | tee "${EXP_ROOT}/candidate_bank/missing_inputs.txt"
  exit 0
fi
args=(--base-predictions-jsonl "${BASE_PREDICTIONS_JSONL}" --output-dir "${EXP_ROOT}/candidate_bank" --k "${K}")
if [[ -n "${BIT_PREDICTIONS_JSONL:-}" ]]; then args+=(--bit-predictions-jsonl "${BIT_PREDICTIONS_JSONL}"); fi
if [[ -n "${RISK_VLA_PREDICTIONS_JSONL:-}" ]]; then args+=(--risk-vla-predictions-jsonl "${RISK_VLA_PREDICTIONS_JSONL}"); fi
if [[ -n "${MAX_SAMPLES}" ]]; then args+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ "${DRY_RUN}" == "1" ]]; then args+=(--dry-run); fi
python scripts/risk_vla/export_candidate_bank.py "${args[@]}"
