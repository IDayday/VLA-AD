#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
python scripts/risk_vla/eval_candidate_bank_pdm.py \
  --candidate-cache-dir "${EXP_ROOT}/candidate_bank" \
  --output-dir "${EXP_ROOT}/eval_small" \
  --max-samples "${MAX_SAMPLES}" \
  $( [[ "${DRY_RUN}" == "1" ]] && printf -- "--dry-run" )
