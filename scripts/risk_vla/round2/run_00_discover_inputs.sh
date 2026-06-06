#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
args=(--exp-root "${EXP_ROOT}")
if [[ -n "${MAX_SAMPLES}" ]]; then args+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ "${DRY_RUN}" == "1" ]]; then args+=(--dry-run); fi
python scripts/risk_vla/run_risk_vla_v2_plan.py "${args[@]}"
