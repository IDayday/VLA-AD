#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/eval_full"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo '{"dry_run": true, "full_eval_started": false, "reason": "dry-run"}' > "${EXP_ROOT}/eval_full/full_eval_gate.json"
  cat "${EXP_ROOT}/eval_full/full_eval_gate.json"
  exit 0
fi
if [[ ! -f "${EXP_ROOT}/eval_small/candidate_bank_pdm_eval_plan.json" ]]; then
  echo '{"full_eval_started": false, "reason": "small eval plan missing"}' > "${EXP_ROOT}/eval_full/full_eval_gate.json"
  cat "${EXP_ROOT}/eval_full/full_eval_gate.json"
  exit 1
fi
echo '{"full_eval_started": false, "reason": "supervised safety gates must be reviewed before full eval"}' > "${EXP_ROOT}/eval_full/full_eval_gate.json"
cat "${EXP_ROOT}/eval_full/full_eval_gate.json"
