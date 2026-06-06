#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/critic"
cat > "${EXP_ROOT}/critic/critic_train_plan.json" <<EOF
{"dry_run": ${DRY_RUN}, "config": "configs/risk_vla/v2/risk_vla_v2_R4_trajectory_critic.yaml", "requires": ["strategy_utility_labels", "candidate_bank", "finegrained_labels"]}
EOF
echo "RISK-VLA v2 critic training plan written to ${EXP_ROOT}/critic/critic_train_plan.json"
