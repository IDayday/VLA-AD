#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/router"
cat > "${EXP_ROOT}/router/router_train_plan.json" <<EOF
{"dry_run": ${DRY_RUN}, "config": "configs/risk_vla/v2/risk_vla_v2_R3_utility_router.yaml", "requires": ["critic", "strategy_utility_labels"]}
EOF
echo "RISK-VLA v2 router training plan written to ${EXP_ROOT}/router/router_train_plan.json"
