#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
DRY_RUN="${DRY_RUN:-1}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/router"
UTILITY_LABELS_JSONL="${UTILITY_LABELS_JSONL:-${EXP_ROOT}/utility_labels/strategy_utility_labels.jsonl}"
if [[ "${DRY_RUN}" == "1" || ! -f "${UTILITY_LABELS_JSONL}" ]]; then
  cat > "${EXP_ROOT}/router/router_train_plan.json" <<EOF
{"dry_run": ${DRY_RUN}, "blocked": true, "config": "configs/risk_vla/v2/risk_vla_v2_R3_utility_router.yaml", "labels_jsonl": "${UTILITY_LABELS_JSONL}", "requires": ["strategy_utility_labels"]}
EOF
  echo "RISK-VLA v2 router training blocked/plan written to ${EXP_ROOT}/router/router_train_plan.json"
  exit 0
fi
python scripts/risk_vla/train_utility_router.py \
  --labels-jsonl "${UTILITY_LABELS_JSONL}" \
  --output-dir "${EXP_ROOT}/router" \
  --purpose training \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}"
