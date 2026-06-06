#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
DRY_RUN="${DRY_RUN:-1}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/critic"
CANDIDATE_NPZ="${CANDIDATE_NPZ:-${EXP_ROOT}/candidate_bank/candidate_trajectories.npz}"
UTILITY_LABELS_JSONL="${UTILITY_LABELS_JSONL:-${EXP_ROOT}/utility_labels/strategy_utility_labels.jsonl}"
if [[ "${DRY_RUN}" == "1" || ! -f "${CANDIDATE_NPZ}" || ! -f "${UTILITY_LABELS_JSONL}" ]]; then
  cat > "${EXP_ROOT}/critic/critic_train_plan.json" <<EOF
{"dry_run": ${DRY_RUN}, "blocked": true, "config": "configs/risk_vla/v2/risk_vla_v2_R4_trajectory_critic.yaml", "candidate_npz": "${CANDIDATE_NPZ}", "labels_jsonl": "${UTILITY_LABELS_JSONL}", "requires": ["strategy_utility_labels", "candidate_bank"]}
EOF
  echo "RISK-VLA v2 critic training blocked/plan written to ${EXP_ROOT}/critic/critic_train_plan.json"
  exit 0
fi
python scripts/risk_vla/train_trajectory_critic_from_assets.py \
  --candidate-npz "${CANDIDATE_NPZ}" \
  --labels-jsonl "${UTILITY_LABELS_JSONL}" \
  --output-dir "${EXP_ROOT}/critic" \
  --purpose training \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}"
