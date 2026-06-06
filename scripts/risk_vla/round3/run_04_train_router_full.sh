#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_EXP_ROOT}/router/full}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
if [[ -z "${LABELS_JSONL:-}" ]]; then
  cat > "${OUTPUT_DIR}/ROUTER_BLOCKER.md" <<EOF
# Router Training Blocker

Set LABELS_JSONL to explicit >=${MIN_REAL_SAMPLES} train strategy utility labels.
Navtest/test labels are forbidden for training.
EOF
  echo "Blocked: missing LABELS_JSONL"
  [[ "${DRY_RUN}" == "1" ]] && exit 0 || exit 1
fi

args=(
  --labels-jsonl "${LABELS_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --purpose training
  --epochs "${EPOCHS:-60}"
  --batch-size "${BATCH_SIZE:-128}"
  --device "${DEVICE:-cuda}"
  --min-real-samples "${MIN_REAL_SAMPLES}"
)
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
python scripts/risk_vla/train_utility_router.py "${args[@]}"
