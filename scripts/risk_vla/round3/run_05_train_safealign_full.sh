#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_EXP_ROOT}/safealign/full}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
if [[ -z "${CANDIDATE_NPZ:-}" || -z "${PAIRS_JSONL:-}" ]]; then
  cat > "${OUTPUT_DIR}/SAFEALIGN_BLOCKER.md" <<EOF
# SafeAlign SFT Blocker

Set CANDIDATE_NPZ and PAIRS_JSONL to explicit >=${MIN_REAL_SAMPLES} train assets.
Navtest/test labels are forbidden for training.
EOF
  echo "Blocked: missing CANDIDATE_NPZ or PAIRS_JSONL"
  [[ "${DRY_RUN}" == "1" ]] && exit 0 || exit 1
fi

args=(
  --candidate-npz "${CANDIDATE_NPZ}"
  --pairs-jsonl "${PAIRS_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --purpose training
  --epochs "${EPOCHS:-80}"
  --batch-size "${BATCH_SIZE:-128}"
  --device "${DEVICE:-cuda}"
  --min-real-samples "${MIN_REAL_SAMPLES}"
)
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
python scripts/risk_vla/train_risk_vla_safealign_sft.py "${args[@]}"
