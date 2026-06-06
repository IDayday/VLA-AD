#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-${BIT_WORK_ROOT}/cache/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
SPLIT="${SPLIT:-train}"
K="${K:-8}"
SEED="${SEED:-20260606}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_CACHE_ROOT}/candidate_bank/${SPLIT}_k${K}}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${BIT_EXP_ROOT}/candidate_bank"

if [[ -z "${BASE_PREDICTIONS_JSONL:-}" ]]; then
  cat > "${BIT_EXP_ROOT}/candidate_bank/CANDIDATE_BANK_BLOCKER.md" <<EOF
# Candidate Bank Blocker

Set BASE_PREDICTIONS_JSONL to an explicit full/10k base prediction JSONL.
Optional inputs: BIT_PREDICTIONS_JSONL, D5_PREDICTIONS_JSONL, RISK_VLA_V1_PREDICTIONS_JSONL, RISK_VLA_V2_PREDICTIONS_JSONL.
Minimum formal scale: ${MIN_REAL_SAMPLES} tokens.
EOF
  echo "Blocked: missing BASE_PREDICTIONS_JSONL"
  [[ "${DRY_RUN}" == "1" ]] && exit 0 || exit 1
fi

args=(
  --base-predictions-jsonl "${BASE_PREDICTIONS_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --k "${K}"
  --seed "${SEED}"
  --min-real-samples "${MIN_REAL_SAMPLES}"
)
[[ -n "${BIT_PREDICTIONS_JSONL:-}" ]] && args+=(--bit-predictions-jsonl "${BIT_PREDICTIONS_JSONL}")
[[ -n "${D5_PREDICTIONS_JSONL:-}" ]] && args+=(--d5-predictions-jsonl "${D5_PREDICTIONS_JSONL}")
[[ -n "${RISK_VLA_PREDICTIONS_JSONL:-}" ]] && args+=(--risk-vla-predictions-jsonl "${RISK_VLA_PREDICTIONS_JSONL}")
[[ -n "${RISK_VLA_V1_PREDICTIONS_JSONL:-}" ]] && args+=(--risk-vla-v1-predictions-jsonl "${RISK_VLA_V1_PREDICTIONS_JSONL}")
[[ -n "${RISK_VLA_V2_PREDICTIONS_JSONL:-}" ]] && args+=(--risk-vla-v2-predictions-jsonl "${RISK_VLA_V2_PREDICTIONS_JSONL}")
[[ -n "${MAX_SAMPLES:-}" ]] && args+=(--max-samples "${MAX_SAMPLES}")
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)

python scripts/risk_vla/export_candidate_bank.py "${args[@]}"
if [[ "${DRY_RUN}" != "1" ]]; then
  python scripts/risk_vla/eval_candidate_bank_pdm.py \
    --candidate-cache-dir "${OUTPUT_DIR}" \
    --metric-cache-dir "${METRIC_CACHE_DIR:-${BIT_CACHE_ROOT}/missing_metric_cache}" \
    --output-dir "${BIT_EXP_ROOT}/candidate_bank/${SPLIT}_k${K}_pdm_plan"
fi
