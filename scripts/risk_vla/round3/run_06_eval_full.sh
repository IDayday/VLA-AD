#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_EXP_ROOT}/eval/full_analysis}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${CRITIC_CHECKPOINT:-}" && -n "${EVAL_CANDIDATE_NPZ:-}" && -n "${EVAL_LABELS_JSONL:-}" ]]; then
  args=(
    --checkpoint "${CRITIC_CHECKPOINT}"
    --candidate-npz "${EVAL_CANDIDATE_NPZ}"
    --labels-jsonl "${EVAL_LABELS_JSONL}"
    --output-dir "${OUTPUT_DIR}/critic"
    --purpose analysis
    --device "${DEVICE:-cuda}"
    --min-real-samples "${MIN_REAL_SAMPLES}"
  )
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
  python scripts/risk_vla/eval_trajectory_risk_critic.py "${args[@]}"
else
  cat > "${OUTPUT_DIR}/EVAL_BLOCKER.md" <<EOF
# Full Evaluation Blocker

Set CRITIC_CHECKPOINT, EVAL_CANDIDATE_NPZ, and EVAL_LABELS_JSONL for >=${MIN_REAL_SAMPLES} held-out/navtest analysis.
Navtest labels are analysis-only and must not be used for tuning.
EOF
  echo "Blocked: missing critic eval inputs"
  [[ "${DRY_RUN}" == "1" ]] || exit 1
fi

if [[ -n "${ROUTER_CHECKPOINT:-}" && -n "${EVAL_LABELS_JSONL:-}" ]]; then
  args=(--checkpoint "${ROUTER_CHECKPOINT}" --labels-jsonl "${EVAL_LABELS_JSONL}" --output-dir "${OUTPUT_DIR}/router" --purpose analysis --device "${DEVICE:-cuda}" --min-real-samples "${MIN_REAL_SAMPLES}")
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
  python scripts/risk_vla/eval_utility_router.py "${args[@]}"
fi
