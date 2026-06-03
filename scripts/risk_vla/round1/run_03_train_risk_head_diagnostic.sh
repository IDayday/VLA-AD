#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project/VLA-AD/checkpoints}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
CACHE_PATH="${CACHE_PATH:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/recogdrive/ReCogDrive-2B-IL}"
MAX_SAMPLES="${MAX_SAMPLES:-4096}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CMD=(python scripts/risk_vla/run_risk_head_diagnostic_train.py --cache-path "${CACHE_PATH}" --output-dir "${OUTPUT_DIR}" --checkpoint-path "${CHECKPOINT_PATH}" --max-samples "${MAX_SAMPLES}" --max-epochs "${MAX_EPOCHS}" --limit-train-batches "${LIMIT_TRAIN_BATCHES}" --num-workers "${NUM_WORKERS}")
if [[ "${EXECUTE}" == "1" ]]; then
  CMD+=(--execute)
fi

echo "[run_03] cache path: ${CACHE_PATH}"
echo "[run_03] output dir: ${OUTPUT_DIR}"
echo "[run_03] checkpoint path: ${CHECKPOINT_PATH}"
printf '[run_03] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_03] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to generate artifacts and launch training."
fi
