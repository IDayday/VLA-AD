#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project/VLA-AD/checkpoints}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
R0_DIR="${R0_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
CACHE_PATH="${CACHE_PATH:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/recogdrive/ReCogDrive-2B-IL}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
RISK_LOSS_WEIGHT="${RISK_LOSS_WEIGHT:-0.05}"

BUILD_CMD=(python scripts/risk_vla/run_risk_head_diagnostic_train.py --cache-path "${CACHE_PATH}" --output-dir "${R0_DIR}" --checkpoint-path "${CHECKPOINT_PATH}" --max-epochs "${MAX_EPOCHS}" --max-samples "${MAX_SAMPLES}" --limit-train-batches "${LIMIT_TRAIN_BATCHES}" --limit-val-batches "${LIMIT_VAL_BATCHES}" --risk-loss-weight "${RISK_LOSS_WEIGHT}" --extra-override use_risk_vla=true --extra-override risk_vla_strategy_token_scale=0.0 --extra-override risk_vla_horizon_residual_scale=0.0 --extra-override risk_vla_use_oracle_router=false)

echo "[run_10] R0 diagnostic only; strategy conditioning disabled."
echo "[run_10] cache path: ${CACHE_PATH}"
echo "[run_10] output dir: ${R0_DIR}"
printf '[run_10] build command: '; printf '%q ' "${BUILD_CMD[@]}"; echo

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[run_10] EXECUTE=0; no training launched."
  exit 0
fi

mkdir -p "${R0_DIR}"
"${BUILD_CMD[@]}"
cp "${R0_DIR}/risk_vla_diagnostic_config_snapshot.yaml" "${R0_DIR}/config_snapshot.yaml"
bash "${R0_DIR}/resolved_train_command.sh" 2>&1 | tee "${R0_DIR}/train.log"
