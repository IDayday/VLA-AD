#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
NAVSIM_ROOT="${NAVSIM_ROOT:-/mnt/navsim}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
R0_DIR="${R0_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
R2_DIR="${R2_DIR:-${ROUND_DIR}/R2_predicted_router_pilot}"
CHECKPOINT="${CHECKPOINT:-${R0_DIR}/train/best.ckpt}"
CHUNK_CACHE_DIR="${CHUNK_CACHE_DIR:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
STRATEGY_TOKEN_SCALE="${STRATEGY_TOKEN_SCALE:-0.25}"
HORIZON_RESIDUAL_SCALE="${HORIZON_RESIDUAL_SCALE:-0.25}"

BUILD_CMD=(python scripts/risk_vla/build_predicted_router_pilot_eval_command.py --checkpoint "${CHECKPOINT}" --output-dir "${R2_DIR}" --navsim-root "${NAVSIM_ROOT}" --chunk-cache-dir "${CHUNK_CACHE_DIR}" --max-samples "${MAX_SAMPLES}" --extra-override risk_vla_strategy_token_scale="${STRATEGY_TOKEN_SCALE}" --extra-override risk_vla_horizon_residual_scale="${HORIZON_RESIDUAL_SCALE}" --extra-override risk_vla_use_oracle_router=false)

printf '[run_14] build command: '; printf '%q ' "${BUILD_CMD[@]}"; echo

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[run_14] EXECUTE=0; no eval launched."
  exit 0
fi

"${BUILD_CMD[@]}"
cp "${R2_DIR}/predicted_router_pilot_config_snapshot.yaml" "${R2_DIR}/config_snapshot.yaml"
cp "${R2_DIR}/resolved_predicted_router_eval_command.sh" "${R2_DIR}/resolved_eval_command.sh"
bash "${R2_DIR}/resolved_eval_command.sh" 2>&1 | tee "${R2_DIR}/eval.log"
