#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
NAVSIM_ROOT="${NAVSIM_ROOT:-/mnt/navsim}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
CHECKPOINT="${CHECKPOINT:-${ROUND_DIR}/R0_risk_head_diagnostic/train/best.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROUND_DIR}/R1_oracle_router_pilot}"
CHUNK_CACHE_DIR="${CHUNK_CACHE_DIR:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"

CMD=(python scripts/risk_vla/build_oracle_router_pilot_eval_command.py --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --navsim-root "${NAVSIM_ROOT}" --chunk-cache-dir "${CHUNK_CACHE_DIR}" --max-samples "${MAX_SAMPLES}")
if [[ "${EXECUTE}" == "1" ]]; then
  CMD+=(--execute)
fi

echo "[run_06] analysis-only oracle router pilot"
echo "[run_06] checkpoint: ${CHECKPOINT}"
echo "[run_06] output dir: ${OUTPUT_DIR}"
printf '[run_06] command: '; printf '%q ' "${CMD[@]}"; echo
if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_06] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to generate artifacts and launch pilot evaluation."
fi
