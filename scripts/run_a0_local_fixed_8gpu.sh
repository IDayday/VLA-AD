#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  A0_CONFIG
  TRAIN_CHUNK_CACHE_ROOT
  TRAIN_CHUNK_NAME_PATTERN
  OUTPUT_DIR
  MASTER_PORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}. Set ALLOW_OVERWRITE=1 to reuse it." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/a0_local_fixed.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  scripts/train_recogdrive_expert_chunked.py
  --config "${A0_CONFIG}"
  --chunk-cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}"
  --global-epochs 200
  --flat-global-dataset
  --batch-size 16
  --gradient-accumulation-steps 1
  --num-workers 8
  --prefetch-factor 2
  --precision bf16
  --final-check-precision fp32
  --lr-scheduler official-cosine
  --lr-scheduler-epochs 200
  --lr-warmup-epochs 3
  --min-lr 1e-6
  --lr-action-head 1e-4
  --save-every 10000
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
)

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "Starting A0-local-fixed. Log: ${TRAIN_LOG}"
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
