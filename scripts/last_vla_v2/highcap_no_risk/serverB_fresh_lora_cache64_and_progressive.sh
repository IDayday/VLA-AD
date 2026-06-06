#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_GEOMETRY_CHUNK_ROOT A0_INIT_CHECKPOINT OUT_ROOT MASTER_PORT VLM_PATH NAVSIM_LOG_PATH SENSOR_BLOBS_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROCS_PER_GPU="${PROCS_PER_GPU:-8}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_SHARDS="${NUM_SHARDS:-$((NUM_GPUS * PROCS_PER_GPU))}"
CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$((MASTER_PORT + 10))}"
SKIP_VALIDATION="${SKIP_VALIDATION:-1}"
B1_EPOCHS="${B1_EPOCHS:-3}"

ROOT="${OUT_ROOT}/serverB_lora_highcap_no_risk"
LOG_DIR="${ROOT}/logs"
COMMANDS_LOG="${ROOT}/commands.log"
mkdir -p "${LOG_DIR}"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "Fresh Server B launch: LoRA alignment, adapter extraction, ${NUM_SHARDS}-shard cache, progressive training"
  echo "out_root=${OUT_ROOT}"
  echo "root=${ROOT}"
  echo "master_port=${MASTER_PORT}"
  echo "cache_master_port=${CACHE_MASTER_PORT}"
  echo "num_gpus=${NUM_GPUS} procs_per_gpu=${PROCS_PER_GPU} num_shards=${NUM_SHARDS}"
  echo "skip_validation=${SKIP_VALIDATION}"
  echo "b1_epochs=${B1_EPOCHS}"
} >>"${COMMANDS_LOG}"

env \
  RUN_TRAIN=1 \
  STOP_AFTER_EXTRACT=1 \
  FULL_GEOMETRY_CHUNK_ROOT="${FULL_GEOMETRY_CHUNK_ROOT}" \
  A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT}" \
  OUT_ROOT="${OUT_ROOT}" \
  MASTER_PORT="${MASTER_PORT}" \
  VLM_PATH="${VLM_PATH}" \
  NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH}" \
  SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SKIP_VALIDATION="${SKIP_VALIDATION}" \
  B1_EPOCHS="${B1_EPOCHS}" \
  scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh

env \
  RUN_TRAIN=1 \
  FULL_GEOMETRY_CHUNK_ROOT="${FULL_GEOMETRY_CHUNK_ROOT}" \
  A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT}" \
  OUT_ROOT="${OUT_ROOT}" \
  MASTER_PORT="${CACHE_MASTER_PORT}" \
  VLM_PATH="${VLM_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  PROCS_PER_GPU="${PROCS_PER_GPU}" \
  NUM_GPUS="${NUM_GPUS}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SKIP_VALIDATION="${SKIP_VALIDATION}" \
  scripts/last_vla_v2/highcap_no_risk/serverB_lora_cache32_and_progressive.sh
