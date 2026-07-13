#!/usr/bin/env bash
set -Eeuo pipefail

required=(TRAIN_CHUNK_CACHE_ROOT OUTPUT_DIR MASTER_PORT READINESS_GATE_JSON A0_INIT_CHECKPOINT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
REQUIRED_FREE_GPUS="${REQUIRED_FREE_GPUS:-8}"
MAX_USED_MB="${MAX_USED_MB:-1024}"
POLL_SECONDS="${POLL_SECONDS:-600}"
STAGE2_LAUNCHER="${STAGE2_LAUNCHER:-scripts/last_vla_v2/two_expert_slot/run_stage2_two_expert_dit_sft_8gpu.sh}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "wait_gpus_then_run_stage2 started at $(date -u)"
echo "output_dir=${OUTPUT_DIR}"
echo "cache_root=${TRAIN_CHUNK_CACHE_ROOT}"
echo "required_free_gpus=${REQUIRED_FREE_GPUS}"
echo "max_used_mb=${MAX_USED_MB}"
echo "poll_seconds=${POLL_SECONDS}"

while true; do
  free_count="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | awk -v max_used="${MAX_USED_MB}" '$1 <= max_used {count += 1} END {print count + 0}'
  )"
  snapshot="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
  echo "$(date -u) free_gpus=${free_count}/${REQUIRED_FREE_GPUS} snapshot=${snapshot}"
  if [[ "${free_count}" -ge "${REQUIRED_FREE_GPUS}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "$(date -u) launching Stage2"
RUN_TRAIN=1 \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTHON_BIN="${PYTHON_BIN}" \
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MASTER_PORT="${MASTER_PORT}" \
READINESS_GATE_JSON="${READINESS_GATE_JSON}" \
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT}" \
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}" \
bash "${STAGE2_LAUNCHER}"
