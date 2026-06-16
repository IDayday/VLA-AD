#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "Missing OUTPUT_DIR" >&2
  exit 2
fi
if [[ -z "${STAGE1_V2_LAUNCHER:-}" ]]; then
  echo "Missing STAGE1_V2_LAUNCHER" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPU_MAX_MEM_USED_MB="${GPU_MAX_MEM_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"
mkdir -p "${OUTPUT_DIR}/logs"
WAIT_LOG="${OUTPUT_DIR}/logs/gpu_wait.log"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"

IFS=',' read -r -a gpu_ids <<<"${GPU_LIST}"

gpu_ready() {
  local csv
  csv="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  for gpu in "${gpu_ids[@]}"; do
    local line mem util
    line="$(printf '%s\n' "${csv}" | awk -F',' -v g="${gpu}" '$1+0==g+0 {print $0; exit}')"
    if [[ -z "${line}" ]]; then
      echo "GPU ${gpu} not found" >>"${WAIT_LOG}"
      return 1
    fi
    mem="$(printf '%s' "${line}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
    util="$(printf '%s' "${line}" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
    if (( mem > GPU_MAX_MEM_USED_MB || util > GPU_MAX_UTIL )); then
      echo "GPU ${gpu} busy: mem=${mem}MiB util=${util}%" >>"${WAIT_LOG}"
      return 1
    fi
  done
  return 0
}

{
  echo "[$(date -Is)] waiting for GPUs ${GPU_LIST}; max_mem=${GPU_MAX_MEM_USED_MB}MiB max_util=${GPU_MAX_UTIL}%"
  echo "launcher=${STAGE1_V2_LAUNCHER}"
} >>"${WAIT_LOG}"

while ! gpu_ready; do
  sleep "${GPU_WAIT_POLL_SECONDS}"
done

{
  printf '[%s] launching after GPU wait: ' "$(date -Is)"
  printf '%q ' bash "${STAGE1_V2_LAUNCHER}"
  printf '\n'
} >>"${COMMANDS_LOG}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export RUN_TRAIN=1
exec bash "${STAGE1_V2_LAUNCHER}"
