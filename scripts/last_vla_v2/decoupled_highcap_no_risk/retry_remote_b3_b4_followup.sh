#!/usr/bin/env bash
set -Eeuo pipefail

required=(ORCH_ROOT OUT_ROOT FULL_HIGHCAP_TRAIN_CHUNK_ROOT MASTER_PORT VLM_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

REMOTE_HOST="${REMOTE_HOST:-training-vla-zt-peer}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
NUM_GPUS="${NUM_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-8}"
NUM_SHARDS="${NUM_SHARDS:-64}"
SKIP_VALIDATION="${SKIP_VALIDATION:-1}"
SKIP_HIDDEN_CACHE_AUDIT="${SKIP_HIDDEN_CACHE_AUDIT:-1}"
WAIT_FOR_ADAPTERS="${WAIT_FOR_ADAPTERS:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
POLL_SECONDS="${POLL_SECONDS:-120}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
RUN_ID="${RUN_ID:-$(basename "${ORCH_ROOT}")_b3b4_resume_$(date -u +%Y%m%dT%H%M%SZ)}"
FOLLOWUP_SCRIPT="${FOLLOWUP_SCRIPT:-scripts/last_vla_v2/decoupled_highcap_no_risk/run_lora_cache_and_progressive_decoupled.sh}"

ROOT="${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
LOG_DIR="${ROOT}/logs"
PID_FILE="${ORCH_ROOT}/serverB_b3_b4_followup_remote.pid"
REMOTE_JOB_LOG="${LOG_DIR}/serverB_b3_b4_followup_remote.log"
mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

adapter_ready() {
  [[ -f "${B1}/adapters/last_vla_cot_adapter.pt" ]] && [[ -d "${B1}/adapters/vlm_lora" ]]
}

if ! adapter_ready; then
  log "B1 adapters are not ready under ${B1}/adapters"
  exit 3
fi

while true; do
  if probe_output="$(
    ssh -o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT}" "${REMOTE_HOST}" \
      "hostname && cd '${REPO_ROOT}' && '${PYTHON_BIN}' -c 'import torch; print(torch.cuda.device_count())'" 2>&1
  )"; then
    log "remote ready on ${REMOTE_HOST}: ${probe_output//$'\n'/; }"
    break
  fi
  log "remote not ready on ${REMOTE_HOST}: ${probe_output//$'\n'/; }"
  sleep "${POLL_SECONDS}"
done

remote_pid="$(
  ssh -o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT}" "${REMOTE_HOST}" 'bash -s' <<REMOTE
set -Eeuo pipefail
cd '${REPO_ROOT}'
mkdir -p '${LOG_DIR}'
test -f '${B1}/adapters/last_vla_cot_adapter.pt'
test -d '${B1}/adapters/vlm_lora'
nohup env \
  FULL_HIGHCAP_TRAIN_CHUNK_ROOT='${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' \
  OUT_ROOT='${OUT_ROOT}' \
  MASTER_PORT='${MASTER_PORT}' \
  VLM_PATH='${VLM_PATH}' \
  PYTHON_BIN='${PYTHON_BIN}' \
  TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' \
  NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' \
  OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' \
  NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' \
  NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' \
  SKIP_VALIDATION='${SKIP_VALIDATION}' \
  NUM_GPUS='${NUM_GPUS}' \
  PROCS_PER_GPU='${PROCS_PER_GPU}' \
  NUM_SHARDS='${NUM_SHARDS}' \
  SKIP_HIDDEN_CACHE_AUDIT='${SKIP_HIDDEN_CACHE_AUDIT}' \
  WAIT_FOR_ADAPTERS='${WAIT_FOR_ADAPTERS}' \
  WAIT_SECONDS='${WAIT_SECONDS}' \
  RUN_ID='${RUN_ID}' \
  bash '${FOLLOWUP_SCRIPT}' > '${REMOTE_JOB_LOG}' 2>&1 < /dev/null &
printf '%s\n' "\$!"
REMOTE
)"

printf '%s\n' "${remote_pid}" > "${PID_FILE}"
log "launched remote B3/B4 follow-up pid=${remote_pid} log=${REMOTE_JOB_LOG}"
