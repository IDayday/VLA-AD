#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST=${REMOTE_HOST:-training-rl-zt3}
RETRY_SECONDS=${RETRY_SECONDS:-60}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/project}
OUT_ROOT=${OUT_ROOT:-/mnt/project/onevl_navsim_exp/remote_${REMOTE_HOST}_ar_answer_latest_navtest_eval_$(date +%Y%m%d_%H%M%S)}
ANSWER_SWIFT_DIR=${ANSWER_SWIFT_DIR:-/mnt/project/onevl_navsim_exp/answer_full_20260624_184133/swift_output/v0-20260624-184217}
REMOTE_GPU_MAX_USED_MB=${REMOTE_GPU_MAX_USED_MB:-20000}
REMOTE_GPU_MAX_UTIL=${REMOTE_GPU_MAX_UTIL:-80}

mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/retry_remote_launcher.log"
SSH_PROBE_LOG="${OUT_ROOT}/logs/ssh_probe.last.log"

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "$0" "$@"
  printf '\n'
  env | sort | grep -E '^(REMOTE_HOST|RETRY_SECONDS|PYTHON_BIN|PROJECT_ROOT|OUT_ROOT|ANSWER_SWIFT_DIR|NPROC_PER_NODE|REMOTE_GPU_MAX_USED_MB|REMOTE_GPU_MAX_UTIL)=' || true
} >> "${OUT_ROOT}/commands.log"

while true; do
  if timeout 12 ssh "${ssh_opts[@]}" "${REMOTE_HOST}" 'hostname >/dev/null' > "${SSH_PROBE_LOG}" 2>&1; then
    log "ssh_ready host=${REMOTE_HOST}; running remote preflight"
    ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
      "PROJECT_ROOT='${PROJECT_ROOT}' PYTHON_BIN='${PYTHON_BIN}' ANSWER_SWIFT_DIR='${ANSWER_SWIFT_DIR}' bash -s" <<'REMOTE_PREFLIGHT'
set -euo pipefail
hostname
test -d "${PROJECT_ROOT}"
test -x "${PYTHON_BIN}"
test -d "${ANSWER_SWIFT_DIR}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import socket
import torch
print("host", socket.gethostname())
print("torch", torch.__version__)
print("cuda_count", torch.cuda.device_count())
if torch.cuda.device_count() < 1:
    raise SystemExit("no CUDA devices visible")
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
REMOTE_PREFLIGHT

    remote_gpu_count_raw=$(ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
      "PYTHON_BIN='${PYTHON_BIN}' bash -s" <<'REMOTE_GPU_COUNT'
set -euo pipefail
"${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
REMOTE_GPU_COUNT
)
    remote_gpu_count=$(printf '%s\n' "${remote_gpu_count_raw}" | awk '/^[0-9]+$/ {v=$1} END {print v}')
    if [ -z "${remote_gpu_count}" ] || [ "${remote_gpu_count}" -lt 1 ]; then
      log "remote_gpu_count_invalid host=${REMOTE_HOST}; value=${remote_gpu_count_raw@Q}"
      exit 1
    fi

    remote_gpu_status=$(ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
      "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")
    busy_gpu_count=$(printf '%s\n' "${remote_gpu_status}" | awk -F, \
      -v max_mem="${REMOTE_GPU_MAX_USED_MB}" \
      -v max_util="${REMOTE_GPU_MAX_UTIL}" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)
          if ($2 > max_mem || $3 > max_util) busy += 1
        }
        END {print busy + 0}
      ')
    if [ "${busy_gpu_count}" -gt 0 ]; then
      log "remote_gpus_busy host=${REMOTE_HOST}; busy=${busy_gpu_count}; max_used_mb=${REMOTE_GPU_MAX_USED_MB}; max_util=${REMOTE_GPU_MAX_UTIL}; retry_in=${RETRY_SECONDS}s"
      printf '%s\n' "${remote_gpu_status}" >> "${LOG_FILE}"
      sleep "${RETRY_SECONDS}"
      continue
    fi
    remote_nproc=${NPROC_PER_NODE:-${remote_gpu_count}}

    log "remote_preflight_ok host=${REMOTE_HOST}; cuda_count=${remote_gpu_count}; nproc_per_node=${remote_nproc}; launching latest AR Answer navtest eval"
    ssh "${ssh_opts[@]}" "${REMOTE_HOST}" \
      "cd '${PROJECT_ROOT}' && mkdir -p '${OUT_ROOT}/logs' && setsid env \
        PYTHON_BIN='${PYTHON_BIN}' \
        ANSWER_SWIFT_DIR='${ANSWER_SWIFT_DIR}' \
        OUT_ROOT='${OUT_ROOT}' \
        NPROC_PER_NODE='${remote_nproc}' \
        PDM_NUM_WORKERS='${remote_nproc}' \
        RUN_PDM_EVAL=1 \
        '${SCRIPT_DIR}/run_ar_answer_latest_infer_eval_full.sh' \
        > '${OUT_ROOT}/logs/remote_launcher.outer.log' 2>&1 < /dev/null & echo \\\$! > '${OUT_ROOT}/remote_launcher.pid'"
    log "remote_launch_done host=${REMOTE_HOST}; pid_file=${OUT_ROOT}/remote_launcher.pid"
    exit 0
  fi
  ssh_reason=$(tail -n 1 "${SSH_PROBE_LOG}" 2>/dev/null | tr '\r\n' '  ' | sed 's/[[:space:]]*$//' || true)
  if printf '%s' "${ssh_reason}" | grep -qi 'Permission denied'; then
    log "ssh_auth_failed host=${REMOTE_HOST}; reason=${ssh_reason}; retry_in=${RETRY_SECONDS}s"
  elif printf '%s' "${ssh_reason}" | grep -qi 'Connection refused'; then
    log "ssh_connection_refused host=${REMOTE_HOST}; reason=${ssh_reason}; retry_in=${RETRY_SECONDS}s"
  elif [ -n "${ssh_reason}" ]; then
    log "ssh_not_ready host=${REMOTE_HOST}; reason=${ssh_reason}; retry_in=${RETRY_SECONDS}s"
  else
    log "ssh_not_ready host=${REMOTE_HOST}; retry_in=${RETRY_SECONDS}s"
  fi
  sleep "${RETRY_SECONDS}"
done
