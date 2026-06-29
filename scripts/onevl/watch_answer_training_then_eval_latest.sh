#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
ANSWER_RUN=${ANSWER_RUN:-/mnt/project/onevl_navsim_exp/answer_full_20260624_184133}
ANSWER_SWIFT_DIR=${ANSWER_SWIFT_DIR:-${ANSWER_RUN}/swift_output/v0-20260624-184217}
TRAIN_PID_FILE=${TRAIN_PID_FILE:-${ANSWER_RUN}/launcher.pid}
OUT_ROOT=${OUT_ROOT:-/mnt/project/onevl_navsim_exp/local_post_train_ar_answer_latest_navtest_eval_$(date +%Y%m%d_%H%M%S)}
TEST_SET_PATH=${TEST_SET_PATH:-/mnt/project/onevl/test_data/navsim_test.json}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PDM_NUM_WORKERS=${PDM_NUM_WORKERS:-${NPROC_PER_NODE}}
POLL_SECONDS=${POLL_SECONDS:-60}
GPU_MEM_MAX_MB=${GPU_MEM_MAX_MB:-20000}
GPU_UTIL_MAX=${GPU_UTIL_MAX:-10}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
INVALID_TRAJECTORY_POLICY=${INVALID_TRAJECTORY_POLICY:-error}

mkdir -p "${OUT_ROOT}/logs"
ln -sfn "${OUT_ROOT}" /mnt/project/onevl_navsim_exp/local_post_train_ar_answer_eval_latest
LOG_FILE="${OUT_ROOT}/logs/watch_then_eval.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "$0" "$@"
  printf '\n'
  env | sort | grep -E '^(PYTHON_BIN|ANSWER_RUN|ANSWER_SWIFT_DIR|TRAIN_PID_FILE|OUT_ROOT|TEST_SET_PATH|NPROC_PER_NODE|PDM_NUM_WORKERS|POLL_SECONDS|GPU_MEM_MAX_MB|GPU_UTIL_MAX|WAIT_FOR_GPUS|INVALID_TRAJECTORY_POLICY)=' || true
} >> "${OUT_ROOT}/commands.log"

training_running() {
  if [ ! -f "${TRAIN_PID_FILE}" ]; then
    return 1
  fi
  local pid
  pid="$(cat "${TRAIN_PID_FILE}" 2>/dev/null || true)"
  [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1
}

latest_step() {
  local log_path="${ANSWER_SWIFT_DIR}/logging.jsonl"
  if [ ! -f "${log_path}" ]; then
    printf 'unknown'
    return
  fi
  tail -n 1 "${log_path}" | "${PYTHON_BIN}" -c 'import json,sys
line=sys.stdin.read().strip()
if not line:
    print("unknown")
else:
    row=json.loads(line)
    print(row.get("global_step/max_steps", "unknown"))'
}

gpus_ready() {
  local rows
  rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  printf '%s\n' "${rows}" > "${OUT_ROOT}/logs/last_gpu_status.csv"
  awk -F, -v mem_max="${GPU_MEM_MAX_MB}" -v util_max="${GPU_UTIL_MAX}" '
    {
      gsub(/ /, "", $2); gsub(/ /, "", $3)
      if ($2 > mem_max || $3 > util_max) bad=1
    }
    END { exit bad ? 1 : 0 }
  ' <<< "${rows}"
}

log "watcher_started answer_run=${ANSWER_RUN} out_root=${OUT_ROOT}"

while training_running; do
  log "training_still_running pid=$(cat "${TRAIN_PID_FILE}") step=$(latest_step); retry_in=${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "training_not_running; preparing local latest checkpoint evaluation"

if [ "${WAIT_FOR_GPUS}" = "1" ]; then
  while ! gpus_ready; do
    max_line="$(sort -t, -k2 -nr "${OUT_ROOT}/logs/last_gpu_status.csv" | head -n 1 || true)"
    log "gpu_not_ready max_row=${max_line}; mem_max=${GPU_MEM_MAX_MB}MB util_max=${GPU_UTIL_MAX}; retry_in=${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
  log "gpu_ready_for_eval"
fi

setsid env \
  PYTHON_BIN="${PYTHON_BIN}" \
  ANSWER_SWIFT_DIR="${ANSWER_SWIFT_DIR}" \
  OUT_ROOT="${OUT_ROOT}" \
  TEST_SET_PATH="${TEST_SET_PATH}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  PDM_NUM_WORKERS="${PDM_NUM_WORKERS}" \
  INVALID_TRAJECTORY_POLICY="${INVALID_TRAJECTORY_POLICY}" \
  RUN_PDM_EVAL=1 \
  "${SCRIPT_DIR}/run_ar_answer_latest_infer_eval_full.sh" \
  > "${OUT_ROOT}/logs/eval_launcher.outer.log" 2>&1 < /dev/null &
echo $! > "${OUT_ROOT}/eval_launcher.pid"
log "eval_launch_done pid=$(cat "${OUT_ROOT}/eval_launcher.pid") out_root=${OUT_ROOT}"
