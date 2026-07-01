#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-training-rl-zt3}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage2 PSI run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"

QUEUE_SPLITS="${QUEUE_SPLITS:-val6000 navtest}"
QUEUE_FOREVER="${QUEUE_FOREVER:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CHECKPOINT_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH:-50}"
CHECKPOINT_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH:-0}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-desc}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
RETRY_FAILED="${RETRY_FAILED:-1}"
VAL6000_MAX_EVALS_PER_CYCLE="${VAL6000_MAX_EVALS_PER_CYCLE:-1}"
NAVTEST_MAX_EVALS_PER_CYCLE="${NAVTEST_MAX_EVALS_PER_CYCLE:-1}"
VAL6000_TOP_K="${VAL6000_TOP_K:-5}"
NAVTEST_FROM_VAL_TOP_K="${NAVTEST_FROM_VAL_TOP_K:-5}"
NAVTEST_BACKUP_TOP_K="${NAVTEST_BACKUP_TOP_K:-3}"
VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST="${VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST:-0}"
EVAL_HOST_ID="${EVAL_HOST_ID:-${REMOTE_HOST}}"
WATCH_LOCK_SCOPE="${WATCH_LOCK_SCOPE:-global}"
EVAL_RESOURCE_LOCK_SCOPE="${EVAL_RESOURCE_LOCK_SCOPE:-global}"

REMOTE_WAIT_FOR_FREE_GPUS="${REMOTE_WAIT_FOR_FREE_GPUS:-0}"
REMOTE_GPU_LIST="${REMOTE_GPU_LIST:-1,2,3,4,5,6}"
REMOTE_GPUS_PER_NODE="${REMOTE_GPUS_PER_NODE:-6}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
MAX_SCENES="${MAX_SCENES:-0}"
BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS:-1}"
BACKUP_ROOT="${BACKUP_ROOT:-${RUN_ROOT}/checkpoint_backups}"
STOP_REMOTE_WATCHERS="${STOP_REMOTE_WATCHERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"
LAUNCH_LOG="${LOG_DIR}/remote_eval_once_queue_launch_${REMOTE_HOST}.log"
REMOTE_LOG="${REMOTE_LOG:-${LOG_DIR}/eval_once_queue.${REMOTE_HOST}.log}"
QUEUE_LOG="${QUEUE_LOG:-${REMOTE_LOG}}"
REMOTE_PID_FILE="${STATE_DIR}/eval_once_queue.${REMOTE_HOST}.pid"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LAUNCH_LOG}"
}

ssh_cmd() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "$@"
}

quote_env() {
  printf '%q=%q ' "$1" "$2"
}

stop_remote_watchers() {
  local script
  script=$(cat <<'SH'
set -euo pipefail
run_root="$1"
active_eval_pids="$(
  for pid in $(pgrep -f 'run_pdm_score_recogdrive|torchrun' || true); do
    [[ "${pid}" != "$$" ]] || continue
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline")"
    if [[ "${cmdline}" == *"${run_root}"* ]]; then
      printf '%s\n' "${pid}"
    fi
  done
)"
if [[ -n "${active_eval_pids}" ]]; then
  echo "active eval process exists for ${run_root}; refusing to stop watchers: ${active_eval_pids}" >&2
  exit 4
fi
stopped=0
for pid in $(pgrep -f 'bash scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh' || true); do
  if [[ -r "/proc/${pid}/environ" ]] && tr '\0' '\n' < "/proc/${pid}/environ" | grep -qx "RUN_ROOT=${run_root}"; then
    kill "${pid}" || true
    stopped=$((stopped + 1))
  fi
done
echo "stopped_watchers=${stopped}"
SH
)
  ssh_cmd "bash -s -- $(printf '%q' "${RUN_ROOT}")" <<< "${script}"
}

log "remote eval once queue launcher start host=${REMOTE_HOST} run_root=${RUN_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  ssh_cmd "test -d $(printf '%q' "${REMOTE_REPO_ROOT}") && test -d $(printf '%q' "${RUN_ROOT}") && hostname && date -u +%Y-%m-%dT%H:%M:%SZ" \
    | sed 's/^/remote_probe: /' | tee -a "${LAUNCH_LOG}"
fi

if [[ "${STOP_REMOTE_WATCHERS}" == "1" ]]; then
  log "stopping idle remote watchers for this run"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN stop_remote_watchers run_root=${RUN_ROOT}" | tee -a "${LAUNCH_LOG}"
  else
    stop_remote_watchers | sed 's/^/remote_stop: /' | tee -a "${LAUNCH_LOG}"
  fi
fi

ENV_ARGS=""
ENV_ARGS+=$(quote_env RUN_ROOT "${RUN_ROOT}")
ENV_ARGS+=$(quote_env CHECKPOINT_ROOT "${CHECKPOINT_ROOT}")
ENV_ARGS+=$(quote_env STAGE "stage2")
ENV_ARGS+=$(quote_env QUEUE_SPLITS "${QUEUE_SPLITS}")
ENV_ARGS+=$(quote_env QUEUE_FOREVER "${QUEUE_FOREVER}")
ENV_ARGS+=$(quote_env POLL_SECONDS "${POLL_SECONDS}")
ENV_ARGS+=$(quote_env EVAL_HOST_ID "${EVAL_HOST_ID}")
ENV_ARGS+=$(quote_env WATCH_LOCK_SCOPE "${WATCH_LOCK_SCOPE}")
ENV_ARGS+=$(quote_env EVAL_RESOURCE_LOCK_SCOPE "${EVAL_RESOURCE_LOCK_SCOPE}")
ENV_ARGS+=$(quote_env QUEUE_LOG "${QUEUE_LOG}")
ENV_ARGS+=$(quote_env CHECKPOINT_MIN_EPOCH "${CHECKPOINT_MIN_EPOCH}")
ENV_ARGS+=$(quote_env CHECKPOINT_MAX_EPOCH "${CHECKPOINT_MAX_EPOCH}")
ENV_ARGS+=$(quote_env CHECKPOINT_ORDER "${CHECKPOINT_ORDER}")
ENV_ARGS+=$(quote_env STABLE_SECONDS "${STABLE_SECONDS}")
ENV_ARGS+=$(quote_env RETRY_FAILED "${RETRY_FAILED}")
ENV_ARGS+=$(quote_env VAL6000_MAX_EVALS_PER_CYCLE "${VAL6000_MAX_EVALS_PER_CYCLE}")
ENV_ARGS+=$(quote_env NAVTEST_MAX_EVALS_PER_CYCLE "${NAVTEST_MAX_EVALS_PER_CYCLE}")
ENV_ARGS+=$(quote_env VAL6000_TOP_K "${VAL6000_TOP_K}")
ENV_ARGS+=$(quote_env NAVTEST_FROM_VAL_TOP_K "${NAVTEST_FROM_VAL_TOP_K}")
ENV_ARGS+=$(quote_env NAVTEST_BACKUP_TOP_K "${NAVTEST_BACKUP_TOP_K}")
ENV_ARGS+=$(quote_env VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST "${VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST}")
ENV_ARGS+=$(quote_env WAIT_FOR_FREE_GPUS "${REMOTE_WAIT_FOR_FREE_GPUS}")
ENV_ARGS+=$(quote_env GPU_LIST "${REMOTE_GPU_LIST}")
ENV_ARGS+=$(quote_env GPUS_PER_NODE "${REMOTE_GPUS_PER_NODE}")
ENV_ARGS+=$(quote_env ASYNC_PDM_WORKERS "${ASYNC_PDM_WORKERS}")
ENV_ARGS+=$(quote_env ASYNC_PDM_BACKEND "${ASYNC_PDM_BACKEND}")
ENV_ARGS+=$(quote_env PDM_EVAL_RUNNER "${PDM_EVAL_RUNNER}")
ENV_ARGS+=$(quote_env FAST_METRIC_CACHE_DIR "${FAST_METRIC_CACHE_DIR}")
ENV_ARGS+=$(quote_env MAX_SCENES "${MAX_SCENES}")
ENV_ARGS+=$(quote_env BACKUP_RANKED_CHECKPOINTS "${BACKUP_RANKED_CHECKPOINTS}")
ENV_ARGS+=$(quote_env BACKUP_ROOT "${BACKUP_ROOT}")

REMOTE_INNER="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && mkdir -p $(printf '%q' "${LOG_DIR}") $(printf '%q' "${STATE_DIR}") && { nohup setsid env ${ENV_ARGS} bash scripts/evaluation/run_psi_stage2_eval_once_queue.sh >$(printf '%q' "${REMOTE_LOG}") 2>&1 < /dev/null & echo \$! > $(printf '%q' "${REMOTE_PID_FILE}"); }"

log "remote start once queue log=${REMOTE_LOG}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${REMOTE_INNER}" | tee -a "${LAUNCH_LOG}"
else
  ssh_cmd "bash -lc $(printf '%q' "${REMOTE_INNER}")"
  log "remote once queue started pid_file=${REMOTE_PID_FILE}"
fi
