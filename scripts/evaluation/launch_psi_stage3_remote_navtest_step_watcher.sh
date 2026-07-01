#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-training-rl-zt3}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage3 PSI run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"

POLL_SECONDS="${POLL_SECONDS:-120}"
STABLE_SECONDS="${STABLE_SECONDS:-180}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-asc}"
CHECKPOINT_EPOCH_ONLY="${CHECKPOINT_EPOCH_ONLY:-0}"
CHECKPOINT_STEP_ONLY="${CHECKPOINT_STEP_ONLY:-1}"
EVAL_MIN_CHECKPOINT_STEP="${EVAL_MIN_CHECKPOINT_STEP:-300}"
EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-300}"
RETRY_FAILED="${RETRY_FAILED:-1}"
ARCHIVE_ALL_BEFORE_EVAL="${ARCHIVE_ALL_BEFORE_EVAL:-1}"
MAX_EVALS_PER_POLL="${MAX_EVALS_PER_POLL:-1}"

REMOTE_WAIT_FOR_FREE_GPUS="${REMOTE_WAIT_FOR_FREE_GPUS:-1}"
REMOTE_GPU_LIST="${REMOTE_GPU_LIST:-0,1,2,3,4,5,6,7}"
REMOTE_GPUS_PER_NODE="${REMOTE_GPUS_PER_NODE:-8}"
REMOTE_GPU_MAX_MEM_USED_MB="${REMOTE_GPU_MAX_MEM_USED_MB:-5000}"
REMOTE_GPU_MAX_UTIL="${REMOTE_GPU_MAX_UTIL:-101}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle}"
MAX_SCENES="${MAX_SCENES:-0}"
BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS:-1}"
BACKUP_TOP_K="${BACKUP_TOP_K:-5}"
BACKUP_ROOT="${BACKUP_ROOT:-${RUN_ROOT}/checkpoint_backups}"
EVAL_HOST_ID="${EVAL_HOST_ID:-${REMOTE_HOST}}"
WATCH_LOCK_SCOPE="${WATCH_LOCK_SCOPE:-global}"
EVAL_RESOURCE_LOCK_SCOPE="${EVAL_RESOURCE_LOCK_SCOPE:-global}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"
LAUNCH_LOG="${LOG_DIR}/remote_stage3_navtest_step_watcher_launch_${REMOTE_HOST}.log"
REMOTE_LOG="${REMOTE_LOG:-${LOG_DIR}/watch_stage3_eval_navtest_step_all.${REMOTE_HOST}.log}"
REMOTE_PID_FILE="${STATE_DIR}/watch_stage3_eval_navtest_step_all.${REMOTE_HOST}.pid"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LAUNCH_LOG}"
}

ssh_cmd() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "$@"
}

quote_env() {
  printf '%q=%q ' "$1" "$2"
}

log "remote Stage3 navtest step watcher launcher start host=${REMOTE_HOST} run_root=${RUN_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  ssh_cmd "test -d $(printf '%q' "${REMOTE_REPO_ROOT}") && test -d $(printf '%q' "${RUN_ROOT}") && hostname && date -u +%Y-%m-%dT%H:%M:%SZ" \
    | sed 's/^/remote_probe: /' | tee -a "${LAUNCH_LOG}"
fi

ENV_ARGS=""
ENV_ARGS+=$(quote_env RUN_ROOT "${RUN_ROOT}")
ENV_ARGS+=$(quote_env CHECKPOINT_ROOT "${CHECKPOINT_ROOT}")
ENV_ARGS+=$(quote_env STAGE "stage3")
ENV_ARGS+=$(quote_env EVAL_SPLIT "navtest")
ENV_ARGS+=$(quote_env EVAL_SCRIPT "${REMOTE_REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh")
ENV_ARGS+=$(quote_env POLL_SECONDS "${POLL_SECONDS}")
ENV_ARGS+=$(quote_env STABLE_SECONDS "${STABLE_SECONDS}")
ENV_ARGS+=$(quote_env RETRY_FAILED "${RETRY_FAILED}")
ENV_ARGS+=$(quote_env CHECKPOINT_ORDER "${CHECKPOINT_ORDER}")
ENV_ARGS+=$(quote_env CHECKPOINT_EPOCH_ONLY "${CHECKPOINT_EPOCH_ONLY}")
ENV_ARGS+=$(quote_env CHECKPOINT_STEP_ONLY "${CHECKPOINT_STEP_ONLY}")
ENV_ARGS+=$(quote_env EVAL_MIN_CHECKPOINT_STEP "${EVAL_MIN_CHECKPOINT_STEP}")
ENV_ARGS+=$(quote_env EVAL_CHECKPOINT_STEP_INTERVAL "${EVAL_CHECKPOINT_STEP_INTERVAL}")
ENV_ARGS+=$(quote_env ARCHIVE_ALL_BEFORE_EVAL "${ARCHIVE_ALL_BEFORE_EVAL}")
ENV_ARGS+=$(quote_env MAX_EVALS_PER_POLL "${MAX_EVALS_PER_POLL}")
ENV_ARGS+=$(quote_env EVAL_HOST_ID "${EVAL_HOST_ID}")
ENV_ARGS+=$(quote_env WATCH_LOCK_SCOPE "${WATCH_LOCK_SCOPE}")
ENV_ARGS+=$(quote_env EVAL_RESOURCE_LOCK_SCOPE "${EVAL_RESOURCE_LOCK_SCOPE}")
ENV_ARGS+=$(quote_env WAIT_FOR_FREE_GPUS "${REMOTE_WAIT_FOR_FREE_GPUS}")
ENV_ARGS+=$(quote_env GPU_LIST "${REMOTE_GPU_LIST}")
ENV_ARGS+=$(quote_env GPUS_PER_NODE "${REMOTE_GPUS_PER_NODE}")
ENV_ARGS+=$(quote_env GPU_MAX_MEM_USED_MB "${REMOTE_GPU_MAX_MEM_USED_MB}")
ENV_ARGS+=$(quote_env GPU_MAX_UTIL "${REMOTE_GPU_MAX_UTIL}")
ENV_ARGS+=$(quote_env ASYNC_PDM_WORKERS "${ASYNC_PDM_WORKERS}")
ENV_ARGS+=$(quote_env ASYNC_PDM_BACKEND "${ASYNC_PDM_BACKEND}")
ENV_ARGS+=$(quote_env PDM_EVAL_RUNNER "${PDM_EVAL_RUNNER}")
ENV_ARGS+=$(quote_env FAST_METRIC_CACHE_DIR "${FAST_METRIC_CACHE_DIR}")
ENV_ARGS+=$(quote_env MAX_SCENES "${MAX_SCENES}")
ENV_ARGS+=$(quote_env BACKUP_RANKED_CHECKPOINTS "${BACKUP_RANKED_CHECKPOINTS}")
ENV_ARGS+=$(quote_env BACKUP_TOP_K "${BACKUP_TOP_K}")
ENV_ARGS+=$(quote_env BACKUP_ROOT "${BACKUP_ROOT}")

REMOTE_INNER="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && mkdir -p $(printf '%q' "${LOG_DIR}") $(printf '%q' "${STATE_DIR}") && { nohup setsid env ${ENV_ARGS} bash scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh >$(printf '%q' "${REMOTE_LOG}") 2>&1 < /dev/null & echo \$! > $(printf '%q' "${REMOTE_PID_FILE}"); }"

log "remote start Stage3 navtest step watcher log=${REMOTE_LOG}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${REMOTE_INNER}" | tee -a "${LAUNCH_LOG}"
else
  ssh_cmd "bash -lc $(printf '%q' "${REMOTE_INNER}")"
  log "remote Stage3 navtest step watcher started pid_file=${REMOTE_PID_FILE}"
fi
