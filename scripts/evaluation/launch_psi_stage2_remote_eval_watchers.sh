#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-training-rl-zt3}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage2 PSI run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-200}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
CHECKPOINT_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH:-0}"
CHECKPOINT_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH:-0}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-asc}"
ARCHIVE_ALL_BEFORE_EVAL="${ARCHIVE_ALL_BEFORE_EVAL:-1}"
MAX_EVALS_PER_POLL="${MAX_EVALS_PER_POLL:-0}"
REMOTE_WAIT_FOR_FREE_GPUS="${REMOTE_WAIT_FOR_FREE_GPUS:-1}"
REMOTE_GPU_LIST="${REMOTE_GPU_LIST:-0,1,2,3,4,5,6,7}"
REMOTE_GPUS_PER_NODE="${REMOTE_GPUS_PER_NODE:-8}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
MAX_SCENES="${MAX_SCENES:-0}"
BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS:-1}"
BACKUP_ROOT="${BACKUP_ROOT:-${RUN_ROOT}/checkpoint_backups}"
STOP_LOCAL_WATCHERS="${STOP_LOCAL_WATCHERS:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"
LAUNCH_LOG="${LOG_DIR}/remote_eval_launch_${REMOTE_HOST}.log"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LAUNCH_LOG}"
}

ssh_probe() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "test -d $(printf '%q' "${REMOTE_REPO_ROOT}") && test -d $(printf '%q' "${RUN_ROOT}") && hostname && date -u +%Y-%m-%dT%H:%M:%SZ"
}

stop_local_watcher() {
  local split="$1"
  local pid_file="${STATE_DIR}/watch_stage2_eval_${split}.pid"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    log "stopping local ${split} watcher pid=${pid}"
    kill "${pid}" || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
      log "local ${split} watcher pid=${pid} still alive after SIGTERM"
    else
      log "local ${split} watcher stopped pid=${pid}"
    fi
  fi
}

remote_start() {
  local name="$1"
  shift
  local remote_log="${LOG_DIR}/${name}.${REMOTE_HOST}.log"
  local remote_pid_file="${STATE_DIR}/${name}.${REMOTE_HOST}.pid"
  local cmd inner
  inner="cd $(printf '%q' "${REMOTE_REPO_ROOT}") && mkdir -p $(printf '%q' "${LOG_DIR}") $(printf '%q' "${STATE_DIR}") && setsid -f env $* >$(printf '%q' "${remote_log}") 2>&1 < /dev/null; sleep 0.5; pgrep -n -f 'bash scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh' || true"
  cmd="bash -lc $(printf '%q' "${inner}")"
  log "remote start ${name} on ${REMOTE_HOST}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${inner}" | tee -a "${LAUNCH_LOG}"
    return 0
  fi
  local pid
  pid="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "${cmd}")"
  printf 'host=%s\npid=%s\nstarted_at=%s\nlog=%s\n' \
    "${REMOTE_HOST}" "${pid}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${remote_log}" > "${remote_pid_file}"
  log "remote started ${name} host=${REMOTE_HOST} pid=${pid} log=${remote_log}"
}

log "remote eval launcher start host=${REMOTE_HOST} run_root=${RUN_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  ssh_probe | sed 's/^/remote_probe: /' | tee -a "${LAUNCH_LOG}"
fi

if [[ "${STOP_LOCAL_WATCHERS}" == "1" ]]; then
  stop_local_watcher val6000
  stop_local_watcher navtest
fi

COMMON_ENV=(
  "RUN_ROOT=$(printf '%q' "${RUN_ROOT}")"
  "CHECKPOINT_ROOT=$(printf '%q' "${CHECKPOINT_ROOT}")"
  "STAGE=stage2"
  "POLL_SECONDS=$(printf '%q' "${POLL_SECONDS}")"
  "STABLE_SECONDS=$(printf '%q' "${STABLE_SECONDS}")"
  "CHECKPOINT_MIN_EPOCH=$(printf '%q' "${CHECKPOINT_MIN_EPOCH}")"
  "CHECKPOINT_MAX_EPOCH=$(printf '%q' "${CHECKPOINT_MAX_EPOCH}")"
  "CHECKPOINT_ORDER=$(printf '%q' "${CHECKPOINT_ORDER}")"
  "ARCHIVE_ALL_BEFORE_EVAL=$(printf '%q' "${ARCHIVE_ALL_BEFORE_EVAL}")"
  "MAX_EVALS_PER_POLL=$(printf '%q' "${MAX_EVALS_PER_POLL}")"
  "RANK_MIN_EPOCH=$(printf '%q' "${CHECKPOINT_MIN_EPOCH}")"
  "RANK_MAX_EPOCH=$(printf '%q' "${CHECKPOINT_MAX_EPOCH}")"
  "WAIT_FOR_FREE_GPUS=$(printf '%q' "${REMOTE_WAIT_FOR_FREE_GPUS}")"
  "GPU_LIST=$(printf '%q' "${REMOTE_GPU_LIST}")"
  "GPUS_PER_NODE=$(printf '%q' "${REMOTE_GPUS_PER_NODE}")"
  "ASYNC_PDM_WORKERS=$(printf '%q' "${ASYNC_PDM_WORKERS}")"
  "ASYNC_PDM_BACKEND=$(printf '%q' "${ASYNC_PDM_BACKEND}")"
  "PDM_EVAL_RUNNER=$(printf '%q' "${PDM_EVAL_RUNNER}")"
  "FAST_METRIC_CACHE_DIR=$(printf '%q' "${FAST_METRIC_CACHE_DIR}")"
  "MAX_SCENES=$(printf '%q' "${MAX_SCENES}")"
  "BACKUP_RANKED_CHECKPOINTS=$(printf '%q' "${BACKUP_RANKED_CHECKPOINTS}")"
  "BACKUP_ROOT=$(printf '%q' "${BACKUP_ROOT}")"
)

remote_start \
  watch_stage2_eval_val6000 \
  "${COMMON_ENV[@]}" \
  "EVAL_SPLIT=val6000" \
  "BACKUP_TOP_K=5" \
  "bash scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh"

remote_start \
  watch_stage2_eval_navtest_top5 \
  "${COMMON_ENV[@]}" \
  "EVAL_SPLIT=navtest" \
  "CHECKPOINT_LIST_TSV=$(printf '%q' "${RUN_ROOT}/rankings/val6000/current_top5.tsv")" \
  "CHECKPOINT_LIST_COLUMN=object_path" \
  "CHECKPOINT_LIST_TOP_K=5" \
  "CHECKPOINT_LIST_MIN_ROWS=5" \
  "BACKUP_TOP_K=3" \
  "bash scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh"

log "remote eval launcher complete"
