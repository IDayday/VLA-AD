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

LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"
TRANSITION_LOG="${LOG_DIR}/eval_priority_queue_transition.${REMOTE_HOST}.log"
TRANSITION_PID_FILE="${STATE_DIR}/eval_priority_queue_transition.${REMOTE_HOST}.pid"
REMOTE_SCRIPT="${STATE_DIR}/eval_priority_queue_transition.${REMOTE_HOST}.sh"
QUEUE_LOG="${LOG_DIR}/eval_once_queue.${REMOTE_HOST}.log"
QUEUE_PID_FILE="${STATE_DIR}/eval_once_queue.${REMOTE_HOST}.pid"

quote_env() {
  printf '%q=%q ' "$1" "$2"
}

ENV_ARGS=""
ENV_ARGS+=$(quote_env RUN_ROOT "${RUN_ROOT}")
ENV_ARGS+=$(quote_env CHECKPOINT_ROOT "${CHECKPOINT_ROOT}")
ENV_ARGS+=$(quote_env STAGE "stage2")
ENV_ARGS+=$(quote_env QUEUE_SPLITS "${QUEUE_SPLITS}")
ENV_ARGS+=$(quote_env QUEUE_FOREVER "${QUEUE_FOREVER}")
ENV_ARGS+=$(quote_env POLL_SECONDS "${POLL_SECONDS}")
ENV_ARGS+=$(quote_env CHECKPOINT_MIN_EPOCH "${CHECKPOINT_MIN_EPOCH}")
ENV_ARGS+=$(quote_env CHECKPOINT_MAX_EPOCH "${CHECKPOINT_MAX_EPOCH}")
ENV_ARGS+=$(quote_env CHECKPOINT_ORDER "${CHECKPOINT_ORDER}")
ENV_ARGS+=$(quote_env STABLE_SECONDS "${STABLE_SECONDS}")
ENV_ARGS+=$(quote_env RETRY_FAILED "${RETRY_FAILED}")
ENV_ARGS+=$(quote_env VAL6000_MAX_EVALS_PER_CYCLE "${VAL6000_MAX_EVALS_PER_CYCLE}")
ENV_ARGS+=$(quote_env NAVTEST_MAX_EVALS_PER_CYCLE "${NAVTEST_MAX_EVALS_PER_CYCLE}")
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

REMOTE_BODY=$(cat <<SH
#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=$(printf '%q' "${RUN_ROOT}")
REMOTE_REPO_ROOT=$(printf '%q' "${REMOTE_REPO_ROOT}")
QUEUE_LOG=$(printf '%q' "${QUEUE_LOG}")
QUEUE_PID_FILE=$(printf '%q' "${QUEUE_PID_FILE}")
TRANSITION_LOG=$(printf '%q' "${TRANSITION_LOG}")

log() {
  echo "\$(date -u +%Y-%m-%dT%H:%M:%SZ) \$*" | tee -a "\${TRANSITION_LOG}"
}

active_eval_pids() {
  for pid in \$(pgrep -f 'run_pdm_score_recogdrive|torchrun' || true); do
    [[ "\${pid}" != "\$\$" ]] || continue
    [[ -r "/proc/\${pid}/cmdline" ]] || continue
    cmdline="\$(tr '\0' ' ' < "/proc/\${pid}/cmdline")"
    if [[ "\${cmdline}" == *"\${RUN_ROOT}"* ]]; then
      printf '%s\n' "\${pid}"
    fi
  done
}

queue_pids() {
  for pid in \$(pgrep -f 'eval_once_queue|run_psi_stage2_eval_once_queue|watch_psi_stage2_checkpoint_eval_8gpu' || true); do
    [[ "\${pid}" != "\$\$" ]] || continue
    [[ -r "/proc/\${pid}/cmdline" ]] || continue
    cmdline="\$(tr '\0' ' ' < "/proc/\${pid}/cmdline")"
    if [[ "\${cmdline}" == *"\${RUN_ROOT}"* && "\${cmdline}" != *"eval_priority_queue_transition"* ]]; then
      printf '%s\n' "\${pid}"
    fi
  done
}

log "priority transition start run_root=\${RUN_ROOT}"
initial_pids="\$(active_eval_pids | sort -n || true)"
if [[ -n "\${initial_pids}" ]]; then
  log "waiting initial active eval pids=\$(echo "\${initial_pids}" | tr '\n' ',')"
fi
while pids="\$(active_eval_pids | sort -n || true)" && [[ -n "\${pids}" ]]; do
  new_pids="\$(comm -13 <(printf '%s\n' "\${initial_pids}") <(printf '%s\n' "\${pids}") || true)"
  if [[ -n "\${new_pids}" ]]; then
    log "detected race-started eval pids=\$(echo "\${new_pids}" | tr '\n' ',')"
    break
  fi
  sleep 30
done

old_pids="\$(queue_pids || true)"
if [[ -n "\${old_pids}" ]]; then
  log "stopping old queue/watch pids=\$(echo "\${old_pids}" | tr '\n' ',')"
  kill \${old_pids} 2>/dev/null || true
  sleep 3
fi

race_pids="\$(active_eval_pids || true)"
if [[ -n "\${race_pids}" ]]; then
  log "stopping race-started eval pids=\$(echo "\${race_pids}" | tr '\n' ',')"
  kill \${race_pids} 2>/dev/null || true
  sleep 5
  race_pids="\$(active_eval_pids || true)"
  if [[ -n "\${race_pids}" ]]; then
    kill -9 \${race_pids} 2>/dev/null || true
  fi
fi

if [[ -d "\${RUN_ROOT}/state" ]]; then
  find "\${RUN_ROOT}/state" -maxdepth 1 \\( -name 'stage2_val6000_*.running' -o -name 'stage2_navtest_*.running' \\) -delete
fi

log "starting continuous priority queue"
cd "\${REMOTE_REPO_ROOT}"
nohup setsid env ${ENV_ARGS} bash scripts/evaluation/run_psi_stage2_eval_once_queue.sh >"\${QUEUE_LOG}" 2>&1 < /dev/null &
echo "\$!" > "\${QUEUE_PID_FILE}"
log "priority queue started pid=\$(cat "\${QUEUE_PID_FILE}") log=\${QUEUE_LOG}"
SH
)

ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
  "mkdir -p $(printf '%q' "${LOG_DIR}") $(printf '%q' "${STATE_DIR}") && cat > $(printf '%q' "${REMOTE_SCRIPT}") && chmod +x $(printf '%q' "${REMOTE_SCRIPT}") && { nohup bash $(printf '%q' "${REMOTE_SCRIPT}") > $(printf '%q' "${TRANSITION_LOG}") 2>&1 < /dev/null & echo \$! > $(printf '%q' "${TRANSITION_PID_FILE}"); }" \
  <<< "${REMOTE_BODY}"

echo "transition_pid_file=${TRANSITION_PID_FILE}"
echo "transition_log=${TRANSITION_LOG}"
echo "queue_pid_file=${QUEUE_PID_FILE}"
echo "queue_log=${QUEUE_LOG}"
