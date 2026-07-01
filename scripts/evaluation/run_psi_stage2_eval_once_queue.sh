#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage2 PSI run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"
STAGE="${STAGE:-stage2}"

QUEUE_SPLITS="${QUEUE_SPLITS:-val6000 navtest}"
QUEUE_FOREVER="${QUEUE_FOREVER:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_HOST_ID="${EVAL_HOST_ID:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo local)}"
WATCH_LOCK_SCOPE="${WATCH_LOCK_SCOPE:-global}"
EVAL_RESOURCE_LOCK_SCOPE="${EVAL_RESOURCE_LOCK_SCOPE:-global}"
CHECKPOINT_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH:-50}"
CHECKPOINT_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH:-0}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-desc}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
RETRY_FAILED="${RETRY_FAILED:-1}"

VAL6000_TOP_K="${VAL6000_TOP_K:-5}"
NAVTEST_FROM_VAL_TOP_K="${NAVTEST_FROM_VAL_TOP_K:-5}"
NAVTEST_BACKUP_TOP_K="${NAVTEST_BACKUP_TOP_K:-3}"
VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST="${VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST:-0}"
VAL6000_MAX_EVALS_PER_CYCLE="${VAL6000_MAX_EVALS_PER_CYCLE:-0}"
NAVTEST_MAX_EVALS_PER_CYCLE="${NAVTEST_MAX_EVALS_PER_CYCLE:-0}"

WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-0}"
GPU_LIST="${GPU_LIST:-1,2,3,4,5,6}"
GPUS_PER_NODE="${GPUS_PER_NODE:-6}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
MAX_SCENES="${MAX_SCENES:-0}"
BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS:-1}"
BACKUP_ROOT="${BACKUP_ROOT:-${RUN_ROOT}/checkpoint_backups}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state"
QUEUE_LOG="${QUEUE_LOG:-${RUN_ROOT}/logs/eval_once_queue.log}"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${QUEUE_LOG}"
}

run_watcher_once() {
  local split="$1"
  local max_evals="$2"
  shift 2
  log "queue split start split=${split} max_evals=${max_evals}"
  env \
    RUN_ROOT="${RUN_ROOT}" \
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    STAGE="${STAGE}" \
    EVAL_HOST_ID="${EVAL_HOST_ID}" \
    WATCH_LOCK_SCOPE="${WATCH_LOCK_SCOPE}" \
    EVAL_RESOURCE_LOCK_SCOPE="${EVAL_RESOURCE_LOCK_SCOPE}" \
    EVAL_SPLIT="${split}" \
    EXIT_ONCE=1 \
    POLL_SECONDS=0 \
    STABLE_SECONDS="${STABLE_SECONDS}" \
    RETRY_FAILED="${RETRY_FAILED}" \
    CHECKPOINT_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH}" \
    CHECKPOINT_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH}" \
    CHECKPOINT_ORDER="${CHECKPOINT_ORDER}" \
    RANK_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH}" \
    RANK_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH}" \
    ARCHIVE_ALL_BEFORE_EVAL=0 \
    MAX_EVALS_PER_POLL="${max_evals}" \
    WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS}" \
    GPU_LIST="${GPU_LIST}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}" \
    ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}" \
    PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER}" \
    FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR}" \
    MAX_SCENES="${MAX_SCENES}" \
    BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS}" \
    BACKUP_ROOT="${BACKUP_ROOT}" \
    "$@" \
    bash "${REPO_ROOT}/scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh"
  log "queue split done split=${split}"
}

refresh_rank() {
  local split="$1"
  local top_k="$2"
  local summary_tsv="${RUN_ROOT}/eval/${split}/summary.tsv"
  if [[ ! -f "${summary_tsv}" ]]; then
    log "ranking skipped split=${split}; missing ${summary_tsv}"
    return 0
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/rank_eval_checkpoints.py" \
    --run-root "${RUN_ROOT}" \
    --split "${split}" \
    --eval-summary "${summary_tsv}" \
    --top-k "${top_k}" \
    --min-epoch "${CHECKPOINT_MIN_EPOCH}" \
    --max-epoch "${CHECKPOINT_MAX_EPOCH}" >> "${QUEUE_LOG}" 2>&1
}

backup_rank() {
  local split="$1"
  local top_k="$2"
  if [[ "${BACKUP_RANKED_CHECKPOINTS}" != "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/backup_ranked_checkpoints.py" \
    --run-root "${RUN_ROOT}" \
    --split "${split}" \
    --top-k "${top_k}" \
    --backup-root "${BACKUP_ROOT}" >> "${QUEUE_LOG}" 2>&1 || \
    log "backup skipped or failed split=${split} top_k=${top_k}"
}

log "eval queue start run_root=${RUN_ROOT} checkpoint_root=${CHECKPOINT_ROOT} splits=${QUEUE_SPLITS} min_epoch=${CHECKPOINT_MIN_EPOCH} order=${CHECKPOINT_ORDER} forever=${QUEUE_FOREVER} poll_seconds=${POLL_SECONDS} host=${EVAL_HOST_ID} watch_lock_scope=${WATCH_LOCK_SCOPE} eval_resource_lock_scope=${EVAL_RESOURCE_LOCK_SCOPE}"

cycle=0
while true; do
  cycle=$((cycle + 1))
  log "eval queue cycle start cycle=${cycle}"
  for split in ${QUEUE_SPLITS}; do
    case "${split}" in
      val6000)
        run_watcher_once val6000 "${VAL6000_MAX_EVALS_PER_CYCLE}" "BACKUP_TOP_K=${VAL6000_TOP_K}"
        refresh_rank val6000 "${VAL6000_TOP_K}"
        backup_rank val6000 "${VAL6000_TOP_K}"
        ;;
      navtest)
        refresh_rank val6000 "${VAL6000_TOP_K}"
        navtest_list="${RUN_ROOT}/rankings/val6000/current_top${NAVTEST_FROM_VAL_TOP_K}.tsv"
        navtest_gate_args=()
        if [[ "${VAL6000_REQUIRE_COMPLETE_BEFORE_NAVTEST}" == "1" ]]; then
          navtest_gate_args=(
            "WAIT_FOR_STATUS_TSV=${RUN_ROOT}/eval/val6000/checkpoint_eval_status.tsv"
            "WAIT_FOR_STATUS_LABEL=val6000_epoch_range"
            "WAIT_FOR_STATUS_FAIL_ON_FAILED=1"
            "WAIT_FOR_STATUS_MIN_EPOCH=${CHECKPOINT_MIN_EPOCH}"
            "WAIT_FOR_STATUS_MAX_EPOCH=${CHECKPOINT_MAX_EPOCH}"
          )
        fi
        run_watcher_once navtest "${NAVTEST_MAX_EVALS_PER_CYCLE}" \
          "CHECKPOINT_LIST_TSV=${navtest_list}" \
          "CHECKPOINT_LIST_COLUMN=object_path" \
          "CHECKPOINT_LIST_TOP_K=${NAVTEST_FROM_VAL_TOP_K}" \
          "CHECKPOINT_LIST_MIN_ROWS=${NAVTEST_FROM_VAL_TOP_K}" \
          "${navtest_gate_args[@]}" \
          "BACKUP_TOP_K=${NAVTEST_BACKUP_TOP_K}"
        refresh_rank navtest "${NAVTEST_FROM_VAL_TOP_K}"
        backup_rank navtest "${NAVTEST_BACKUP_TOP_K}"
        ;;
      *)
        log "unknown split in QUEUE_SPLITS=${split}"
        exit 2
        ;;
    esac
  done
  log "eval queue cycle complete cycle=${cycle}"
  if [[ "${QUEUE_FOREVER}" != "1" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "eval queue complete run_root=${RUN_ROOT}"
