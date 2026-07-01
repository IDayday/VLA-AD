#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:?SUPPLEMENT_ROOT must point to stage2_gt_only_structured_elite_buffer_*}"
SUPPORT_ENV="${SUPPORT_ENV:-${SUPPLEMENT_ROOT}/state/supplemented_support_index.env}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STOP_EXISTING_AT_START="${STOP_EXISTING_AT_START:-false}"
STOP_EXISTING_BEFORE_LAUNCH="${STOP_EXISTING_BEFORE_LAUNCH:-true}"
EXISTING_PID_FILE="${EXISTING_PID_FILE:-}"

STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PSI_STAGE2_INIT_MODE="random"
export CACHE_TRAIN_ALL_RECORDS="${CACHE_TRAIN_ALL_RECORDS:-true}"
export MAX_EPOCHS="${MAX_EPOCHS:-200}"
export LR="${LR:-1e-4}"
export MASTER_PORT="${MASTER_PORT:-63719}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-psi_drive_stage2_apsd_random_init_full103k_gt_supp_${STAMP}}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/${EXPERIMENT_NAME}}"

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/state"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

stop_existing_training() {
  local pid_file="$1"
  [[ -n "${pid_file}" && -f "${pid_file}" ]] || return 0
  local pid
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] || return 0
  if ! ps -p "${pid}" >/dev/null 2>&1; then
    log "existing training already stopped pid=${pid}"
    return 0
  fi

  local -a tree_pids=("${pid}")
  local -a frontier=("${pid}")
  local -a next_frontier=()
  local child
  while [[ "${#frontier[@]}" -gt 0 ]]; do
    next_frontier=()
    for child in $(pgrep -P "$(IFS=,; echo "${frontier[*]}")" 2>/dev/null || true); do
      tree_pids+=("${child}")
      next_frontier+=("${child}")
    done
    frontier=("${next_frontier[@]}")
  done

  local self_pgid
  self_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  local -a pgids=()
  local tree_pid pgid already_seen
  for tree_pid in "${tree_pids[@]}"; do
    pgid="$(ps -o pgid= -p "${tree_pid}" 2>/dev/null | tr -d ' ')"
    [[ -n "${pgid}" ]] || continue
    already_seen=false
    for seen in "${pgids[@]:-}"; do
      if [[ "${seen}" == "${pgid}" ]]; then
        already_seen=true
        break
      fi
    done
    [[ "${already_seen}" == "true" ]] || pgids+=("${pgid}")
  done

  log "stopping existing training pid=${pid} tree_pids=${tree_pids[*]} pgids=${pgids[*]:-}"
  for pgid in "${pgids[@]:-}"; do
    if [[ "${pgid}" == "${self_pgid}" ]]; then
      continue
    fi
    kill -TERM "-${pgid}" || true
  done
  if [[ "${#tree_pids[@]}" -gt 0 ]]; then
    kill -TERM "${tree_pids[@]}" || true
  fi
  for _ in $(seq 1 60); do
    if ! ps -p "${pid}" >/dev/null 2>&1; then
      log "existing training stopped pid=${pid}"
      return 0
    fi
    sleep 2
  done
  log "existing training still alive; sending KILL to remaining tree"
  for pgid in "${pgids[@]:-}"; do
    if [[ "${pgid}" == "${self_pgid}" ]]; then
      continue
    fi
    kill -KILL "-${pgid}" || true
  done
  if [[ "${#tree_pids[@]}" -gt 0 ]]; then
    kill -KILL "${tree_pids[@]}" || true
  fi
}

if [[ "${STOP_EXISTING_AT_START}" == "true" ]]; then
  stop_existing_training "${EXISTING_PID_FILE}"
fi

log "waiting for supplemented support env: ${SUPPORT_ENV}"
while [[ ! -s "${SUPPORT_ENV}" ]]; do
  if [[ -s "${SUPPLEMENT_ROOT}/errors.jsonl" ]]; then
    log "supplement generation has errors: ${SUPPLEMENT_ROOT}/errors.jsonl"
    tail -n 20 "${SUPPLEMENT_ROOT}/errors.jsonl" >&2
    exit 2
  fi
  record_count="$(find "${SUPPLEMENT_ROOT}/buffer" -maxdepth 1 -type f -name '*.pkl.xz' 2>/dev/null | wc -l)"
  summary_count="$(find "${SUPPLEMENT_ROOT}/summaries" -maxdepth 1 -type f -name 'shard_*.json' 2>/dev/null | wc -l)"
  log "supplement not ready records=${record_count} summaries=${summary_count}"
  sleep "${POLL_SECONDS}"
done

# shellcheck disable=SC1090
source "${SUPPORT_ENV}"

if [[ -z "${SUPPORT_INDEX:-}" || ! -f "${SUPPORT_INDEX}" ]]; then
  log "invalid SUPPORT_INDEX from ${SUPPORT_ENV}: ${SUPPORT_INDEX:-<unset>}"
  exit 2
fi
if [[ "${STOP_EXISTING_BEFORE_LAUNCH}" == "true" ]]; then
  stop_existing_training "${EXISTING_PID_FILE}"
fi

export PSI_STAGE2_SUPPORT_INDEX="${SUPPORT_INDEX}"

log "launching random-init Stage2 with support index: ${PSI_STAGE2_SUPPORT_INDEX}"
log "output dir: ${OUTPUT_DIR}"
"${REPO_ROOT}/scripts/psi_drive/run_stage2_apsd_random_init_8gpu.sh" &
train_pid="$!"
echo "${train_pid}" > "${OUTPUT_DIR}/state/stage2_random_train.pid"
log "stage2 random-init launch pid=${train_pid}"
wait "${train_pid}"
