#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the PSI Stage2 run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
STAGE="${STAGE:-stage2}"
EVAL_SPLIT="${EVAL_SPLIT:-val6000}"
EVAL_SUMMARY="${EVAL_SUMMARY:-${RUN_ROOT}/eval/${EVAL_SPLIT}/summary.tsv}"
EVAL_COMMAND_TEMPLATE="${EVAL_COMMAND_TEMPLATE:-}"
EXIT_ONCE="${EXIT_ONCE:-0}"

mkdir -p \
  "${RUN_ROOT}/checkpoints/raw" \
  "${RUN_ROOT}/checkpoints/val_loss_top5" \
  "${RUN_ROOT}/checkpoint_store/objects" \
  "${RUN_ROOT}/rankings/val6000/history" \
  "${RUN_ROOT}/rankings/navtest/history" \
  "${RUN_ROOT}/eval/val6000" \
  "${RUN_ROOT}/eval/navtest" \
  "${RUN_ROOT}/state" \
  "${RUN_ROOT}/logs"

exec 9>"${RUN_ROOT}/state/watch_${STAGE}_${EVAL_SPLIT}.lock"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another watcher holds the lock for ${RUN_ROOT}" | tee -a "${RUN_ROOT}/logs/watch_${STAGE}_${EVAL_SPLIT}.log"
  exit 0
fi

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${RUN_ROOT}/logs/watch_${STAGE}_${EVAL_SPLIT}.log"
}

seen_marker() {
  local ckpt="$1"
  local key
  key="$(printf '%s' "${ckpt}" | sha256sum | awk '{print $1}')"
  echo "${RUN_ROOT}/state/${STAGE}_${EVAL_SPLIT}_${key}.done"
}

run_eval_template() {
  local checkpoint="$1"
  local object_path="$2"
  local sha="$3"
  if [[ -z "${EVAL_COMMAND_TEMPLATE}" ]]; then
    return 0
  fi
  local cmd
  cmd="${EVAL_COMMAND_TEMPLATE//\{checkpoint\}/${checkpoint}}"
  cmd="${cmd//\{object\}/${object_path}}"
  cmd="${cmd//\{sha256\}/${sha}}"
  log "eval start checkpoint=${checkpoint}"
  bash -lc "${cmd}"
  log "eval done checkpoint=${checkpoint}"
}

process_checkpoint() {
  local ckpt="$1"
  local marker
  marker="$(seen_marker "${ckpt}")"
  [[ -f "${marker}" ]] && return 0
  [[ -s "${ckpt}" ]] || return 0
  if [[ $(( $(date +%s) - $(stat -c '%Y' "${ckpt}") )) -lt "${STABLE_SECONDS}" ]]; then
    return 0
  fi
  local archive_out sha object_path
  archive_out="$("${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/archive_checkpoint_immutable.py" \
    --run-root "${RUN_ROOT}" \
    --checkpoint "${ckpt}" \
    --stage "${STAGE}" \
    --stable-seconds 0)"
  log "${archive_out}"
  sha="$(printf '%s\n' "${archive_out}" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p' | tail -1)"
  object_path="${RUN_ROOT}/checkpoint_store/objects/${sha}.ckpt"
  run_eval_template "${ckpt}" "${object_path}" "${sha}"
  if [[ -f "${EVAL_SUMMARY}" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/rank_eval_checkpoints.py" \
      --run-root "${RUN_ROOT}" \
      --split "${EVAL_SPLIT}" \
      --eval-summary "${EVAL_SUMMARY}" \
      || log "ranking refresh failed for ${EVAL_SPLIT}; see ${EVAL_SUMMARY}"
  fi
  touch "${marker}"
}

while true; do
  if [[ -d "${CHECKPOINT_ROOT}" ]]; then
    while IFS= read -r -d '' ckpt; do
      process_checkpoint "${ckpt}"
    done < <(find "${CHECKPOINT_ROOT}" -type f -name '*.ckpt' -print0 | sort -z)
  else
    log "waiting for checkpoint root: ${CHECKPOINT_ROOT}"
  fi
  if [[ "${EXIT_ONCE}" == "1" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}" 9>&-
done
