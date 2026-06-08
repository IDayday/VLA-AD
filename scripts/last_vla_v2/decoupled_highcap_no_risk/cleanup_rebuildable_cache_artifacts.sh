#!/usr/bin/env bash
set -Eeuo pipefail

# Conservative cleanup helper for Last-VLA v2 cache/output sprawl.
# Default mode is dry-run. Set EXECUTE=1 to remove listed artifacts.

EXECUTE="${EXECUTE:-0}"
INCLUDE_READY_REBUILDABLE="${INCLUDE_READY_REBUILDABLE:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
FINAL_CACHE="${FINAL_CACHE:-${PROJECT_ROOT}/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks}"
READY_REPORT="${READY_REPORT:-}"

safe_now=(
  "${PROJECT_ROOT}/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/cache"
  "${PROJECT_ROOT}/outputs/last_vla_v2/highcap_no_risk_navtest_eval_20260606T034658Z/checkpoint_snapshot"
  "${PROJECT_ROOT}/cache/last_vla_v2/decoupled_highcap_no_risk/test_vggt128_fp32_20260606T191047Z"
)

ready_rebuildable=(
  "${PROJECT_ROOT}/cache/last_vla_v2/decoupled_highcap_no_risk/train_geometry192_overlay_raw"
)

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

size_of() {
  if [[ -e "$1" ]]; then
    du -sh "$1" 2>/dev/null | awk '{print $1}'
  else
    printf 'missing'
  fi
}

final_cache_ready() {
  [[ -f "${FINAL_CACHE}/merge_summary.json" ]] || return 1
  if [[ -n "${READY_REPORT}" ]]; then
    grep -q '^Status: READY' "${READY_REPORT}" || return 1
  fi
  return 0
}

remove_or_print() {
  local path="$1"
  local reason="$2"
  local size
  size="$(size_of "${path}")"
  if [[ "${size}" == "missing" ]]; then
    log "skip missing path=${path}"
    return 0
  fi
  if [[ "${EXECUTE}" == "1" ]]; then
    log "removing size=${size} reason=${reason} path=${path}"
    rm -rf -- "${path}"
  else
    log "dry-run size=${size} reason=${reason} path=${path}"
  fi
}

main() {
  log "mode=$([[ "${EXECUTE}" == "1" ]] && printf execute || printf dry-run)"
  for path in "${safe_now[@]}"; do
    remove_or_print "${path}" "old_rebuildable_or_low_value_artifact"
  done

  if [[ "${INCLUDE_READY_REBUILDABLE}" == "1" ]]; then
    if ! final_cache_ready; then
      log "refusing ready-gated cleanup because final cache is not proven ready: ${FINAL_CACHE}"
      exit 3
    fi
    for path in "${ready_rebuildable[@]}"; do
      remove_or_print "${path}" "rebuildable_after_final_cache_ready"
    done
  else
    log "ready-gated cleanup disabled; set INCLUDE_READY_REBUILDABLE=1 after final cache/preflight READY."
  fi
}

main "$@"
