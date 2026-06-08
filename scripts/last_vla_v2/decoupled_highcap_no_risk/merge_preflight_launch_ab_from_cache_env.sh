#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CACHE_ENV="${CACHE_ENV:-/mnt/project/VLA-AD/cache/last_vla_v2/experiments/decoupled_highcap_no_risk/cache.env}"
if [[ ! -f "${CACHE_ENV}" ]]; then
  echo "Missing cache env: ${CACHE_ENV}" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${CACHE_ENV}"

BASE_CHUNK_ROOT="$(readlink -m "${BASE_CHUNK_ROOT}")"
JEPA_DENSE_CACHE_ROOT="$(readlink -m "${JEPA_DENSE_CACHE_ROOT}")"
GEOMETRY_CACHE_ROOT="$(readlink -m "${GEOMETRY_CACHE_ROOT}")"
GEOMETRY_SHARD_ROOT="$(readlink -m "${GEOMETRY_SHARD_ROOT}")"
FULL_HIGHCAP_TRAIN_CHUNK_ROOT="$(readlink -m "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}")"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/mnt/project/VLA-AD/outputs/last_vla_v2}"
ORCH_ROOT="${ORCH_ROOT:-${OUT_ROOT_BASE}/decoupled_highcap_no_risk_ab_launch_${RUN_ID}}"
LOG_DIR="${ORCH_ROOT}/logs"
STATUS_FILE="${ORCH_ROOT}/status.env"
mkdir -p "${LOG_DIR}"

REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
MASTER_PORT_A="${MASTER_PORT_A:-29631}"
MASTER_PORT_B="${MASTER_PORT_B:-29651}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
GEOMETRY_CHUNK_NAME_PATTERN="${GEOMETRY_CHUNK_NAME_PATTERN:-*}"
JEPA_CHUNK_NAME_PATTERN="${JEPA_CHUNK_NAME_PATTERN:-*}"
COPY_MODE="${COPY_MODE:-hardlink}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-/mnt/navsim/trainval_sensor_blobs/trainval}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
VLM_PATH="${VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

write_status() {
  {
    printf 'ORCH_ROOT=%q\n' "${ORCH_ROOT}"
    printf 'CACHE_ENV=%q\n' "${CACHE_ENV}"
    printf 'FULL_HIGHCAP_TRAIN_CHUNK_ROOT=%q\n' "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
    printf 'STATUS=%q\n' "$1"
    printf 'UPDATED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${STATUS_FILE}"
}

on_error() {
  local rc=$?
  log "failed rc=${rc}"
  write_status "FAILED_RC_${rc}"
  exit "${rc}"
}
trap on_error ERR

require_path() {
  if [[ ! -e "$1" ]]; then
    log "missing required path: $1"
    write_status "FAILED_MISSING_PATH"
    exit 2
  fi
}

merge_final_cache() {
  if [[ -e "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]]; then
    log "refusing to overwrite existing final train cache: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
    write_status "FAILED_TRAIN_CACHE_EXISTS"
    exit 3
  fi
  mkdir -p "$(dirname "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}")"
  log "merging final train cache: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py \
      --base-chunk-root "${BASE_CHUNK_ROOT}" \
      --geometry-cache-root "${GEOMETRY_CACHE_ROOT}" \
      --geometry-chunk-name-pattern "${GEOMETRY_CHUNK_NAME_PATTERN}" \
      --jepa-cache-root "${JEPA_DENSE_CACHE_ROOT}" \
      --jepa-chunk-name-pattern "${JEPA_CHUNK_NAME_PATTERN}" \
      --output-chunk-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
      --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}" \
      --copy-mode "${COPY_MODE}" \
      --overwrite-context \
      --strict-coverage \
      --min-coverage 0.99 \
      --strict-jepa-coverage \
      --min-jepa-coverage 0.99 \
      --expected-jepa-tokens 128 \
      --jepa-dim 1024 \
      --expected-vggt-context-tokens 128 \
      --vggt-dim 2048 \
      --num-geometry-tokens 192 \
      --geometry-grid-rows 12 \
      --geometry-grid-cols 16 \
      --geometry-teacher-dim 512
  ) >"${LOG_DIR}/merge_final_cache.log" 2>&1
}

audit_final_cache() {
  log "auditing final train cache"
  (
    cd "${REPO_ROOT}"
    "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py \
      --cache-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
      --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}" \
      --strict-full-geometry \
      --min-full-geometry-coverage 0.99 \
      --expected-jepa-tokens 128 \
      --expected-geometry-tokens 192 \
      --geometry-teacher-dim 512 \
      --strict-no-risk \
      --output "${ORCH_ROOT}/train_manifest.json"
  ) >"${LOG_DIR}/audit_final_cache.log" 2>&1
}

run_preflight() {
  log "running strict preflight"
  (
    cd "${REPO_ROOT}"
    RUN_PREFLIGHT=1 \
    FULL_HIGHCAP_TRAIN_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
    OUT_ROOT="${ORCH_ROOT}/preflight" \
    PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/run_strict_preflight_decoupled.sh
  ) >"${LOG_DIR}/preflight.log" 2>&1

  if ! grep -q '^Status: READY' "${ORCH_ROOT}/preflight/decoupled_highcap_no_risk_preflight/readiness.md"; then
    log "preflight did not report READY"
    write_status "FAILED_PREFLIGHT"
    exit 4
  fi
}

launch_a_local() {
  local a_out="${ORCH_ROOT}/A"
  local log_path="${LOG_DIR}/serverA_launcher.log"
  mkdir -p "${a_out}"
  log "launching A locally: ${a_out}"
  (
    cd "${REPO_ROOT}"
    setsid env \
      RUN_TRAIN=1 \
      FULL_HIGHCAP_TRAIN_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
      OUT_ROOT="${a_out}" \
      MASTER_PORT="${MASTER_PORT_A}" \
      NPROC_PER_NODE="${NPROC_PER_NODE}" \
      TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT}" \
      NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT}" \
      OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT}" \
      NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT}" \
      NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/serverA_frozen_vlm_decoupled_highcap_no_risk.sh \
      >"${log_path}" 2>&1 < /dev/null &
    echo "$!" >"${ORCH_ROOT}/serverA.pid"
  )
}

launch_b1_remote() {
  local b_out="${ORCH_ROOT}/B"
  local log_path="${LOG_DIR}/serverB_launcher.log"
  mkdir -p "${b_out}"
  log "launching B1 remotely on ${REMOTE_HOST}: ${b_out}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd '${REPO_ROOT}' && setsid env RUN_TRAIN=1 STOP_AFTER_EXTRACT=1 SKIP_VALIDATION=1 FULL_HIGHCAP_TRAIN_CHUNK_ROOT='${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' OUT_ROOT='${b_out}' MASTER_PORT='${MASTER_PORT_B}' NPROC_PER_NODE='${NPROC_PER_NODE}' TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' VLM_PATH='${VLM_PATH}' NAVSIM_LOG_PATH='${NAVSIM_LOG_PATH}' SENSOR_BLOBS_PATH='${SENSOR_BLOBS_PATH}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' PYTHON_BIN='${PYTHON_BIN}' bash scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh > '${log_path}' 2>&1 < /dev/null & echo \\$! > '${ORCH_ROOT}/serverB_remote.pid'"
}

launch_b_followup_remote() {
  local b_out="${ORCH_ROOT}/B"
  local log_path="${LOG_DIR}/serverB_b3_b4_followup.log"
  log "launching B3/B4 follow-up watcher remotely on ${REMOTE_HOST}: ${b_out}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd '${REPO_ROOT}' && setsid env FULL_HIGHCAP_TRAIN_CHUNK_ROOT='${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' OUT_ROOT='${b_out}' MASTER_PORT='$((MASTER_PORT_B + 2))' VLM_PATH='${VLM_PATH}' PYTHON_BIN='${PYTHON_BIN}' TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' SKIP_VALIDATION=1 NUM_GPUS=8 PROCS_PER_GPU=8 NUM_SHARDS=64 SKIP_HIDDEN_CACHE_AUDIT=1 WAIT_FOR_ADAPTERS=1 WAIT_SECONDS=1200 RUN_ID='${RUN_ID}_b3b4' bash scripts/last_vla_v2/decoupled_highcap_no_risk/run_lora_cache_and_progressive_decoupled.sh > '${log_path}' 2>&1 < /dev/null & echo \\$! > '${ORCH_ROOT}/serverB_b3_b4_followup_remote.pid'"
}

main() {
  log "orchestrator root: ${ORCH_ROOT}"
  write_status "CHECKING_INPUTS"
  require_path "${BASE_CHUNK_ROOT}"
  require_path "${JEPA_DENSE_CACHE_ROOT}"
  require_path "${GEOMETRY_CACHE_ROOT}"
  require_path "${NAVSIM_LOG_PATH}"
  require_path "${SENSOR_BLOBS_PATH}"
  require_path "${VLM_PATH}"

  write_status "MERGING_FINAL_CACHE"
  merge_final_cache
  write_status "AUDITING_FINAL_CACHE"
  audit_final_cache
  write_status "PREFLIGHT"
  run_preflight
  write_status "LAUNCHING_TRAINING"
  launch_a_local
  launch_b1_remote
  launch_b_followup_remote
  write_status "LAUNCHED"
  log "launched A, B1, and remote B3/B4 follow-up watcher. Status file: ${STATUS_FILE}"
}

main "$@"
