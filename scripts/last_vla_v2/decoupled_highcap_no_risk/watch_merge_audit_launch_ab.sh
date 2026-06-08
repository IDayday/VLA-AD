#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
POLL_SECONDS="${POLL_SECONDS:-1200}"
EXPECTED_NUM_SHARDS="${EXPECTED_NUM_SHARDS:-32}"

BASE_CHUNK_ROOT="${BASE_CHUNK_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
CACHE_OUTPUT_ROOT="${CACHE_OUTPUT_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2}"
JEPA_DENSE_CACHE_ROOT="${JEPA_DENSE_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/highcap_no_risk/train_jepa128_overlay}"
FULL_HIGHCAP_TRAIN_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-${CACHE_OUTPUT_ROOT}/decoupled_highcap_no_risk/train_full_highcap_chunks}"
GEOMETRY_SHARD_ROOT="${GEOMETRY_SHARD_ROOT:-${CACHE_OUTPUT_ROOT}/decoupled_highcap_no_risk/train_geometry192_overlay_raw/shards}"
TRAIN_GEOMETRY_ROOT="${CACHE_OUTPUT_ROOT}/decoupled_highcap_no_risk/train_geometry192_overlay"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/mnt/project/VLA-AD/outputs/last_vla_v2}"
ORCH_ROOT="${ORCH_ROOT:-${OUT_ROOT_BASE}/decoupled_highcap_no_risk_ab_launch_${RUN_ID}}"
LOG_DIR="${ORCH_ROOT}/logs"
STATUS_FILE="${ORCH_ROOT}/status.env"
mkdir -p "${LOG_DIR}"

MASTER_PORT_A="${MASTER_PORT_A:-29631}"
MASTER_PORT_B="${MASTER_PORT_B:-29651}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
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

count_metadata() {
  find "${GEOMETRY_SHARD_ROOT}" -maxdepth 2 -name metadata.json 2>/dev/null | wc -l
}

count_samples() {
  find "${GEOMETRY_SHARD_ROOT}" -maxdepth 3 -type f -name '*.pt' 2>/dev/null | wc -l
}

count_remote_geometry_procs() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "ps -C python -o args= | grep -c scripts/build_last_vla_full_geometry_cache.py || true" 2>/dev/null | tail -1
}

write_status() {
  {
    printf 'ORCH_ROOT=%q\n' "${ORCH_ROOT}"
    printf 'FULL_HIGHCAP_TRAIN_CHUNK_ROOT=%q\n' "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
    printf 'STATUS=%q\n' "$1"
    printf 'UPDATED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${STATUS_FILE}"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    log "missing required path: $1"
    write_status "FAILED_MISSING_PATH"
    exit 2
  fi
}

wait_for_shards() {
  while true; do
    local meta samples localp remotep
    meta="$(count_metadata)"
    samples="$(count_samples)"
    localp="$(ps -C python -o args= | grep -c scripts/build_last_vla_full_geometry_cache.py || true)"
    remotep="$(count_remote_geometry_procs || true)"
    log "cache status: samples=${samples} metadata=${meta}/${EXPECTED_NUM_SHARDS} local_procs=${localp} remote_procs=${remotep:-unknown}"
    if [[ "${meta}" == "${EXPECTED_NUM_SHARDS}" ]]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

merge_and_audit() {
  if [[ -e "${TRAIN_GEOMETRY_ROOT}" || -e "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]]; then
    log "refusing to overwrite existing merge output: ${TRAIN_GEOMETRY_ROOT} or ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
    write_status "FAILED_OUTPUT_EXISTS"
    exit 3
  fi

  log "running merge and strict audit"
  (
    cd "${REPO_ROOT}"
    RUN_CACHE=1 \
    BASE_CHUNK_ROOT="${BASE_CHUNK_ROOT}" \
    OUTPUT_ROOT="${CACHE_OUTPUT_ROOT}" \
    NUM_SHARDS="${EXPECTED_NUM_SHARDS}" \
    EXPECTED_NUM_SHARDS="${EXPECTED_NUM_SHARDS}" \
    SKIP_BUILD_SHARDS=1 \
    MERGE_SHARDS=1 \
    JEPA_DENSE_CACHE_ROOT="${JEPA_DENSE_CACHE_ROOT}" \
    ALLOW_FULL_CACHE_WITHOUT_MAX=1 \
    COPY_MODE=hardlink \
    PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh
  ) >"${LOG_DIR}/merge_audit.log" 2>&1

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
  log "launching B1 LoRA alignment remotely on ${REMOTE_HOST}: ${b_out}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd '${REPO_ROOT}' && setsid env RUN_TRAIN=1 STOP_AFTER_EXTRACT=1 SKIP_VALIDATION=1 FULL_HIGHCAP_TRAIN_CHUNK_ROOT='${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' OUT_ROOT='${b_out}' MASTER_PORT='${MASTER_PORT_B}' NPROC_PER_NODE='${NPROC_PER_NODE}' TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' VLM_PATH='${VLM_PATH}' NAVSIM_LOG_PATH='${NAVSIM_LOG_PATH}' SENSOR_BLOBS_PATH='${SENSOR_BLOBS_PATH}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' PYTHON_BIN='${PYTHON_BIN}' bash scripts/last_vla_v2/decoupled_highcap_no_risk/serverB_vlm_lora_decoupled_highcap_no_risk.sh > '${log_path}' 2>&1 < /dev/null & echo \\$! > '${ORCH_ROOT}/serverB_remote.pid'"
}

launch_b_followup_remote() {
  local b_out="${ORCH_ROOT}/B"
  local log_path="${LOG_DIR}/serverB_b3_b4_followup.log"
  log "launching remote B3/B4 follow-up watcher on ${REMOTE_HOST}: ${b_out}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd '${REPO_ROOT}' && setsid env FULL_HIGHCAP_TRAIN_CHUNK_ROOT='${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' OUT_ROOT='${b_out}' MASTER_PORT='$((MASTER_PORT_B + 2))' VLM_PATH='${VLM_PATH}' PYTHON_BIN='${PYTHON_BIN}' TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' NAVSIM_DATA_ROOT='${NAVSIM_DATA_ROOT}' OPENSCENE_DATA_ROOT='${OPENSCENE_DATA_ROOT}' NUPLAN_MAPS_ROOT='${NUPLAN_MAPS_ROOT}' NAVSIM_EXP_ROOT='${NAVSIM_EXP_ROOT}' SKIP_VALIDATION=1 NUM_GPUS=8 PROCS_PER_GPU=8 NUM_SHARDS=64 SKIP_HIDDEN_CACHE_AUDIT=1 WAIT_FOR_ADAPTERS=1 WAIT_SECONDS=1200 RUN_ID='${RUN_ID}_b3b4' bash scripts/last_vla_v2/decoupled_highcap_no_risk/run_lora_cache_and_progressive_decoupled.sh > '${log_path}' 2>&1 < /dev/null & echo \\$! > '${ORCH_ROOT}/serverB_b3_b4_followup_remote.pid'"
}

main() {
  log "orchestrator root: ${ORCH_ROOT}"
  write_status "WAITING_FOR_SHARDS"
  require_path "${BASE_CHUNK_ROOT}"
  require_path "${JEPA_DENSE_CACHE_ROOT}"
  require_path "${NAVSIM_LOG_PATH}"
  require_path "${SENSOR_BLOBS_PATH}"
  require_path "${VLM_PATH}"
  wait_for_shards
  write_status "MERGING_AUDITING"
  merge_and_audit
  write_status "LAUNCHING_TRAINING"
  launch_a_local
  launch_b1_remote
  launch_b_followup_remote
  write_status "LAUNCHED"
  log "launched A, B1, and remote B3/B4 follow-up watcher. Status file: ${STATUS_FILE}"
}

main "$@"
