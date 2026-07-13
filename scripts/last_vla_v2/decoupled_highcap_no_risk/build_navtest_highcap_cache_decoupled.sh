#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BASE_CHUNK_ROOT="${BASE_CHUNK_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
ROOT="${ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk}"
FINAL_CHUNK_ROOT="${FINAL_CHUNK_ROOT:-${ROOT}/navtest_full_highcap_chunks}"
JEPA_RAW_ROOT="${JEPA_RAW_ROOT:-${ROOT}/navtest_jepa128_overlay_raw}"
JEPA_ROOT="${JEPA_ROOT:-${ROOT}/navtest_jepa128_overlay}"
GEOMETRY_RAW_ROOT="${GEOMETRY_RAW_ROOT:-${ROOT}/navtest_geometry192_overlay_raw}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-${ROOT}/navtest_geometry192_overlay}"
MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/manifests}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-${ROOT}/navtest_cache_logs/${RUN_ID}}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/status.env}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
VJEPA_MODEL_PATH="${VJEPA_MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B}"
CHUNK_PATTERN="${CHUNK_PATTERN:-navtest_full_chunk_*}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
COPY_MODE="${COPY_MODE:-hardlink}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING_SHARDS="${SKIP_EXISTING_SHARDS:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}"

write_status() {
  {
    printf 'STATUS=%q\n' "$1"
    printf 'UPDATED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'RUN_ID=%q\n' "${RUN_ID}"
    printf 'BASE_CHUNK_ROOT=%q\n' "${BASE_CHUNK_ROOT}"
    printf 'FINAL_CHUNK_ROOT=%q\n' "${FINAL_CHUNK_ROOT}"
    printf 'JEPA_ROOT=%q\n' "${JEPA_ROOT}"
    printf 'GEOMETRY_ROOT=%q\n' "${GEOMETRY_ROOT}"
    printf 'LOG_DIR=%q\n' "${LOG_DIR}"
  } >"${STATUS_FILE}"
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_DIR}/build.log"
}

fail() {
  local status="$1"
  shift
  log "$*"
  write_status "${status}"
  exit 1
}

on_error() {
  local rc=$?
  log "failed rc=${rc}"
  write_status "FAILED_RC_${rc}"
  exit "${rc}"
}
trap on_error ERR

require_path() {
  [[ -e "$1" ]] || fail "FAILED_MISSING_INPUT" "missing required path: $1"
}

prepare_output_root() {
  local path="$1"
  if [[ -e "${path}" && "${OVERWRITE}" == "1" ]]; then
    mv "${path}" "${path}.bak_${RUN_ID}"
  fi
  if [[ -e "${path}" ]]; then
    fail "FAILED_OUTPUT_EXISTS" "output exists: ${path}; set OVERWRITE=1 to move it aside"
  fi
  mkdir -p "${path}"
}

shard_done() {
  local dir="$1"
  [[ -s "${dir}/index.jsonl" && -s "${dir}/metadata.json" ]]
}

run_sharded_stage() {
  local stage="$1"
  local raw_root="$2"
  shift 2
  local -a base_cmd=("$@")
  local -a pids=()
  local -a gpus=()
  IFS=',' read -r -a gpus <<<"${GPU_LIST}"
  if (( ${#gpus[@]} == 0 )); then
    fail "FAILED_GPU_LIST" "GPU_LIST is empty"
  fi

  mkdir -p "${raw_root}/shards"
  log "starting ${stage} shards: num_shards=${NUM_SHARDS} gpu_list=${GPU_LIST}"
  for ((idx=0; idx<NUM_SHARDS; idx++)); do
    local shard_dir="${raw_root}/shards/shard_$(printf '%05d' "${idx}")"
    if [[ "${SKIP_EXISTING_SHARDS}" == "1" ]] && shard_done "${shard_dir}"; then
      log "${stage} shard ${idx} already exists; skipping"
      continue
    fi
    if [[ -e "${shard_dir}" && "${OVERWRITE}" == "1" ]]; then
      mv "${shard_dir}" "${shard_dir}.bak_${RUN_ID}"
    elif [[ -e "${shard_dir}" ]]; then
      rm -rf "${shard_dir}"
    fi
    local gpu="${gpus[$((idx % ${#gpus[@]}))]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${base_cmd[@]}" \
        --shard-index "${idx}" \
        --num-shards "${NUM_SHARDS}" \
        --output-shard-subdir \
        --write-shard-manifest
    ) >"${LOG_DIR}/${stage}_shard_${idx}.log" 2>&1 &
    pids+=("$!")
    log "${stage} shard ${idx} launched on CUDA_VISIBLE_DEVICES=${gpu}, pid=${pids[-1]}"
  done

  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if (( rc != 0 )); then
    fail "FAILED_${stage^^}" "${stage} shard generation failed; inspect ${LOG_DIR}/${stage}_shard_*.log"
  fi
  log "${stage} shards complete"
}

main() {
  write_status "STARTING"
  require_path "${BASE_CHUNK_ROOT}"
  require_path "${VJEPA_MODEL_PATH}"
  require_path "${VGGT_MODEL_PATH}"

  local base_count
  base_count="$(find "${BASE_CHUNK_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${CHUNK_PATTERN}" | wc -l)"
  [[ "${base_count}" -gt 0 ]] || fail "FAILED_NO_BASE_CHUNKS" "no base chunks matching ${CHUNK_PATTERN} under ${BASE_CHUNK_ROOT}"

  if [[ -e "${FINAL_CHUNK_ROOT}" && "${OVERWRITE}" != "1" ]]; then
    fail "FAILED_FINAL_EXISTS" "final cache already exists: ${FINAL_CHUNK_ROOT}"
  fi

  log "navtest highcap cache build root: ${ROOT}"
  log "base chunks: ${BASE_CHUNK_ROOT} (${base_count} chunks)"
  log "final output: ${FINAL_CHUNK_ROOT}"
  log "logs: ${LOG_DIR}"

  write_status "BUILDING_JEPA128"
  run_sharded_stage jepa128 "${JEPA_RAW_ROOT}" \
    "${PYTHON_BIN}" scripts/build_recogdrive_jepa_overlay_from_chunks.py \
      --base-chunk-root "${BASE_CHUNK_ROOT}" \
      --chunk-name-pattern "${CHUNK_PATTERN}" \
      --output-dir "${JEPA_RAW_ROOT}" \
      --data-root "${NAVSIM_DATA_ROOT}" \
      --split navtest \
      --jepa-model-path "${VJEPA_MODEL_PATH}" \
      --num-jepa-tokens 128 \
      --strict-highcap-jepa \
      --precision bf16 \
      --device cuda \
      --strict-coverage

  write_status "BUILDING_GEOMETRY192"
  run_sharded_stage geometry192 "${GEOMETRY_RAW_ROOT}" \
    "${PYTHON_BIN}" scripts/build_last_vla_full_geometry_cache.py \
      --chunk-cache-root "${BASE_CHUNK_ROOT}" \
      --chunk-name-pattern "${CHUNK_PATTERN}" \
      --output-cache-root "${GEOMETRY_RAW_ROOT}" \
      --vggt-model-path "${VGGT_MODEL_PATH}" \
      --precision fp32 \
      --device cuda \
      --num-vggt-tokens 128 \
      --vggt-context-grid-rows 8 \
      --vggt-context-grid-cols 16 \
      --num-geometry-tokens 192 \
      --geometry-grid-rows 12 \
      --geometry-grid-cols 16 \
      --geometry-teacher-dim 512 \
      --require-full-geometry

  write_status "MERGING_OVERLAYS"
  prepare_output_root "${JEPA_ROOT}"
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py \
    --sharded-root "${JEPA_RAW_ROOT}" \
    --output-root "${JEPA_ROOT}" \
    --expected-num-shards "${NUM_SHARDS}" \
    --overlay-type jepa128 \
    --copy-mode "${COPY_MODE}" \
    --strict >"${LOG_DIR}/merge_jepa128.log" 2>&1

  prepare_output_root "${GEOMETRY_ROOT}"
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py \
    --sharded-root "${GEOMETRY_RAW_ROOT}" \
    --output-root "${GEOMETRY_ROOT}" \
    --expected-num-shards "${NUM_SHARDS}" \
    --overlay-type geometry192 \
    --copy-mode "${COPY_MODE}" \
    --strict >"${LOG_DIR}/merge_geometry192.log" 2>&1

  write_status "MERGING_FINAL_CACHE"
  prepare_output_root "${FINAL_CHUNK_ROOT}"
  "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py \
    --base-chunk-root "${BASE_CHUNK_ROOT}" \
    --geometry-cache-root "${GEOMETRY_ROOT}" \
    --geometry-chunk-name-pattern "*" \
    --jepa-cache-root "${JEPA_ROOT}" \
    --jepa-chunk-name-pattern "*" \
    --output-chunk-root "${FINAL_CHUNK_ROOT}" \
    --chunk-name-pattern "${CHUNK_PATTERN}" \
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
    --geometry-teacher-dim 512 >"${LOG_DIR}/merge_final_cache.log" 2>&1

  "${PYTHON_BIN}" scripts/strip_legacy_vggt_target_tokens.py \
    --cache-root "${FINAL_CHUNK_ROOT}" \
    --chunk-name-pattern "${CHUNK_PATTERN}" \
    --workers 4 \
    --update-metadata \
    --output "${MANIFEST_DIR}/navtest_strip_legacy_vggt_target_tokens.json" \
    >"${LOG_DIR}/strip_legacy_vggt_target_tokens.log" 2>&1

  if [[ "${RUN_AUDIT}" == "1" ]]; then
    write_status "AUDITING_FINAL_CACHE"
    "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py \
      --cache-root "${FINAL_CHUNK_ROOT}" \
      --chunk-name-pattern "${CHUNK_PATTERN}" \
      --strict-full-geometry \
      --min-full-geometry-coverage 0.99 \
      --expected-jepa-tokens 128 \
      --expected-geometry-tokens 192 \
      --geometry-teacher-dim 512 \
      --strict-no-risk \
      --output "${MANIFEST_DIR}/navtest_manifest.json" >"${LOG_DIR}/audit_final_cache.log" 2>&1
  fi

  write_status "READY"
  log "ready: ${FINAL_CHUNK_ROOT}"
}

main "$@"
