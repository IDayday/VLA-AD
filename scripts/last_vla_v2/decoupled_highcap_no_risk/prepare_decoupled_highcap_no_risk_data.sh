#!/usr/bin/env bash
set -Eeuo pipefail

required=(BASE_CHUNK_ROOT OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

SKIP_BUILD_SHARDS="${SKIP_BUILD_SHARDS:-0}"
MERGE_SHARDS="${MERGE_SHARDS:-0}"
if [[ "${SKIP_BUILD_SHARDS}" != "1" ]]; then
  if [[ -z "${VGGT_MODEL_PATH:-}" ]]; then
    echo "Missing required environment variable: VGGT_MODEL_PATH" >&2
    exit 2
  fi
  if [[ -z "${VJEPA_MODEL_PATH:-}" && -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
    echo "Set VJEPA_MODEL_PATH for JEPA128 generation or JEPA_DENSE_CACHE_ROOT for a prebuilt JEPA128 overlay." >&2
    exit 2
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT="${OUTPUT_ROOT}/decoupled_highcap_no_risk"
MANIFEST_DIR="${ROOT}/manifests"
TRAIN_JEPA="${ROOT}/train_jepa128_overlay"
TRAIN_JEPA_RAW="${ROOT}/train_jepa128_overlay_raw"
TRAIN_GEOMETRY="${ROOT}/train_geometry192_overlay"
TRAIN_GEOMETRY_RAW="${ROOT}/train_geometry192_overlay_raw"
TRAIN_FULL="${ROOT}/train_full_highcap_chunks"
COMMANDS_LOG="${ROOT}/commands.log"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_SHARDS_VALUE="${NUM_SHARDS:-1}"
SHARD_INDEX_VALUE="${SHARD_INDEX:-0}"
EXPECTED_NUM_SHARDS_VALUE="${EXPECTED_NUM_SHARDS:-${NUM_SHARDS_VALUE}}"
COPY_MODE_VALUE="${COPY_MODE:-hardlink}"
TRAIN_SPLIT="${TRAIN_SPLIT:-navtrain}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
LOADER_MAX_SCENES="${LOADER_MAX_SCENES:-}"
mkdir -p "${ROOT}" "${MANIFEST_DIR}"

if (( NUM_SHARDS_VALUE <= 0 )); then
  echo "NUM_SHARDS must be positive." >&2
  exit 2
fi
if (( SHARD_INDEX_VALUE < 0 || SHARD_INDEX_VALUE >= NUM_SHARDS_VALUE )); then
  echo "SHARD_INDEX must be in [0, NUM_SHARDS)." >&2
  exit 2
fi

prepare_output_dir() {
  local dir="$1"
  if [[ -e "${dir}" ]]; then
    if [[ "${OVERWRITE_HIGHCAP_CACHE:-0}" != "1" ]]; then
      echo "Output directory exists: ${dir}. Set OVERWRITE_HIGHCAP_CACHE=1 to move it aside." >&2
      exit 2
    fi
    mv "${dir}" "${dir}.bak_${TIMESTAMP}"
  fi
  mkdir -p "${dir}"
}

run_and_log() {
  local log_path="$1"
  shift
  "$@" 2>&1 | tee "${log_path}"
}

BUILD_JEPA_ROOT="${TRAIN_JEPA}"
BUILD_GEOMETRY_ROOT="${TRAIN_GEOMETRY}"
JEPA_SOURCE="${JEPA_DENSE_CACHE_ROOT:-${TRAIN_JEPA}}"
if (( NUM_SHARDS_VALUE > 1 )); then
  BUILD_JEPA_ROOT="${TRAIN_JEPA_RAW}"
  BUILD_GEOMETRY_ROOT="${TRAIN_GEOMETRY_RAW}"
fi

JEPA_SHARD_DIR="${BUILD_JEPA_ROOT}/shards/shard_$(printf '%05d' "${SHARD_INDEX_VALUE}")"
GEOMETRY_SHARD_DIR="${BUILD_GEOMETRY_ROOT}/shards/shard_$(printf '%05d' "${SHARD_INDEX_VALUE}")"

cmd_jepa=()
if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
  cmd_jepa=(
    "${PYTHON_BIN}" scripts/build_recogdrive_jepa_overlay_from_chunks.py
    --base-chunk-root "${BASE_CHUNK_ROOT}"
    --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
    --output-dir "${BUILD_JEPA_ROOT}"
    --data-root "${NAVSIM_DATA_ROOT}"
    --split "${TRAIN_SPLIT}"
    --jepa-model-path "${VJEPA_MODEL_PATH:-}"
    --num-jepa-tokens 128
    --strict-highcap-jepa
    --precision "${PRECISION:-bf16}"
    --device "${DEVICE:-cuda}"
    --shard-index "${SHARD_INDEX_VALUE}"
    --num-shards "${NUM_SHARDS_VALUE}"
  )
  if (( NUM_SHARDS_VALUE > 1 )); then
    cmd_jepa+=(--output-shard-subdir --write-shard-manifest)
  fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_jepa+=(--max-samples "${MAX_SAMPLES}"); fi
  if [[ -n "${LOADER_MAX_SCENES}" ]]; then cmd_jepa+=(--loader-max-scenes "${LOADER_MAX_SCENES}"); fi
else
  cmd_jepa=(echo "Using prebuilt JEPA128 overlay: ${JEPA_DENSE_CACHE_ROOT}")
fi

cmd_geometry=(
  "${PYTHON_BIN}" scripts/build_last_vla_full_geometry_cache.py
  --chunk-cache-root "${BASE_CHUNK_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output-cache-root "${BUILD_GEOMETRY_ROOT}"
  --vggt-model-path "${VGGT_MODEL_PATH:-}"
  --precision "${PRECISION:-fp32}"
  --device "${DEVICE:-cuda}"
  --num-vggt-tokens 128
  --vggt-context-grid-rows 8
  --vggt-context-grid-cols 16
  --num-geometry-tokens 192
  --geometry-grid-rows 12
  --geometry-grid-cols 16
  --geometry-teacher-dim 512
  --require-full-geometry
  --shard-index "${SHARD_INDEX_VALUE}"
  --num-shards "${NUM_SHARDS_VALUE}"
)
if (( NUM_SHARDS_VALUE > 1 )); then
  cmd_geometry+=(--output-shard-subdir --write-shard-manifest)
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_geometry+=(--max-samples "${MAX_SAMPLES}"); fi

cmd_merge_jepa_shards=(
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py
  --sharded-root "${TRAIN_JEPA_RAW}"
  --output-root "${TRAIN_JEPA}"
  --expected-num-shards "${EXPECTED_NUM_SHARDS_VALUE}"
  --overlay-type jepa128
  --copy-mode "${COPY_MODE_VALUE}"
  --strict
)
cmd_merge_geometry_shards=(
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py
  --sharded-root "${TRAIN_GEOMETRY_RAW}"
  --output-root "${TRAIN_GEOMETRY}"
  --expected-num-shards "${EXPECTED_NUM_SHARDS_VALUE}"
  --overlay-type geometry192
  --copy-mode "${COPY_MODE_VALUE}"
  --strict
)

cmd_merge=(
  "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py
  --base-chunk-root "${BASE_CHUNK_ROOT}"
  --geometry-cache-root "${TRAIN_GEOMETRY}"
  --geometry-chunk-name-pattern "${GEOMETRY_CHUNK_NAME_PATTERN:-*}"
  --jepa-cache-root "${JEPA_SOURCE}"
  --jepa-chunk-name-pattern "${JEPA_CHUNK_NAME_PATTERN:-*}"
  --output-chunk-root "${TRAIN_FULL}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --copy-mode "${COPY_MODE_VALUE}"
  --overwrite-context
  --strict-coverage
  --min-coverage 0.99
  --strict-jepa-coverage
  --min-jepa-coverage 0.99
  --expected-jepa-tokens 128
  --jepa-dim 1024
  --expected-vggt-context-tokens 128
  --vggt-dim 2048
  --num-geometry-tokens 192
  --geometry-grid-rows 12
  --geometry-grid-cols 16
  --geometry-teacher-dim 512
)

cmd_audit=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${TRAIN_FULL}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --strict-full-geometry
  --min-full-geometry-coverage 0.99
  --expected-jepa-tokens 128
  --expected-geometry-tokens 192
  --geometry-teacher-dim 512
  --strict-no-risk
  --output "${MANIFEST_DIR}/train_manifest.json"
)

{
  date -Is
  printf '# num_shards=%q shard_index=%q expected_num_shards=%q skip_build_shards=%q merge_shards=%q\n' \
    "${NUM_SHARDS_VALUE}" "${SHARD_INDEX_VALUE}" "${EXPECTED_NUM_SHARDS_VALUE}" "${SKIP_BUILD_SHARDS}" "${MERGE_SHARDS}"
  printf '# train_split=%q navsim_data_root=%q loader_max_scenes=%q\n' \
    "${TRAIN_SPLIT}" "${NAVSIM_DATA_ROOT}" "${LOADER_MAX_SCENES}"
  if (( NUM_SHARDS_VALUE > 1 )); then
    printf '# jepa_shard_dir=%q\n' "${JEPA_SHARD_DIR}"
    printf '# geometry_shard_dir=%q\n' "${GEOMETRY_SHARD_DIR}"
  fi
  if [[ "${SKIP_BUILD_SHARDS}" != "1" ]]; then
    printf '%q ' "${cmd_jepa[@]}"; printf '\n'
    printf '%q ' "${cmd_geometry[@]}"; printf '\n'
  fi
  if (( NUM_SHARDS_VALUE <= 1 )) || [[ "${MERGE_SHARDS}" == "1" ]]; then
    if (( NUM_SHARDS_VALUE > 1 )); then
      if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then printf '%q ' "${cmd_merge_jepa_shards[@]}"; printf '\n'; fi
      printf '%q ' "${cmd_merge_geometry_shards[@]}"; printf '\n'
    fi
    printf '%q ' "${cmd_merge[@]}"; printf '\n'
    printf '%q ' "${cmd_audit[@]}"; printf '\n'
  fi
} >>"${COMMANDS_LOG}"

if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
if [[ -z "${MAX_SAMPLES:-}" && "${ALLOW_FULL_CACHE_WITHOUT_MAX:-0}" != "1" ]]; then
  echo "Set MAX_SAMPLES, or ALLOW_FULL_CACHE_WITHOUT_MAX=1 for a full cache build." >&2
  exit 2
fi

if (( NUM_SHARDS_VALUE > 1 )); then
  if [[ "${SKIP_BUILD_SHARDS}" != "1" ]]; then
    if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
      prepare_output_dir "${JEPA_SHARD_DIR}"
      run_and_log "${ROOT}/jepa128_shard_${SHARD_INDEX_VALUE}.log" "${cmd_jepa[@]}"
    fi
    prepare_output_dir "${GEOMETRY_SHARD_DIR}"
    run_and_log "${ROOT}/geometry192_shard_${SHARD_INDEX_VALUE}.log" "${cmd_geometry[@]}"
  fi
  if [[ "${MERGE_SHARDS}" != "1" ]]; then
    echo "Shard build complete; MERGE_SHARDS is not 1, so final merge and audit were not run."
    exit 0
  fi
  if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
    prepare_output_dir "${TRAIN_JEPA}"
    run_and_log "${ROOT}/merge_jepa128_shards.log" "${cmd_merge_jepa_shards[@]}"
  fi
  prepare_output_dir "${TRAIN_GEOMETRY}"
  prepare_output_dir "${TRAIN_FULL}"
  run_and_log "${ROOT}/merge_geometry192_shards.log" "${cmd_merge_geometry_shards[@]}"
  run_and_log "${ROOT}/merge.log" "${cmd_merge[@]}"
  run_and_log "${ROOT}/audit.log" "${cmd_audit[@]}"
  exit 0
fi

prepare_output_dir "${TRAIN_GEOMETRY}"
if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then prepare_output_dir "${TRAIN_JEPA}"; fi
prepare_output_dir "${TRAIN_FULL}"

run_and_log "${ROOT}/jepa128.log" "${cmd_jepa[@]}"
run_and_log "${ROOT}/geometry192.log" "${cmd_geometry[@]}"
run_and_log "${ROOT}/merge.log" "${cmd_merge[@]}"
run_and_log "${ROOT}/audit.log" "${cmd_audit[@]}"
