#!/usr/bin/env bash
set -Eeuo pipefail

required=(BASE_CHUNK_ROOT OUTPUT_ROOT VGGT_MODEL_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ -z "${VJEPA_MODEL_PATH:-}" && -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
  echo "Set VJEPA_MODEL_PATH for JEPA128 generation or JEPA_DENSE_CACHE_ROOT for a prebuilt JEPA128 overlay." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ROOT="${OUTPUT_ROOT}/decoupled_highcap_no_risk"
MANIFEST_DIR="${ROOT}/manifests"
TRAIN_JEPA="${ROOT}/train_jepa128_overlay"
TRAIN_GEOMETRY="${ROOT}/train_geometry192_overlay"
TRAIN_FULL="${ROOT}/train_full_highcap_chunks"
COMMANDS_LOG="${ROOT}/commands.log"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ROOT}" "${MANIFEST_DIR}"

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

JEPA_SOURCE="${JEPA_DENSE_CACHE_ROOT:-${TRAIN_JEPA}}"
if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then
  cmd_jepa=(
    "${PYTHON_BIN}" scripts/build_recogdrive_jepa_overlay_from_chunks.py
    --base-chunk-root "${BASE_CHUNK_ROOT}"
    --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
    --output-dir "${TRAIN_JEPA}"
    --jepa-model-path "${VJEPA_MODEL_PATH}"
    --num-jepa-tokens 128
    --strict-highcap-jepa
    --precision "${PRECISION:-bf16}"
    --device "${DEVICE:-cuda}"
  )
  if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_jepa+=(--max-samples "${MAX_SAMPLES}"); fi
  if [[ -n "${SHARD_INDEX:-}" ]]; then cmd_jepa+=(--shard-index "${SHARD_INDEX}"); fi
  if [[ -n "${NUM_SHARDS:-}" ]]; then cmd_jepa+=(--num-shards "${NUM_SHARDS}"); fi
else
  cmd_jepa=(echo "Using prebuilt JEPA128 overlay: ${JEPA_DENSE_CACHE_ROOT}")
fi

cmd_geometry=(
  "${PYTHON_BIN}" scripts/build_last_vla_full_geometry_cache.py
  --chunk-cache-root "${BASE_CHUNK_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output-cache-root "${TRAIN_GEOMETRY}"
  --vggt-model-path "${VGGT_MODEL_PATH}"
  --precision "${PRECISION:-fp32}"
  --device "${DEVICE:-cuda}"
  --num-geometry-tokens 192
  --geometry-grid-rows 12
  --geometry-grid-cols 16
  --geometry-teacher-dim 512
  --require-full-geometry
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_geometry+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ -n "${SHARD_INDEX:-}" ]]; then cmd_geometry+=(--shard-index "${SHARD_INDEX}"); fi
if [[ -n "${NUM_SHARDS:-}" ]]; then cmd_geometry+=(--num-shards "${NUM_SHARDS}"); fi

cmd_merge=(
  "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py
  --base-chunk-root "${BASE_CHUNK_ROOT}"
  --geometry-cache-root "${TRAIN_GEOMETRY}"
  --geometry-chunk-name-pattern "${GEOMETRY_CHUNK_NAME_PATTERN:-*}"
  --jepa-cache-root "${JEPA_SOURCE}"
  --jepa-chunk-name-pattern "${JEPA_CHUNK_NAME_PATTERN:-*}"
  --output-chunk-root "${TRAIN_FULL}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --copy-mode "${COPY_MODE:-hardlink}"
  --strict-coverage
  --min-coverage 0.99
  --strict-jepa-coverage
  --min-jepa-coverage 0.99
  --expected-jepa-tokens 128
  --jepa-dim 1024
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
  printf '%q ' "${cmd_jepa[@]}"; printf '\n'
  printf '%q ' "${cmd_geometry[@]}"; printf '\n'
  printf '%q ' "${cmd_merge[@]}"; printf '\n'
  printf '%q ' "${cmd_audit[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
if [[ -z "${MAX_SAMPLES:-}" && "${ALLOW_FULL_CACHE_WITHOUT_MAX:-0}" != "1" ]]; then
  echo "Set MAX_SAMPLES, or ALLOW_FULL_CACHE_WITHOUT_MAX=1 for a full cache build." >&2
  exit 2
fi

prepare_output_dir "${TRAIN_GEOMETRY}"
if [[ -z "${JEPA_DENSE_CACHE_ROOT:-}" ]]; then prepare_output_dir "${TRAIN_JEPA}"; fi
prepare_output_dir "${TRAIN_FULL}"

"${cmd_jepa[@]}" 2>&1 | tee "${ROOT}/jepa128.log"
"${cmd_geometry[@]}" 2>&1 | tee "${ROOT}/geometry192.log"
"${cmd_merge[@]}" 2>&1 | tee "${ROOT}/merge.log"
"${cmd_audit[@]}" 2>&1 | tee "${ROOT}/audit.log"
