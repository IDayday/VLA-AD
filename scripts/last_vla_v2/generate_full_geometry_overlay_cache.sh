#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUTPUT_CACHE_ROOT VGGT_MODEL_PATH CHUNK_CACHE_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PRECISION="${PRECISION:-fp32}"
DEVICE="${DEVICE:-cuda}"
GEOMETRY_TEACHER_DIM="${GEOMETRY_TEACHER_DIM:-512}"
NUM_GEOMETRY_TOKENS="${NUM_GEOMETRY_TOKENS:-192}"
GEOMETRY_GRID_ROWS="${GEOMETRY_GRID_ROWS:-12}"
GEOMETRY_GRID_COLS="${GEOMETRY_GRID_COLS:-16}"
CHUNK_NAME="${CHUNK_NAME:-full_geometry_overlay_shard_${SHARD_INDEX:-0}_of_${NUM_SHARDS:-1}}"
OUT_CHUNK="${OUTPUT_CACHE_ROOT}/${CHUNK_NAME}"

mkdir -p "${OUT_CHUNK}"
cmd=(
  "${PYTHON_BIN}" scripts/build_last_vla_full_geometry_cache.py
  --chunk-cache-root "${CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output-cache-root "${OUT_CHUNK}"
  --vggt-model-path "${VGGT_MODEL_PATH}"
  --precision "${PRECISION}"
  --device "${DEVICE}"
  --geometry-teacher-dim "${GEOMETRY_TEACHER_DIM}"
  --num-geometry-tokens "${NUM_GEOMETRY_TOKENS}"
  --geometry-grid-rows "${GEOMETRY_GRID_ROWS}"
  --geometry-grid-cols "${GEOMETRY_GRID_COLS}"
)
if [[ "${ALLOW_PATCH_FALLBACK:-0}" != "1" ]]; then cmd+=(--require-full-geometry); fi
if [[ "${ALLOW_PATCH_FALLBACK:-0}" == "1" ]]; then cmd+=(--allow-patch-fallback); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ -n "${SHARD_INDEX:-}" ]]; then cmd+=(--shard-index "${SHARD_INDEX}"); fi
if [[ -n "${NUM_SHARDS:-}" ]]; then cmd+=(--num-shards "${NUM_SHARDS}"); fi

printf '%q ' "${cmd[@]}" >>"${OUT_CHUNK}/commands.log"; printf '\n' >>"${OUT_CHUNK}/commands.log"
"${cmd[@]}" 2>&1 | tee "${OUT_CHUNK}/generate.log"
