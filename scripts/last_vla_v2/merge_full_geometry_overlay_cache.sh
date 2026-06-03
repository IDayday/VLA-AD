#!/usr/bin/env bash
set -Eeuo pipefail

required=(BASE_CHUNK_ROOT GEOMETRY_CACHE_ROOT OUTPUT_CHUNK_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${OUTPUT_CHUNK_ROOT}"
cmd=(
  "${PYTHON_BIN}" scripts/merge_last_vla_geometry_cache_into_chunks.py
  --base-chunk-root "${BASE_CHUNK_ROOT}"
  --geometry-cache-root "${GEOMETRY_CACHE_ROOT}"
  --geometry-chunk-name-pattern "${GEOMETRY_CHUNK_NAME_PATTERN:-*}"
  --output-chunk-root "${OUTPUT_CHUNK_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --copy-mode "${COPY_MODE:-hardlink}"
  --min-coverage "${MIN_COVERAGE:-0.99}"
  --geometry-teacher-dim "${GEOMETRY_TEACHER_DIM:-512}"
  --num-geometry-tokens "${NUM_GEOMETRY_TOKENS:-12}"
)
if [[ "${OVERWRITE_CONTEXT:-0}" == "1" ]]; then cmd+=(--overwrite-context); fi
if [[ "${STRICT_COVERAGE:-0}" == "1" ]]; then cmd+=(--strict-coverage); fi

printf '%q ' "${cmd[@]}" >>"${OUTPUT_CHUNK_ROOT}/commands.log"; printf '\n' >>"${OUTPUT_CHUNK_ROOT}/commands.log"
"${cmd[@]}" 2>&1 | tee "${OUTPUT_CHUNK_ROOT}/merge.log"
