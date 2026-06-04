#!/usr/bin/env bash
set -Eeuo pipefail

required=(CONFIG CHECKPOINT TRAIN_CHUNK_CACHE_ROOT OUTPUT_CACHE_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
SCORE_MODE="${SCORE_MODE:-pdm}"
if [[ "${SCORE_MODE}" == "pdm" && -z "${METRIC_CACHE_DIR:-}" ]]; then
  echo "METRIC_CACHE_DIR is required when SCORE_MODE=pdm." >&2
  exit 2
fi
mkdir -p "${OUTPUT_CACHE_ROOT}"
cmd=(
  "${PYTHON_BIN}" scripts/generate_last_vla_teacher_trajectory_cache.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --chunk-cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output-cache-root "${OUTPUT_CACHE_ROOT}"
  --num-candidates "${NUM_CANDIDATES:-8}"
  --precision "${PRECISION:-fp32}"
  --score-mode "${SCORE_MODE}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ -n "${METRIC_CACHE_DIR:-}" ]]; then cmd+=(--metric-cache-dir "${METRIC_CACHE_DIR}"); fi
if [[ "${ONLY_SAVE_IF_BETTER:-0}" == "1" ]]; then cmd+=(--only-save-if-better); fi
printf '%q ' "${cmd[@]}" >>"${OUTPUT_CACHE_ROOT}/commands.log"; printf '\n' >>"${OUTPUT_CACHE_ROOT}/commands.log"
"${cmd[@]}" 2>&1 | tee "${OUTPUT_CACHE_ROOT}/generate.log"
