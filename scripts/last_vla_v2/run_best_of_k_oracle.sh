#!/usr/bin/env bash
set -Eeuo pipefail

required=(CONFIG CHECKPOINT TRAIN_CHUNK_CACHE_ROOT OUTPUT_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${OUTPUT_DIR}"
cmd=(
  "${PYTHON_BIN}" scripts/eval_last_vla_best_of_k_oracle.py
  --config "${CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --chunk-cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --num-candidates "${NUM_CANDIDATES:-8}"
  --output-dir "${OUTPUT_DIR}"
  --precision "${PRECISION:-fp32}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ -n "${METRIC_CACHE_DIR:-}" ]]; then cmd+=(--metric-cache-dir "${METRIC_CACHE_DIR}"); fi
printf '%q ' "${cmd[@]}" >>"${OUTPUT_DIR}/commands.log"; printf '\n' >>"${OUTPUT_DIR}/commands.log"
"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/oracle.log"
