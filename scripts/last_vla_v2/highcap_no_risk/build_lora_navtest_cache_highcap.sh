#!/usr/bin/env bash
set -Eeuo pipefail

required=(NAVTEST_CHUNK_CACHE_ROOT VLM_PATH OUTPUT_CHUNK_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ -z "${VLM_LORA_ADAPTER_DIR:-}" && -z "${VLM_LORA_ADAPTER:-}" ]]; then
  echo "Missing required environment variable: VLM_LORA_ADAPTER_DIR or VLM_LORA_ADAPTER" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${OUTPUT_CHUNK_ROOT}"
cmd=(
  "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py
  --base-chunk-root "${NAVTEST_CHUNK_CACHE_ROOT}"
  --output-chunk-root "${OUTPUT_CHUNK_ROOT}"
  --vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --precision "${PRECISION:-bf16}"
  --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
  --batch-size "${LORA_CACHE_BATCH_SIZE:-1}"
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
  --cache-variant highcap_no_risk
)
if [[ -n "${VLM_LORA_ADAPTER_DIR:-}" ]]; then
  cmd+=(--vlm-lora-adapter-dir "${VLM_LORA_ADAPTER_DIR}")
else
  cmd+=(--vlm-lora-adapter "${VLM_LORA_ADAPTER}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
printf '%q ' "${cmd[@]}" >>"${OUTPUT_CHUNK_ROOT}/commands.log"; printf '\n' >>"${OUTPUT_CHUNK_ROOT}/commands.log"

if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Command written to ${OUTPUT_CHUNK_ROOT}/commands.log"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee "${OUTPUT_CHUNK_ROOT}/build_lora_navtest_cache.log"
