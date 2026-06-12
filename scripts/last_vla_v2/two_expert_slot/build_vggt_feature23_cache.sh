#!/usr/bin/env bash
set -Eeuo pipefail

required=(INPUT_CHUNK_ROOT OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ "${RUN_CACHE:-0}" == "1" && -e "${OUTPUT_ROOT}" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "OUTPUT_ROOT exists: ${OUTPUT_ROOT}; set OVERWRITE=1 to replace shard outputs." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${OUTPUT_ROOT}"
COMMANDS_LOG="${OUTPUT_ROOT}/commands.log"

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/two_expert_slot/build_vggt_feature23_cache.py
  --input-chunk-root "${INPUT_CHUNK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --split "${SPLIT:-navtrain}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --num-shards "${NUM_SHARDS:-1}"
  --shard-index "${SHARD_INDEX:-0}"
)
if [[ -n "${VGGT_MODEL_PATH:-}" ]]; then cmd+=(--vggt-model-path "${VGGT_MODEL_PATH}"); fi
if [[ -n "${VGGT_MODEL_CLASS:-}" ]]; then cmd+=(--vggt-model-class "${VGGT_MODEL_CLASS}"); fi
cmd+=(
  --image-key "${IMAGE_KEY:-image_path_tensor}"
  --camera "${CAMERA:-front}"
  --image-size "${IMAGE_SIZE:-518}"
  --feature-layer-index "${FEATURE_LAYER_INDEX:-23}"
  --pack-tokens "${PACK_TOKENS:-12}"
  --device "${DEVICE:-cuda}"
  --precision "${PRECISION:-bf16}"
)
if [[ "${STRICT_TEACHER:-0}" == "1" ]]; then cmd+=(--strict-teacher); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then cmd+=(--overwrite); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/build_vggt_feature23_cache.log"
