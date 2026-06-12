#!/usr/bin/env bash
set -Eeuo pipefail

required=(BASE_CHUNK_ROOT OUTPUT_ROOT VLM_PATH)
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
  "${PYTHON_BIN}" scripts/last_vla_v2/two_expert_slot/build_two_expert_hidden_cache.py
  --base-chunk-root "${BASE_CHUNK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --split "${SPLIT:-navtrain}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --num-shards "${NUM_SHARDS:-1}"
  --shard-index "${SHARD_INDEX:-0}"
  --batch-size "${BATCH_SIZE:-1}"
  --device "${DEVICE:-cuda}"
  --precision "${PRECISION:-bf16}"
  --train-vlm-mode "${TRAIN_VLM_MODE:-frozen}"
)
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then cmd+=(--stage1-checkpoint "${STAGE1_CHECKPOINT}"); fi
if [[ "${INCLUDE_TEACHER_TARGETS:-0}" == "1" ]]; then cmd+=(--include-teacher-targets); fi
if [[ "${ALLOW_EVAL_TEACHER_TARGETS:-0}" == "1" ]]; then cmd+=(--allow-eval-teacher-targets); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then cmd+=(--overwrite); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/build_two_expert_hidden_cache.log"
