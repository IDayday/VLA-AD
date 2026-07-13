#!/usr/bin/env bash
set -Eeuo pipefail

required=(INPUT_CHUNK_ROOT OUTPUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ "${RUN_CACHE:-0}" == "1" && -e "${OUTPUT_ROOT}" && "${OVERWRITE:-0}" != "1" && "${RESUME_EXISTING:-0}" != "1" ]]; then
  echo "OUTPUT_ROOT exists: ${OUTPUT_ROOT}; set OVERWRITE=1 to replace shard outputs or RESUME_EXISTING=1 to reuse complete samples." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${OUTPUT_ROOT}"
COMMANDS_LOG="${OUTPUT_ROOT}/commands.log"

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/two_expert_slot/build_jepa_dynamic_teacher_cache.py
  --input-chunk-root "${INPUT_CHUNK_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --split "${SPLIT:-navtrain}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --num-shards "${NUM_SHARDS:-1}"
  --shard-index "${SHARD_INDEX:-0}"
)
if [[ -n "${JEPA_MODEL_PATH:-}" ]]; then cmd+=(--jepa-model-path "${JEPA_MODEL_PATH}"); fi
if [[ "${ALLOW_DEV_FALLBACK_TEACHERS:-0}" != "1" && "${STRICT_TEACHER:-1}" == "1" ]]; then
  cmd+=(--strict-teacher)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then cmd+=(--overwrite); fi
if [[ "${RESUME_EXISTING:-0}" == "1" ]]; then cmd+=(--resume-existing); fi
cmd+=(--extract-batch-size "${JEPA_EXTRACT_BATCH_SIZE:-1}")
cmd+=(--progress-interval "${PROGRESS_INTERVAL:-1000}")
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  echo "RUN_CACHE is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/build_jepa_dynamic_teacher_cache.log"
