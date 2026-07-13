#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUTPUT_DIR MASTER_PORT NAVSIM_LOG_PATH SENSOR_BLOBS_PATH VLM_PATH TEACHER_CACHE_ROOT BASE_CHUNK_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/two_expert_stage1_vlm_sft.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py
  --base-chunk-root "${BASE_CHUNK_ROOT}"
  --teacher-cache-root "${TEACHER_CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --vlm-path "${VLM_PATH}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --train-mode "${TRAIN_MODE:-lora}"
  --top-layers "${TOP_LAYERS:-2}"
  --batch-size "${BATCH_SIZE_PER_GPU:-2}"
  --grad-accum "${GRAD_ACCUM:-4}"
  --max-epochs "${MAX_EPOCHS:-2}"
  --lr-vlm "${LR_VLM:-1e-5}"
  --lr-slots-adapters "${LR_SLOTS_ADAPTERS:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --precision "${PRECISION:-bf16-mixed}"
  --teacher-lru-size "${TEACHER_LRU_SIZE:-0}"
  --max-image-patches "${MAX_IMAGE_PATCHES:-12}"
  --log-every-steps "${LOG_EVERY_STEPS:-50}"
)

if [[ -n "${VGGT_FEATURE_DIM:-}" ]]; then
  cmd+=(--vggt-feature-dim "${VGGT_FEATURE_DIM}")
fi
if [[ "${ALLOW_FULL_VLM_SFT:-0}" == "1" ]]; then
  cmd+=(--allow-full-vlm-sft)
fi
if [[ "${ALLOW_DEV_FALLBACK_TEACHERS:-0}" == "1" ]]; then
  cmd+=(--allow-dev-fallback-teachers)
fi
if [[ "${ALLOW_MINIMAL_PROMPT:-0}" == "1" ]]; then
  cmd+=(--allow-minimal-prompt)
fi
if [[ "${SAVE_FULL_STAGE1_STATE:-0}" == "1" ]]; then
  cmd+=(--save-full-stage1-state)
fi

echo "two_expert prompt version: two_expert_slot_prompt_v1" >>"${COMMANDS_LOG}"

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
if [[ "${ALLOW_SKIP_READINESS_GATE:-0}" != "1" ]]; then
  if [[ -z "${READINESS_GATE_JSON:-}" ]]; then
    echo "READINESS_GATE_JSON is required for Stage1 full training unless ALLOW_SKIP_READINESS_GATE=1." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); ok=d.get("status")=="READY" and d.get("ok") is True; sys.exit(0 if ok else 1)' "${READINESS_GATE_JSON}" || {
    echo "Readiness gate is not READY: ${READINESS_GATE_JSON}" >&2
    exit 2
  }
  if [[ "${ALLOW_LOW_COVERAGE:-0}" != "1" ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); ok=d.get("coverage_ok") is True and d.get("all_teacher_coverage") is not None; sys.exit(0 if ok else 1)' "${READINESS_GATE_JSON}" || {
      echo "Readiness gate JSON lacks passing coverage. Run preflight_two_expert_coverage.py and rebuild the gate, or set ALLOW_LOW_COVERAGE=1 for explicit dev-only override." >&2
      exit 2
    }
  fi
fi
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
