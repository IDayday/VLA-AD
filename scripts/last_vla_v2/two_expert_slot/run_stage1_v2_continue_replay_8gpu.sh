#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUTPUT_DIR MASTER_PORT VLM_PATH TEACHER_CACHE_ROOT RECOGDRIVE_STAGE1_REPLAY_CACHE INIT_STAGE1_CHECKPOINT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ "${ALLOW_NAVSIM_REPLAY_FALLBACK:-0}" == "1" ]]; then
  echo "Stage1-v2 official NAVSIM JSONL replay is available; fallback NAVSIM GT replay is refused by this launcher." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/stage1_v2_continue.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-8}" --master_port "${MASTER_PORT}"
  scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py
  --teacher-cache-root "${TEACHER_CACHE_ROOT}"
  --recogdrive-replay-cache-root "${RECOGDRIVE_STAGE1_REPLAY_CACHE}"
  --output-dir "${OUTPUT_DIR}"
  --vlm-path "${VLM_PATH}"
  --init-stage1-checkpoint "${INIT_STAGE1_CHECKPOINT}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --train-mode "${TRAIN_MODE:-lora}"
  --top-layers "${TOP_LAYERS:-2}"
  --batch-size "${BATCH_SIZE_PER_GPU:-4}"
  --grad-accum "${GRAD_ACCUM:-2}"
  --max-epochs "${MAX_EPOCHS:-2}"
  --lr-vlm "${LR_VLM:-2e-6}"
  --lr-slots-adapters "${LR_SLOTS_ADAPTERS:-5e-5}"
  --weight-decay "${WEIGHT_DECAY:-1e-4}"
  --precision "${PRECISION:-bf16-mixed}"
  --teacher-lru-size "${TEACHER_LRU_SIZE:-128}"
  --max-image-patches "${MAX_IMAGE_PATCHES:-12}"
  --log-every-steps "${LOG_EVERY_STEPS:-50}"
  --stage1-image-mask-ratio "${STAGE1_IMAGE_MASK_RATIO:-0.7}"
  --stage1-image-mask-ratio-start "${STAGE1_IMAGE_MASK_RATIO_START:-0.4}"
  --stage1-image-mask-warmup-fraction "${STAGE1_IMAGE_MASK_WARMUP_FRACTION:-0.25}"
  --stage1-image-dropout-prob "${STAGE1_IMAGE_DROPOUT_PROB:-0.25}"
  --stage1-image-dropout-prob-start "${STAGE1_IMAGE_DROPOUT_PROB_START:-0.05}"
  --stage1-image-dropout-warmup-fraction "${STAGE1_IMAGE_DROPOUT_WARMUP_FRACTION:-0.25}"
  --no-slot-only-use-image-memory
  --slot-only-dyn-loss-weight "${SLOT_ONLY_DYN_LOSS_WEIGHT:-0.2}"
  --slot-only-geo-loss-weight "${SLOT_ONLY_GEO_LOSS_WEIGHT:-0.2}"
  --contrastive-dyn-loss-weight "${CONTRASTIVE_DYN_LOSS_WEIGHT:-0.02}"
  --contrastive-geo-loss-weight "${CONTRASTIVE_GEO_LOSS_WEIGHT:-0.02}"
  --contrastive-loss-warmup-fraction "${CONTRASTIVE_LOSS_WARMUP_FRACTION:-0.30}"
  --contrastive-temperature "${CONTRASTIVE_TEMPERATURE:-0.07}"
  --probe-fused-loss-weight "${PROBE_FUSED_LOSS_WEIGHT:-0.2}"
  --probe-dyn-loss-weight "${PROBE_DYN_LOSS_WEIGHT:-0.05}"
  --probe-geo-loss-weight "${PROBE_GEO_LOSS_WEIGHT:-0.10}"
  --probe-heading-loss-weight "${PROBE_HEADING_LOSS_WEIGHT:-0.05}"
  --probe-progress-loss-weight "${PROBE_PROGRESS_LOSS_WEIGHT:-0.05}"
  --geo-lateral-profile-loss-weight "${GEO_LATERAL_PROFILE_LOSS_WEIGHT:-0.05}"
  --geo-heading-profile-loss-weight "${GEO_HEADING_PROFILE_LOSS_WEIGHT:-0.05}"
  --recogdrive-replay-ce-loss-weight "${RECOGDRIVE_REPLAY_CE_LOSS_WEIGHT:-0.1}"
  --recogdrive-replay-every-n-steps "${RECOGDRIVE_REPLAY_EVERY_N_STEPS:-2}"
  --recogdrive-replay-source "${RECOGDRIVE_REPLAY_SOURCE:-official}"
  --recogdrive-replay-max-answer-tokens "${RECOGDRIVE_REPLAY_MAX_ANSWER_TOKENS:-256}"
  --replay-ce-loss-warmup-fraction "${REPLAY_CE_LOSS_WARMUP_FRACTION:-0.30}"
  --hidden-anchor-weight "${HIDDEN_ANCHOR_WEIGHT:-0.03}"
  --hidden-anchor-every-n-steps "${HIDDEN_ANCHOR_EVERY_N_STEPS:-4}"
)

if [[ -n "${BASE_CHUNK_ROOT:-}" ]]; then
  cmd+=(--base-chunk-root "${BASE_CHUNK_ROOT}")
fi
if [[ "${ALLOW_REPLAY_ONLY_BASE:-0}" == "1" ]]; then
  cmd+=(--allow-replay-only-base)
fi

if [[ -n "${INIT_VLM_LORA_ADAPTER_DIR:-}" ]]; then
  cmd+=(--init-vlm-lora-adapter-dir "${INIT_VLM_LORA_ADAPTER_DIR}")
fi
if [[ -n "${VGGT_FEATURE_DIM:-}" ]]; then
  cmd+=(--vggt-feature-dim "${VGGT_FEATURE_DIM}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
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

printf '[%s] ' "$(date -Is)" >>"${COMMANDS_LOG}"
printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"
printf '\n' >>"${COMMANDS_LOG}"

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
if [[ -z "${READINESS_GATE_JSON:-}" ]]; then
  echo "READINESS_GATE_JSON is required for Stage1-v2 full training." >&2
  exit 2
fi
"${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=d.get("status")=="READY" and d.get("ok") is True; sys.exit(0 if ok else 1)' "${READINESS_GATE_JSON}" || {
  echo "Readiness gate is not READY: ${READINESS_GATE_JSON}" >&2
  exit 2
}

set +e
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
echo "stage1_v2_continue exited with status ${status}" >>"${COMMANDS_LOG}"
exit "${status}"
