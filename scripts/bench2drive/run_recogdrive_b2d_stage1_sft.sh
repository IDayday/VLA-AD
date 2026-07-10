#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}

BASE_VLM_PATH=${BASE_VLM_PATH:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
DATA_ROOT=${DATA_ROOT:-/mnt/data/Bench2Drive-Base}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_sft_data_v1}
TRAIN_META=${TRAIN_META:-${STAGE1_DATA_DIR}/train_meta.json}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_sft_${RUN_ID}}

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python - <<'PY'
import os
print(max(1, len([item for item in os.environ.get("GPU_LIST", "0").split(",") if item.strip()])))
PY
)}
MASTER_PORT=${MASTER_PORT:-29531}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-12}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
SAVE_STEPS=${SAVE_STEPS:-250}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SEED=${SEED:-20260710}
MAX_STEPS=${MAX_STEPS:--1}
REPORT_TO=${REPORT_TO:-none}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}
PREPARE_DATA=${PREPARE_DATA:-1}

if [[ ! -d "${BASE_VLM_PATH}" ]]; then
  echo "Stage1 base VLM not found: ${BASE_VLM_PATH}" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NPROC_PER_NODE))

cd "${VLA_AD_ROOT}"
if [[ "${PREPARE_DATA}" == "1" || "${PREPARE_DATA}" == "true" || "${PREPARE_DATA}" == "TRUE" ]]; then
  if [[ ! -f "${TRAIN_META}" ]]; then
    mkdir -p "${STAGE1_DATA_DIR}"
    env PYTHONPATH="${VLA_AD_ROOT}:${VLA_AD_ROOT}/internvl_chat${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" \
      python scripts/bench2drive/prepare_recogdrive_b2d_stage1_sft.py \
        --data-root "${DATA_ROOT}" \
        --output-dir "${STAGE1_DATA_DIR}"
  fi
fi
if [[ ! -f "${TRAIN_META}" ]]; then
  echo "Stage1 training metadata not found: ${TRAIN_META}" >&2
  exit 2
fi
if [[ -n "${DEEPSPEED_CONFIG}" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
GIT_COMMIT=$(git rev-parse HEAD)
cat > "${OUTPUT_DIR}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=${GIT_COMMIT}
base_vlm_path=${BASE_VLM_PATH}
data_root=${DATA_ROOT}
stage1_data_dir=${STAGE1_DATA_DIR}
train_meta=${TRAIN_META}
output_dir=${OUTPUT_DIR}
gpu_list=${GPU_LIST}
nproc_per_node=${NPROC_PER_NODE}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
num_train_epochs=${NUM_TRAIN_EPOCHS}
learning_rate=${LEARNING_RATE}
weight_decay=${WEIGHT_DECAY}
warmup_ratio=${WARMUP_RATIO}
max_seq_length=${MAX_SEQ_LENGTH}
max_dynamic_patch=${MAX_DYNAMIC_PATCH}
max_steps=${MAX_STEPS}
seed=${SEED}
deepspeed_config=${DEEPSPEED_CONFIG}
EOF

args=(
  internvl/train/internvl_chat_finetune.py
  --model_name_or_path "${BASE_VLM_PATH}"
  --conv_style internvl2_5
  --use_fast_tokenizer False
  --output_dir "${OUTPUT_DIR}"
  --meta_path "${TRAIN_META}"
  --overwrite_output_dir True
  --force_image_size 448
  --min_dynamic_patch 1
  --max_dynamic_patch "${MAX_DYNAMIC_PATCH}"
  --down_sample_ratio 0.5
  --drop_path_rate 0.0
  --freeze_llm False
  --freeze_mlp False
  --freeze_backbone False
  --vision_select_layer -1
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --bf16 True
  --tf32 True
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --eval_strategy no
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --do_train True
  --grad_checkpoint True
  --group_by_length True
  --dynamic_image_size True
  --use_thumbnail True
  --ps_version v2
  --report_to "${REPORT_TO}"
  --seed "${SEED}"
  --data_seed "${SEED}"
  --max_steps "${MAX_STEPS}"
  --ddp_find_unused_parameters False
)
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  args+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

cmd=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${args[@]}")
printf '%q ' "${cmd[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

cd "${VLA_AD_ROOT}/internvl_chat"
env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTHONPATH="${VLA_AD_ROOT}:${VLA_AD_ROOT}/internvl_chat${PYTHONPATH:+:${PYTHONPATH}}" \
  LAUNCHER=pytorch \
  "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" \
  "${cmd[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

# Trainer.save_model does not copy local trust_remote_code modules. Keep the
# final Stage1 directory directly loadable by AutoModel and the Stage2 cache
# builder without relying on a pre-populated Hugging Face module cache.
runtime_files=(
  configuration_intern_vit.py
  configuration_internvl_chat.py
  conversation.py
  modeling_intern_vit.py
  modeling_internvl_chat.py
  generation_config.json
)
for filename in "${runtime_files[@]}"; do
  if [[ ! -f "${BASE_VLM_PATH}/${filename}" ]]; then
    echo "Missing required InternVL runtime file: ${BASE_VLM_PATH}/${filename}" >&2
    exit 2
  fi
  cp -f "${BASE_VLM_PATH}/${filename}" "${OUTPUT_DIR}/${filename}"
done
