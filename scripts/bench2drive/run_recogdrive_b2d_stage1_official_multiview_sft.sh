#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}
TRAIN_PYTHON=${TRAIN_PYTHON:-}
TORCHRUN_BIN=${TORCHRUN_BIN:-}
RUN_MODE=${RUN_MODE:-formal}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

BASE_VLM_PATH=${BASE_VLM_PATH:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
REPRODUCTION_GATE=${REPRODUCTION_GATE:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_reproduction_gate.json}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_official_data}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${VLA_AD_ROOT}/internvl_chat/zero_stage1_config.json}

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
export GPU_LIST
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python - <<'PY'
import os
print(max(1, len([item for item in os.environ.get("GPU_LIST", "0").split(",") if item.strip()])))
PY
)}
MASTER_PORT=${MASTER_PORT:-29541}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-4e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-12288}
MAX_DYNAMIC_PATCH=${MAX_DYNAMIC_PATCH:-16}
DROP_PATH_RATE=${DROP_PATH_RATE:-0.1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SEED=${SEED:-20260711}
REPORT_TO=${REPORT_TO:-}
DRY_RUN=${DRY_RUN:-0}
SKIP_DATA_PREFLIGHT=${SKIP_DATA_PREFLIGHT:-0}
ALLOW_PUBLIC_PROXY_OVERRIDE=${ALLOW_PUBLIC_PROXY_OVERRIDE:-0}
ALLOW_ENVIRONMENT_VERSION_MISMATCH=${ALLOW_ENVIRONMENT_VERSION_MISMATCH:-0}

if [[ -n "${TRAIN_PYTHON}" ]]; then
  if [[ ! -x "${TRAIN_PYTHON}" ]]; then
    echo "TRAIN_PYTHON is not executable: ${TRAIN_PYTHON}" >&2
    exit 2
  fi
  if [[ -z "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_BIN=$(dirname "${TRAIN_PYTHON}")/torchrun
  fi
  if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun is not executable: ${TORCHRUN_BIN}" >&2
    exit 2
  fi
  PYTHON_RUN=("${TRAIN_PYTHON}")
  TRAIN_ENV_PREFIX=()
  TRAIN_ENV_DESCRIPTION="venv:${TRAIN_PYTHON}"
else
  TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
  PYTHON_RUN=("${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" python)
  TRAIN_ENV_PREFIX=("${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}")
  TRAIN_ENV_DESCRIPTION="conda:${NAVSIM_ENV}"
fi

case "${RUN_MODE}" in
  formal)
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
    MAX_STEPS=${MAX_STEPS:--1}
    SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
    SAVE_STEPS=${SAVE_STEPS:-200}
    SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
    REPORT_TO=${REPORT_TO:-tensorboard}
    SMOKE_RECORDS_PER_DATASET=${SMOKE_RECORDS_PER_DATASET:-8}
    OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_official_multiview_${RUN_ID}}
    GATE_FULL_HASH=${GATE_FULL_HASH:-1}
    ;;
  smoke)
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    MAX_STEPS=${MAX_STEPS:-2}
    SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
    SAVE_STEPS=${SAVE_STEPS:-1}
    SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
    REPORT_TO=${REPORT_TO:-none}
    SMOKE_RECORDS_PER_DATASET=${SMOKE_RECORDS_PER_DATASET:-8}
    OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_official_multiview_smoke_${RUN_ID}}
    GATE_FULL_HASH=${GATE_FULL_HASH:-0}
    ;;
  *)
    echo "RUN_MODE must be formal or smoke, got ${RUN_MODE}" >&2
    exit 2
    ;;
esac

require_formal_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "${RUN_MODE}" == "formal" && "${ALLOW_PUBLIC_PROXY_OVERRIDE}" != "1" && "${actual}" != "${expected}" ]]; then
    echo "Formal Stage1 requires ${name}=${expected}, got ${actual}. Set ALLOW_PUBLIC_PROXY_OVERRIDE=1 only for a labeled ablation." >&2
    exit 2
  fi
}

require_formal_value NUM_TRAIN_EPOCHS "${NUM_TRAIN_EPOCHS}" 3
require_formal_value GLOBAL_BATCH_SIZE "${GLOBAL_BATCH_SIZE}" 1024
require_formal_value PER_DEVICE_BATCH_SIZE "${PER_DEVICE_BATCH_SIZE}" 1
require_formal_value NPROC_PER_NODE "${NPROC_PER_NODE}" 8
require_formal_value LEARNING_RATE "${LEARNING_RATE}" 4e-5
require_formal_value WEIGHT_DECAY "${WEIGHT_DECAY}" 0.05
require_formal_value WARMUP_RATIO "${WARMUP_RATIO}" 0.1
require_formal_value MAX_SEQ_LENGTH "${MAX_SEQ_LENGTH}" 12288
require_formal_value MAX_DYNAMIC_PATCH "${MAX_DYNAMIC_PATCH}" 16
require_formal_value DROP_PATH_RATE "${DROP_PATH_RATE}" 0.1
require_formal_value MAX_STEPS "${MAX_STEPS}" -1
require_formal_value SAVE_STRATEGY "${SAVE_STRATEGY}" steps
require_formal_value SAVE_STEPS "${SAVE_STEPS}" 200
require_formal_value SAVE_TOTAL_LIMIT "${SAVE_TOTAL_LIMIT}" 10

if [[ ! -d "${BASE_VLM_PATH}" ]]; then
  echo "Official Stage1 base VLM not found: ${BASE_VLM_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE) != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NPROC_PER_NODE))

cd "${VLA_AD_ROOT}"
mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}.launch.lock"
if ! flock -n 9; then
  echo "Another Stage1 launcher owns ${OUTPUT_DIR}.launch.lock" >&2
  exit 75
fi

gate_args=(
  --manifest "${REPRODUCTION_GATE}"
  --target stage1
  --base-vlm-model "${BASE_VLM_PATH}/model.safetensors"
  --report "${OUTPUT_DIR}/reproduction_gate.json"
)
if [[ "${GATE_FULL_HASH}" == "1" ]]; then
  gate_args+=(--full-hash)
fi
python scripts/bench2drive/check_recogdrive_b2d_reproduction_gate.py "${gate_args[@]}" >/dev/null

python scripts/bench2drive/prepare_recogdrive_b2d_official_stage1.py \
  --output-dir "${STAGE1_DATA_DIR}" \
  --max-dynamic-patch "${MAX_DYNAMIC_PATCH}" \
  --smoke-records-per-dataset "${SMOKE_RECORDS_PER_DATASET}" >/dev/null

if [[ "${RUN_MODE}" == "formal" ]]; then
  TRAIN_META=${STAGE1_DATA_DIR}/official_meta.json
else
  TRAIN_META=${STAGE1_DATA_DIR}/smoke_meta.json
fi
if [[ ! -f "${TRAIN_META}" ]]; then
  echo "Prepared Stage1 meta missing: ${TRAIN_META}" >&2
  exit 2
fi

environment_args=(
  --min-gpus "${NPROC_PER_NODE}"
  --report "${OUTPUT_DIR}/environment_preflight.json"
)
if [[ "${ALLOW_ENVIRONMENT_VERSION_MISMATCH}" == "1" ]]; then
  environment_args+=(--allow-version-mismatch)
fi
env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  "${PYTHON_RUN[@]}" \
  "${VLA_AD_ROOT}/scripts/bench2drive/check_recogdrive_b2d_stage1_environment.py" \
  "${environment_args[@]}" >/dev/null
"${PYTHON_RUN[@]}" -m pip freeze --all > "${OUTPUT_DIR}/pip_freeze.txt"

if [[ "${SKIP_DATA_PREFLIGHT}" != "1" ]]; then
  env PYTHONPATH="${VLA_AD_ROOT}/internvl_chat${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_RUN[@]}" \
    "${VLA_AD_ROOT}/scripts/bench2drive/preflight_recogdrive_b2d_official_stage1.py" \
      --meta "${STAGE1_DATA_DIR}/smoke_meta.json" \
      --base-vlm "${BASE_VLM_PATH}" \
      --max-seq-length "${MAX_SEQ_LENGTH}" \
      --max-dynamic-patch "${MAX_DYNAMIC_PATCH}" \
      --report "${OUTPUT_DIR}/data_preflight.json" >/dev/null
fi

GIT_COMMIT=$(git rev-parse HEAD)
cat > "${OUTPUT_DIR}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=${GIT_COMMIT}
classification=closest-public official-checkpoint-anchored Bench2Drive Stage1
run_mode=${RUN_MODE}
train_environment=${TRAIN_ENV_DESCRIPTION}
train_python=${TRAIN_PYTHON:-conda:${NAVSIM_ENV}}
torchrun_bin=${TORCHRUN_BIN}
base_vlm_path=${BASE_VLM_PATH}
reproduction_gate=${REPRODUCTION_GATE}
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
max_dynamic_patch_total=${MAX_DYNAMIC_PATCH}
num_camera_views=6
max_dynamic_patch_per_view=2
thumbnail_per_view=1
expected_tiles_per_sample=18
drop_path_rate=${DROP_PATH_RATE}
max_steps=${MAX_STEPS}
save_strategy=${SAVE_STRATEGY}
deepspeed_config=${DEEPSPEED_CONFIG}
seed=${SEED}
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
  --drop_path_rate "${DROP_PATH_RATE}"
  --freeze_llm False
  --freeze_mlp False
  --freeze_backbone False
  --vision_select_layer -1
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --bf16 True
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --evaluation_strategy no
  --save_strategy "${SAVE_STRATEGY}"
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
  --use_data_resampling False
  --ps_version v2
  --deepspeed "${DEEPSPEED_CONFIG}"
  --report_to "${REPORT_TO}"
  --seed "${SEED}"
  --data_seed "${SEED}"
  --max_steps "${MAX_STEPS}"
  --ddp_find_unused_parameters False
)

cmd=("${TORCHRUN_BIN}" --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" "${args[@]}")
printf '%q ' "${cmd[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run prepared: ${OUTPUT_DIR}/launch_command.txt"
  exit 0
fi

cd "${VLA_AD_ROOT}/internvl_chat"
env CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTHONPATH="${VLA_AD_ROOT}:${VLA_AD_ROOT}/internvl_chat${PYTHONPATH:+:${PYTHONPATH}}" \
  LAUNCHER=pytorch \
  "${TRAIN_ENV_PREFIX[@]}" \
  "${cmd[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

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
