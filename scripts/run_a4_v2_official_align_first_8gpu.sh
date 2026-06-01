#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  RECOGDRIVE_VLM_PATH
  CACHE_PATH
  TRAIN_TEST_SPLIT
  OUTPUT_DIR
  MASTER_PORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}. Set ALLOW_OVERWRITE=1 to reuse it." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
SEED="${SEED:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-200}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TRAIN_PRECISION="${TRAIN_PRECISION:-16-mixed}"
KEY_STEPS="${KEY_STEPS:-50000,60000,80000,100000,120000,140000,160000}"

STAGE1_DIR="${OUTPUT_DIR}/stage1_align"
STAGE2_DIR="${OUTPUT_DIR}/stage2_noalign"
mkdir -p "${OUTPUT_DIR}/logs" "${STAGE1_DIR}" "${STAGE2_DIR}"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"

run_stage() {
  local stage_name="$1"
  local output_dir="$2"
  local agent_config="$3"
  local max_epochs="$4"
  local master_port="$5"
  shift 5
  local log_file="${OUTPUT_DIR}/logs/${stage_name}.train.log"
  local cmd=(
    "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${master_port}"
    navsim/planning/script/run_training_recogdrive.py
    "agent=${agent_config}"
    "agent.vlm_path=${RECOGDRIVE_VLM_PATH}"
    agent.cam_type=single
    agent.cache_hidden_state=True
    agent.vlm_type=internvl
    agent.dit_type=small
    agent.vlm_size=small
    agent.sampling_method=ddim
    agent.use_expert_features=True
    agent.expert_feature_source=chunk
    agent.lr=1e-4
    "agent.scheduler_epochs=${max_epochs}"
    agent.scheduler_warmup_epochs=3
    agent.scheduler_min_lr=1e-6
    "trainer.params.max_epochs=${max_epochs}"
    trainer.params.devices=8
    "trainer.params.precision=${TRAIN_PRECISION}"
    "trainer.params.accumulate_grad_batches=${GRAD_ACC}"
    "trainer.params.default_root_dir=${output_dir}"
    "dataloader.params.batch_size=${PER_GPU_BATCH}"
    "dataloader.params.num_workers=${NUM_WORKERS}"
    "dataloader.params.prefetch_factor=${PREFETCH_FACTOR}"
    "train_test_split=${TRAIN_TEST_SPLIT}"
    "cache_path=${CACHE_PATH}"
    use_cache_without_dataset=True
    force_cache_computation=False
    "seed=${SEED}"
    "output_dir=${output_dir}"
    "experiment_name=${stage_name}"
    "$@"
  )
  {
    printf '[%s] RECOGDRIVE_KEY_STEPS=%q ' "$(date -Is)" "${KEY_STEPS}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${COMMANDS_LOG}"
  echo "Starting ${stage_name}. Log: ${log_file}"
  RECOGDRIVE_KEY_STEPS="${KEY_STEPS}" "${cmd[@]}" >"${log_file}" 2>&1
}

run_stage \
  a4_v2_stage1_align \
  "${STAGE1_DIR}" \
  recogdrive_agent_a4_v2_align_stage1 \
  "${STAGE1_EPOCHS}" \
  "${MASTER_PORT}"

run_stage \
  a4_v2_stage2_noalign \
  "${STAGE2_DIR}" \
  recogdrive_agent_a4_v2_stage2_noalign \
  "${STAGE2_EPOCHS}" \
  "$((MASTER_PORT + 1))" \
  "agent.checkpoint_path=${STAGE1_DIR}/latest.ckpt"
