#!/usr/bin/env bash
set -Eeuo pipefail

required=(NAVSIM_LOG_PATH SENSOR_BLOBS_PATH OUTPUT_DIR MASTER_PORT VLM_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

EXPERT_CACHE_ROOT="${EXPERT_TEACHER_CACHE_ROOT:-${TRAIN_CHUNK_CACHE_ROOT:-}}"
if [[ -z "${EXPERT_CACHE_ROOT}" ]]; then
  echo "Set EXPERT_TEACHER_CACHE_ROOT or TRAIN_CHUNK_CACHE_ROOT to the full-geometry teacher cache." >&2
  exit 2
fi
[[ -d "${EXPERT_CACHE_ROOT}" ]] || { echo "Teacher cache missing: ${EXPERT_CACHE_ROOT}" >&2; exit 2; }
[[ -e "${VLM_PATH}" || -d "${VLM_PATH}" ]] || { echo "VLM_PATH missing: ${VLM_PATH}" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SEED="${SEED:-0}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/last_vla_vlm_lora_cot_alignment.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_vlm_lora_cot_alignment
  use_cache_without_dataset=false
  force_cache_computation=false
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "output_dir=${OUTPUT_DIR}"
  seed="${SEED}"
  agent.cache_hidden_state=false
  agent.train_backbone=true
  agent.last_vla_train_vlm_lora=true
  agent.vlm_path="${VLM_PATH}"
  agent.vlm_type="${VLM_TYPE:-internvl}"
  agent.use_expert_features=true
  agent.expert_feature_source=chunk
  agent.expert_cache_dir="${EXPERT_CACHE_ROOT}"
  agent.allow_expert_target_features=true
  agent.use_last_vla=true
  agent.last_vla_stage=cot_alignment
  agent.use_last_rd=false
  agent.last_vla_require_full_geometry=true
  agent.last_vla_allow_patch_geometry_fallback=false
  agent.last_vla_geometry_teacher_dim="${GEOMETRY_TEACHER_DIM:-512}"
  agent.diffusion_loss_weight=0.0
  agent.policy_kd_loss_weight=0.0
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
if [[ -n "${A0_INIT_CHECKPOINT}" ]]; then
  [[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }
  cmd+=(agent.checkpoint_path="${A0_INIT_CHECKPOINT}")
fi

{
  date -Is
  printf 'NAVSIM_LOG_PATH=%s\n' "${NAVSIM_LOG_PATH}"
  printf 'SENSOR_BLOBS_PATH=%s\n' "${SENSOR_BLOBS_PATH}"
  printf 'EXPERT_CACHE_ROOT=%s\n' "${EXPERT_CACHE_ROOT}"
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi

echo "Starting Last-VLA VLM-LoRA CoT alignment. Log: ${TRAIN_LOG}"
set +e
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "${status}" >>"${COMMANDS_LOG}"
exit "${status}"
