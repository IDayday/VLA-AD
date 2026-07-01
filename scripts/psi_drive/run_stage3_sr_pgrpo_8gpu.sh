#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD/outputs}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
STAGE2_CKPT="${PSI_STAGE2_INIT_CKPT:-}"
CACHE_PATH="${RECOGDRIVE_OFFICIAL_STAGE2_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
DEFAULT_SUPPLEMENTED_SUPPORT_INDEX="/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt"
DEFAULT_CLEAN_SUPPORT_INDEX="/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt"
if [[ -n "${PSI_STAGE2_SUPPORT_INDEX:-}" ]]; then
  SUPPORT_INDEX="${PSI_STAGE2_SUPPORT_INDEX}"
elif [[ -f "${DEFAULT_SUPPLEMENTED_SUPPORT_INDEX}" ]]; then
  SUPPORT_INDEX="${DEFAULT_SUPPLEMENTED_SUPPORT_INDEX}"
else
  SUPPORT_INDEX="${DEFAULT_CLEAN_SUPPORT_INDEX}"
fi
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/mnt/project/VLA-AD/cache/metric_cache_train_full}"
REFERENCE_CKPT="${REFERENCE_POLICY_CHECKPOINT:-${STAGE2_CKPT}}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/trainval_navsim_logs/${SPLIT}}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/trainval_sensor_blobs/${SPLIT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-psi_drive_stage3_sr_pgrpo_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train.log}"

GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"
NODES="${NODES:-$((GPUS / GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63689}}"
DDP_STRATEGY="${DDP_STRATEGY:-ddp}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${MAX_EPOCHS:-20}}"
GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}"
GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}"
GRPO_CORE_PARETO_DUAL_LR="${GRPO_CORE_PARETO_DUAL_LR:-0.02}"

if [[ -z "${STAGE2_CKPT}" || ! -f "${STAGE2_CKPT}" ]]; then
  echo "PSI_STAGE2_INIT_CKPT must point to the val6000-selected APSD Stage2 checkpoint." >&2
  exit 2
fi
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "ReCogDrive official VLM path does not exist: ${VLM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${CACHE_PATH}" ]]; then
  echo "Official ReCogDrive Stage1 hidden cache path does not exist: ${CACHE_PATH}" >&2
  exit 2
fi
if [[ ! -f "${SUPPORT_INDEX}" ]]; then
  echo "Stage2 Pareto support index does not exist: ${SUPPORT_INDEX}" >&2
  exit 2
fi
if [[ ! -d "${METRIC_CACHE_PATH}" ]]; then
  echo "Metric cache path does not exist: ${METRIC_CACHE_PATH}" >&2
  exit 2
fi
if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
  echo "NAVSIM_LOG_PATH does not exist: ${NAVSIM_LOG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
  echo "SENSOR_BLOBS_PATH does not exist: ${SENSOR_BLOBS_PATH}" >&2
  exit 2
fi

mkdir -p \
  "${OUTPUT_DIR}/logs" \
  "${OUTPUT_DIR}/checkpoints/raw" \
  "${OUTPUT_DIR}/checkpoints/val_loss_top5" \
  "${OUTPUT_DIR}/checkpoint_store/objects" \
  "${OUTPUT_DIR}/rankings/val6000/history" \
  "${OUTPUT_DIR}/rankings/navtest/history" \
  "${OUTPUT_DIR}/eval/val6000" \
  "${OUTPUT_DIR}/eval/navtest" \
  "${OUTPUT_DIR}/state"

{
  echo "experiment_name=${EXPERIMENT_NAME}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "stage2_checkpoint=${STAGE2_CKPT}"
  echo "support_index=${SUPPORT_INDEX}"
  echo "cache_path=${CACHE_PATH}"
  echo "metric_cache_path=${METRIC_CACHE_PATH}"
  echo "lr=${LR}"
  echo "max_epochs=${MAX_EPOCHS:-20}"
  echo "gpus=${GPUS}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "batch_size=${BATCH_SIZE}"
  echo "accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  echo "effective_batch=$((GPUS_PER_NODE * BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"
  echo "grpo_sample_time=16"
  echo "grpo_reward_mode=core_pareto"
  echo "grpo_use_core_pareto=true"
  echo "grpo_use_support_relative=true"
  echo "grpo_core_pareto_dual_lr=${GRPO_CORE_PARETO_DUAL_LR}"
  echo "grpo_scheduler_epochs=${GRPO_SCHEDULER_EPOCHS}"
  echo "grpo_scheduler_warmup_epochs=${GRPO_SCHEDULER_WARMUP_EPOCHS}"
  echo "grpo_scheduler_min_lr=${GRPO_SCHEDULER_MIN_LR}"
  echo "reference_kl_coeff=0.02"
  echo "bc_coeff=0.10->0.05"
  echo "bc_anneal_epochs=5"
  echo "ddp_strategy=${DDP_STRATEGY}"
  echo "use_cache_without_dataset=True"
  echo "force_cache_computation=False"
} > "${OUTPUT_DIR}/sr_pgrpo_launch_config.txt"

CMD=(
  torchrun
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py"
  "agent=recogdrive_agent"
  "agent.lr=${LR}"
  "agent.checkpoint_path=${STAGE2_CKPT}"
  "agent.vlm_path=${VLM_PATH}"
  "agent.cam_type=single"
  "agent.cache_hidden_state=True"
  "agent.vlm_type=internvl"
  "agent.dit_type=small"
  "agent.vlm_size=small"
  "agent.sampling_method=ddim"
  "agent.train_backbone=false"
  "agent.use_expert_features=false"
  "agent.expert_feature_source=none"
  "agent.allow_expert_target_features=false"
  "agent.use_jepa=false"
  "agent.use_vggt=false"
  "agent.use_last_rd=false"
  "agent.last_rd_stage=disabled"
  "agent.use_last_vla=false"
  "agent.last_vla_stage=disabled"
  "agent.use_two_expert_slots=false"
  "agent.grpo=True"
  "agent.stage3_objective=grpo"
  "agent.grpo_sample_time=16"
  "agent.grpo_reward_mode=core_pareto"
  "agent.grpo_use_core_pareto=true"
  "agent.grpo_normalize_advantage_batch=false"
  "agent.grpo_use_support_relative=true"
  "agent.grpo_support_index_path=${SUPPORT_INDEX}"
  "agent.grpo_use_rms_advantage_scale=true"
  "agent.grpo_reapply_final_caps=true"
  "agent.grpo_core_pareto_dual_lr=${GRPO_CORE_PARETO_DUAL_LR}"
  "agent.grpo_scheduler_epochs=${GRPO_SCHEDULER_EPOCHS}"
  "agent.grpo_scheduler_warmup_epochs=${GRPO_SCHEDULER_WARMUP_EPOCHS}"
  "agent.grpo_scheduler_min_lr=${GRPO_SCHEDULER_MIN_LR}"
  "agent.reference_kl_coeff=0.02"
  "agent.bc_anneal=true"
  "agent.bc_coeff_start=0.10"
  "agent.bc_coeff_end=0.05"
  "agent.bc_anneal_epochs=5"
  "agent.metric_cache_path=${METRIC_CACHE_PATH}"
  "agent.reference_policy_checkpoint=${REFERENCE_CKPT}"
  "trainer.params.max_epochs=${MAX_EPOCHS:-20}"
  "trainer.params.num_nodes=${NODES}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "trainer.params.strategy=${DDP_STRATEGY}"
  "trainer.params.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  "dataloader.params.batch_size=${BATCH_SIZE}"
  "dataloader.params.num_workers=${NUM_WORKERS}"
  "checkpoint.layout=psi"
  "checkpoint.every_n_epochs=${CHECKPOINT_EVERY_N_EPOCHS:-1}"
  "checkpoint.every_n_train_steps=${CHECKPOINT_EVERY_N_TRAIN_STEPS:-300}"
  "experiment_name=${EXPERIMENT_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "cache_path=${CACHE_PATH}"
  "use_cache_without_dataset=True"
  "force_cache_computation=False"
)

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/resolved_command.sh"
printf '\n' >> "${OUTPUT_DIR}/resolved_command.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat "${OUTPUT_DIR}/resolved_command.sh"
  exit 0
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1
