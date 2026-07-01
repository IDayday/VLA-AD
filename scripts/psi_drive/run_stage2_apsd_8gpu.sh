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
STAGE2_INIT_MODE="${PSI_STAGE2_INIT_MODE:-official}"
DEFAULT_STAGE2_INIT_CKPT="/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt"
if [[ "${STAGE2_INIT_MODE}" == "random" ]]; then
  STAGE2_INIT_CKPT=""
else
  STAGE2_INIT_CKPT="${RECOGDRIVE_STAGE2_INIT_CKPT:-${DEFAULT_STAGE2_INIT_CKPT}}"
fi
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
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/trainval_navsim_logs/${SPLIT}}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/trainval_sensor_blobs/${SPLIT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-psi_drive_stage2_apsd_${STAGE2_INIT_MODE}_init_clean_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train.log}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"

GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"
NODES="${NODES:-$((GPUS / GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63679}}"

case "${STAGE2_INIT_MODE}" in
  official|random) ;;
  *)
    echo "PSI_STAGE2_INIT_MODE must be either 'official' or 'random'; got '${STAGE2_INIT_MODE}'." >&2
    exit 2
    ;;
esac
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "ReCogDrive official VLM path does not exist: ${VLM_PATH}" >&2
  exit 2
fi
if [[ "${STAGE2_INIT_MODE}" == "official" && ! -f "${STAGE2_INIT_CKPT}" ]]; then
  echo "ReCogDrive official Stage2 init checkpoint does not exist: ${STAGE2_INIT_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${CACHE_PATH}" ]]; then
  echo "Stage2 cache path does not exist: ${CACHE_PATH}" >&2
  exit 2
fi
if [[ ! -f "${SUPPORT_INDEX}" ]]; then
  echo "Stage2 Pareto support index does not exist: ${SUPPORT_INDEX}" >&2
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

CMD=(
  torchrun
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py"
  "agent=recogdrive_agent"
  "agent.checkpoint_path=${STAGE2_INIT_CKPT}"
  "agent.allow_random_init=true"
  "agent.lr=${LR}"
  "agent.grpo=False"
  "agent.stage3_objective=none"
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
  "trainer.params.max_epochs=${MAX_EPOCHS}"
  "trainer.params.num_nodes=${NODES}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "dataloader.params.batch_size=${BATCH_SIZE}"
  "dataloader.params.num_workers=${NUM_WORKERS}"
  "checkpoint.layout=psi"
  "checkpoint.every_n_epochs=${CHECKPOINT_EVERY_N_EPOCHS:-1}"
  "checkpoint.every_n_train_steps=${CHECKPOINT_EVERY_N_TRAIN_STEPS:-0}"
  "experiment_name=${EXPERIMENT_NAME}"
  "output_dir=${OUTPUT_DIR}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "cache_path=${CACHE_PATH}"
  "use_cache_without_dataset=True"
  "force_cache_computation=False"
  "cache_train_all_records=${CACHE_TRAIN_ALL_RECORDS:-false}"
  "stage2_target_source=pareto_support"
  "stage2_pareto_support_index_path=${SUPPORT_INDEX}"
  "stage2_pareto_require_index=true"
  "stage2_pareto_log_diagnostics=true"
)

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/resolved_command.sh"
printf '\n' >> "${OUTPUT_DIR}/resolved_command.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cat "${OUTPUT_DIR}/resolved_command.sh"
  exit 0
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1
