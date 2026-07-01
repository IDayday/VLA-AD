#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/navsim}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD/outputs}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SPLIT="${SPLIT:-trainval}"
VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
BASE_CACHE_PATH="${RECOGDRIVE_OFFICIAL_HIDDEN_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
USE_ELITE_TARGETS="${USE_ELITE_TARGETS:-1}"
ELITE_CACHE_PATH="${RECOGDRIVE_OFFICIAL_HIDDEN_ELITE_CACHE_DIR:-${BASE_CACHE_PATH}_elite_targets}"
DEFAULT_CACHE_PATH="${BASE_CACHE_PATH}"
if [[ "${USE_ELITE_TARGETS}" == "1" ]]; then
  DEFAULT_CACHE_PATH="${ELITE_CACHE_PATH}"
fi
CACHE_PATH="${RECOGDRIVE_OFFICIAL_STAGE2_CACHE_DIR:-${DEFAULT_CACHE_PATH}}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/trainval_navsim_logs/${SPLIT}}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/trainval_sensor_blobs/${SPLIT}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-training_recogdrive_official_stage2_2b}"
LOG_FILE="${LOG_FILE:-${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}.log}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
CACHE_AUDIT_MAX_SAMPLES="${CACHE_AUDIT_MAX_SAMPLES:-16}"
CACHE_AUDIT_MIN_SAMPLES="${CACHE_AUDIT_MIN_SAMPLES:-1}"

GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${GPUS}}"
NODES="${NODES:-$((GPUS / GPUS_PER_NODE))}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"

if [[ ! -d "${VLM_PATH}" ]]; then
  echo "ReCogDrive official VLM path does not exist: ${VLM_PATH}" >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" && "${SKIP_CACHE_AUDIT:-0}" != "1" ]]; then
  python "${REPO_ROOT}/scripts/cache_dataset/audit_recogdrive_official_hidden_cache.py" \
    --cache-path "${CACHE_PATH}" \
    --expected-vlm-path "${VLM_PATH}" \
    --max-samples "${CACHE_AUDIT_MAX_SAMPLES}" \
    --require-min-samples "${CACHE_AUDIT_MIN_SAMPLES}" \
    --require-official
fi

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
  "agent.grpo=False"
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
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "cache_path=${CACHE_PATH}"
  "use_cache_without_dataset=True"
  "force_cache_computation=False"
  "cache_train_all_records=${CACHE_TRAIN_ALL_RECORDS:-false}"
  "stage2_target_source=gt"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "${LOG_FILE}")"
"${CMD[@]}" > "${LOG_FILE}" 2>&1
