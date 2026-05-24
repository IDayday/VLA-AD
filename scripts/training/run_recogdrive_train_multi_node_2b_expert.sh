#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the NAVSIM experiment/cache root.}"
: "${RECOGDRIVE_VLM_PATH:?Set RECOGDRIVE_VLM_PATH to the ReCogDrive VLM checkpoint/model path.}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export OPENSCENE_DATA_ROOT
export NAVSIM_EXP_ROOT
export RECOGDRIVE_EXPERT_CACHE_DIR="${RECOGDRIVE_EXPERT_CACHE_DIR:-${NAVSIM_EXP_ROOT}/recogdrive_expert_cache}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
EXPERT_VARIANT=${EXPERT_VARIANT:-jepa_vggt}
RECOGDRIVE_HIDDEN_CACHE_DIR=${RECOGDRIVE_HIDDEN_CACHE_DIR:-${NAVSIM_EXP_ROOT}/recogdrive_agent_cache_dir_train_2b}
RECOGDRIVE_NUM_JEPA_TOKENS=${RECOGDRIVE_NUM_JEPA_TOKENS:-4}
RECOGDRIVE_NUM_VGGT_TOKENS=${RECOGDRIVE_NUM_VGGT_TOKENS:-4}
RECOGDRIVE_JEPA_DIM=${RECOGDRIVE_JEPA_DIM:-768}
RECOGDRIVE_VGGT_DIM=${RECOGDRIVE_VGGT_DIM:-2048}
ALLOW_DUMMY_EXPERT_CACHE=${ALLOW_DUMMY_EXPERT_CACHE:-false}
MAX_EPOCHS=${MAX_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}

case "${EXPERT_VARIANT}" in
  baseline) AGENT_CONFIG=recogdrive_agent_expert_baseline; JEPA_ALIGN=${RECOGDRIVE_JEPA_ALIGNMENT_WEIGHT:-0.0}; VGGT_ALIGN=${RECOGDRIVE_VGGT_ALIGNMENT_WEIGHT:-0.0} ;;
  jepa) AGENT_CONFIG=recogdrive_agent_expert_jepa; JEPA_ALIGN=${RECOGDRIVE_JEPA_ALIGNMENT_WEIGHT:-0.0}; VGGT_ALIGN=${RECOGDRIVE_VGGT_ALIGNMENT_WEIGHT:-0.0} ;;
  vggt) AGENT_CONFIG=recogdrive_agent_expert_vggt; JEPA_ALIGN=${RECOGDRIVE_JEPA_ALIGNMENT_WEIGHT:-0.0}; VGGT_ALIGN=${RECOGDRIVE_VGGT_ALIGNMENT_WEIGHT:-0.0} ;;
  jepa_vggt) AGENT_CONFIG=recogdrive_agent_expert_jepa_vggt; JEPA_ALIGN=${RECOGDRIVE_JEPA_ALIGNMENT_WEIGHT:-0.0}; VGGT_ALIGN=${RECOGDRIVE_VGGT_ALIGNMENT_WEIGHT:-0.0} ;;
  jepa_vggt_alignment) AGENT_CONFIG=recogdrive_agent_expert_jepa_vggt_alignment; JEPA_ALIGN=${RECOGDRIVE_JEPA_ALIGNMENT_WEIGHT:-0.1}; VGGT_ALIGN=${RECOGDRIVE_VGGT_ALIGNMENT_WEIGHT:-0.1} ;;
  *) echo "Unknown EXPERT_VARIANT=${EXPERT_VARIANT}. Expected baseline|jepa|vggt|jepa_vggt|jepa_vggt_alignment." >&2; exit 2 ;;
esac

if [[ "${EXPERT_VARIANT}" != "baseline" ]]; then
  : "${RECOGDRIVE_EXPERT_CACHE_DIR:?Set RECOGDRIVE_EXPERT_CACHE_DIR for expert variants.}"
fi

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-${GPUS}}
NODES=${NODES:-$((GPUS / GPUS_PER_NODE))}
NODE_RANK=${NODE_RANK:-${MLP_ROLE_INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}
LOG_FILE=${LOG_FILE:-${NAVSIM_EXP_ROOT}/train_recogdrive_${EXPERT_VARIANT}_expert_il.txt}

torchrun \
  --nnodes="${NODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  "agent=${AGENT_CONFIG}" \
  "agent.lr=${LR}" \
  "agent.grpo=False" \
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH}" \
  "agent.cam_type=single" \
  "agent.cache_hidden_state=True" \
  "agent.vlm_type=internvl" \
  "agent.dit_type=small" \
  "agent.vlm_size=small" \
  "agent.sampling_method=ddim" \
  "agent.expert_cache_dir=${RECOGDRIVE_EXPERT_CACHE_DIR}" \
  "agent.num_jepa_tokens=${RECOGDRIVE_NUM_JEPA_TOKENS}" \
  "agent.num_vggt_tokens=${RECOGDRIVE_NUM_VGGT_TOKENS}" \
  "agent.jepa_dim=${RECOGDRIVE_JEPA_DIM}" \
  "agent.vggt_dim=${RECOGDRIVE_VGGT_DIM}" \
  "agent.jepa_alignment_weight=${JEPA_ALIGN}" \
  "agent.vggt_alignment_weight=${VGGT_ALIGN}" \
  "trainer.params.max_epochs=${MAX_EPOCHS}" \
  "trainer.params.num_nodes=${NODES}" \
  "trainer.params.devices=${GPUS_PER_NODE}" \
  "dataloader.params.batch_size=${BATCH_SIZE}" \
  "experiment_name=training_recogdrive_${EXPERT_VARIANT}_expert_il" \
  "allow_dummy_expert_cache=${ALLOW_DUMMY_EXPERT_CACHE}" \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "cache_path=${RECOGDRIVE_HIDDEN_CACHE_DIR}" \
  "use_cache_without_dataset=True" \
  "force_cache_computation=False" > "${LOG_FILE}" 2>&1
