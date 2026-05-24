#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the NAVSIM experiment/cache root.}"
: "${RECOGDRIVE_VLM_PATH:?Set RECOGDRIVE_VLM_PATH to the ReCogDrive VLM checkpoint/model path.}"
: "${RECOGDRIVE_EVAL_CHECKPOINT:?Set RECOGDRIVE_EVAL_CHECKPOINT to the IL or RL checkpoint to evaluate.}"

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

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
EXPERT_VARIANT=${EXPERT_VARIANT:-jepa_vggt}
RECOGDRIVE_METRIC_CACHE_DIR=${RECOGDRIVE_METRIC_CACHE_DIR:-${NAVSIM_EXP_ROOT}/metric_cache}
RECOGDRIVE_NUM_JEPA_TOKENS=${RECOGDRIVE_NUM_JEPA_TOKENS:-4}
RECOGDRIVE_NUM_VGGT_TOKENS=${RECOGDRIVE_NUM_VGGT_TOKENS:-4}
RECOGDRIVE_JEPA_DIM=${RECOGDRIVE_JEPA_DIM:-768}
RECOGDRIVE_VGGT_DIM=${RECOGDRIVE_VGGT_DIM:-2048}

case "${EXPERT_VARIANT}" in
  baseline) AGENT_CONFIG=recogdrive_agent_expert_baseline ;;
  jepa) AGENT_CONFIG=recogdrive_agent_expert_jepa ;;
  vggt) AGENT_CONFIG=recogdrive_agent_expert_vggt ;;
  jepa_vggt) AGENT_CONFIG=recogdrive_agent_expert_jepa_vggt ;;
  jepa_vggt_alignment) AGENT_CONFIG=recogdrive_agent_expert_jepa_vggt_alignment ;;
  *) echo "Unknown EXPERT_VARIANT=${EXPERT_VARIANT}. Expected baseline|jepa|vggt|jepa_vggt|jepa_vggt_alignment." >&2; exit 2 ;;
esac

if [[ "${EXPERT_VARIANT}" != "baseline" ]]; then
  : "${RECOGDRIVE_EXPERT_CACHE_DIR:?Set RECOGDRIVE_EXPERT_CACHE_DIR for expert variants.}"
fi

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-${GPUS}}
MASTER_PORT=${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}
LOG_FILE=${LOG_FILE:-${NAVSIM_EXP_ROOT}/eval_recogdrive_${EXPERT_VARIANT}_expert.txt}

torchrun \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "agent=${AGENT_CONFIG}" \
  "agent.checkpoint_path=${RECOGDRIVE_EVAL_CHECKPOINT}" \
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH}" \
  "agent.cam_type=single" \
  "agent.grpo=False" \
  "agent.cache_hidden_state=False" \
  "agent.vlm_type=internvl" \
  "agent.dit_type=small" \
  "agent.vlm_size=small" \
  "agent.sampling_method=ddim" \
  "agent.expert_cache_dir=${RECOGDRIVE_EXPERT_CACHE_DIR}" \
  "agent.num_jepa_tokens=${RECOGDRIVE_NUM_JEPA_TOKENS}" \
  "agent.num_vggt_tokens=${RECOGDRIVE_NUM_VGGT_TOKENS}" \
  "agent.jepa_dim=${RECOGDRIVE_JEPA_DIM}" \
  "agent.vggt_dim=${RECOGDRIVE_VGGT_DIM}" \
  "metric_cache_path=${RECOGDRIVE_METRIC_CACHE_DIR}" \
  "experiment_name=recogdrive_${EXPERT_VARIANT}_expert_eval" > "${LOG_FILE}" 2>&1
