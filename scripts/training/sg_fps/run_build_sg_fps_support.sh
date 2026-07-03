#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"

CACHE_PATH="${CACHE_PATH:-${ARTIFACT_ROOT}/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH to the SG-FPS v3 support archive directory}"
OUT_ROOT="${OUT_ROOT:-$(dirname "${OUTPUT_PATH}")/sg_fps_support_build}"
IL_CHECKPOINT="${IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
VLM_PATH="${VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
EXTERNAL_CANDIDATE_ROOTS="${EXTERNAL_CANDIDATE_ROOTS:-}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export OUT_ROOT
export ELITE_BUFFER_DIR="${OUTPUT_PATH}"
export RUN_STAGE3=1
export WRITE_SG_FPS_V3=1
export SG_FPS_SUPPORT_TOP_M="${SG_FPS_SUPPORT_TOP_M:-12}"
export MAX_SCENES="${MAX_SCENES:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-false}"
export EXTERNAL_CANDIDATE_ROOTS

"${PYTHON_BIN}" scripts/training/build_recogdrive_stage3_awac_elite_buffer.py \
  train_test_split=navtrain \
  output_dir="${OUT_ROOT}/hydra" \
  cache_path="${CACHE_PATH}" \
  force_cache_computation=False \
  use_cache_without_dataset=True \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS:-0}" \
  agent=recogdrive_agent \
  agent.cache_hidden_state=True \
  agent.cache_mode=True \
  agent.vlm_path="${VLM_PATH}" \
  agent.checkpoint_path="${IL_CHECKPOINT}" \
  agent.reference_policy_checkpoint="${IL_CHECKPOINT}" \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim
