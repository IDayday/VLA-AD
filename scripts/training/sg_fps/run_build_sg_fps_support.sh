#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"

CACHE_PATH="${CACHE_PATH:-${ARTIFACT_ROOT}/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH to the SG-FPS v3 support archive directory}"
OUT_ROOT="${OUT_ROOT:-$(dirname "${OUTPUT_PATH}")/sg_fps_support_build}"
BUILD_LOG_SPLIT="${BUILD_LOG_SPLIT:-train}"
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
export SG_FPS_EXPAND_EXTERNAL_CANDIDATES="${SG_FPS_EXPAND_EXTERNAL_CANDIDATES:-true}"
export SG_FPS_EXTERNAL_EXPANSION_MAX_PER_SCENE="${SG_FPS_EXTERNAL_EXPANSION_MAX_PER_SCENE:-32}"
export SG_FPS_SELECTION_STRATEGY="${SG_FPS_SELECTION_STRATEGY:-quality_pareto}"
export SG_FPS_ENABLE_TRAIN_QUALITY_GATE="${SG_FPS_ENABLE_TRAIN_QUALITY_GATE:-true}"
export SG_FPS_ENABLE_SEMANTIC_GATE="${SG_FPS_ENABLE_SEMANTIC_GATE:-true}"
export SG_FPS_ALLOW_TURN_CLASS_MISMATCH="${SG_FPS_ALLOW_TURN_CLASS_MISMATCH:-false}"
export SG_FPS_MAX_SEMANTIC_FINAL_HEADING_ERROR_RAD="${SG_FPS_MAX_SEMANTIC_FINAL_HEADING_ERROR_RAD:-0.75}"
export SG_FPS_MAX_SEMANTIC_PATH_ANGLE_ERROR_RAD="${SG_FPS_MAX_SEMANTIC_PATH_ANGLE_ERROR_RAD:-0.75}"
export SG_FPS_MAX_SEMANTIC_ENDPOINT_LATERAL_ERROR_M="${SG_FPS_MAX_SEMANTIC_ENDPOINT_LATERAL_ERROR_M:-4.0}"
export SG_FPS_DDC_GATE_MODE="${SG_FPS_DDC_GATE_MODE:-ref_relative}"
export SG_FPS_RELAX_DDC_WHEN_REF_BELOW_MIN="${SG_FPS_RELAX_DDC_WHEN_REF_BELOW_MIN:-true}"
export SG_FPS_FEAS_GATE_MODE="${SG_FPS_FEAS_GATE_MODE:-relax_ref_above_max}"
export SG_FPS_FEAS_COST_TOLERANCE="${SG_FPS_FEAS_COST_TOLERANCE:-0.03}"
export SG_FPS_COMFORT_GATE_MODE="${SG_FPS_COMFORT_GATE_MODE:-ref_relative}"
export SG_FPS_COMFORT_DROP_TOLERANCE="${SG_FPS_COMFORT_DROP_TOLERANCE:-0.05}"
export SG_FPS_REWARD_GATE_MODE="${SG_FPS_REWARD_GATE_MODE:-absolute_or_gt_improver}"
export SG_FPS_MIN_NON_GT_IMPROVER_PDMS="${SG_FPS_MIN_NON_GT_IMPROVER_PDMS:-0.70}"
export SG_FPS_GT_IMPROVER_MARGIN="${SG_FPS_GT_IMPROVER_MARGIN:-0.05}"
export SG_FPS_GT_IMPROVER_REF_MAX_PDMS="${SG_FPS_GT_IMPROVER_REF_MAX_PDMS:-0.90}"
export MAX_SCENES="${MAX_SCENES:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-false}"
export EXTERNAL_CANDIDATE_ROOTS
export BUILD_LOG_SPLIT
export ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES:-0}"
export ONLINE_USE_CURRENT_POLICY="${ONLINE_USE_CURRENT_POLICY:-false}"
export ONLINE_USE_OLD_POLICY="${ONLINE_USE_OLD_POLICY:-false}"
export ONLINE_USE_GT="${ONLINE_USE_GT:-true}"
export PERTURB_GT="${PERTURB_GT:-true}"
export PERTURB_IL="${PERTURB_IL:-false}"

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
