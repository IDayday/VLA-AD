#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/navsim/bin/torchrun}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_rl_2b_safe_diffgrpo_online_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

IL_CHECKPOINT="${IL_CHECKPOINT:-${RECOGDRIVE_IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}}"
VLM_PATH="${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${RECOGDRIVE_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}}"
HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-${RECOGDRIVE_HIDDEN_CACHE_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_agent_cache_dir_train_2b}}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${NAVSIM_DATA_ROOT}/trainval_sensor_blobs/trainval}"

CACHE_MODE="${CACHE_MODE:-online}"  # online or offline
GPUS="${GPUS:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
if [[ -z "${MASTER_PORT:-}" && -z "${MLP_WORKER_0_PORT:-}" ]]; then
  MASTER_PORT="$("${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
else
  MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"
fi
DDP_STRATEGY="${DDP_STRATEGY:-ddp}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Stage 3 RL hyperparameters for the Safe DiffGRPO continuation.
STAGE3_LR="${LR:-1e-4}"
STAGE3_OBJECTIVE="${STAGE3_OBJECTIVE:-none}"
STAGE3_MAX_EPOCHS="${MAX_EPOCHS:-20}"
STAGE3_BATCH_SIZE="${BATCH_SIZE:-8}"
STAGE3_ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
STAGE3_GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-16}"
STAGE3_BC_ANNEAL="${BC_ANNEAL:-true}"
STAGE3_BC_COEFF_START="${BC_COEFF_START:-0.10}"
STAGE3_BC_COEFF_END="${BC_COEFF_END:-0.05}"
STAGE3_BC_ANNEAL_EPOCHS="${BC_ANNEAL_EPOCHS:-5}"
STAGE3_REFERENCE_KL_COEFF="${REFERENCE_KL_COEFF:-0.02}"
STAGE3_REFERENCE_KL_CHUNK_SIZE="${REFERENCE_KL_CHUNK_SIZE:-0}"
STAGE3_GRPO_USE_GSPO_RATIO="${GRPO_USE_GSPO_RATIO:-false}"
STAGE3_GRPO_GSPO_CLIP_LOW="${GRPO_GSPO_CLIP_LOW:-0.05}"
STAGE3_GRPO_GSPO_CLIP_HIGH="${GRPO_GSPO_CLIP_HIGH:-0.05}"
STAGE3_GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL="${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-4}"
STAGE3_GRPO_BEHAVIOR_POLICY_SAMPLE="${GRPO_BEHAVIOR_POLICY_SAMPLE:-true}"
STAGE3_GRPO_NORMALIZE_ADVANTAGE_BATCH="${GRPO_NORMALIZE_ADVANTAGE_BATCH:-false}"
STAGE3_GRPO_ADVANTAGE_CLIP_ABS="${GRPO_ADVANTAGE_CLIP_ABS:-0.0}"
STAGE3_GRPO_HARD_GATE_TTC="${GRPO_HARD_GATE_TTC:-false}"
STAGE3_GRPO_HARD_GATE_DDC="${GRPO_HARD_GATE_DDC:-false}"
STAGE3_GRPO_TTC_SAFE_THRESHOLD="${GRPO_TTC_SAFE_THRESHOLD:-1.0}"
STAGE3_GRPO_DDC_SAFE_THRESHOLD="${GRPO_DDC_SAFE_THRESHOLD:-1.0}"
STAGE3_GRPO_PPO_REPLAY_INNER_EPOCHS="${GRPO_PPO_REPLAY_INNER_EPOCHS:-1}"
STAGE3_GRPO_PPO_REPLAY_MINIBATCH_SIZE="${GRPO_PPO_REPLAY_MINIBATCH_SIZE:-0}"
STAGE3_GRPO_PPO_REPLAY_MAX_GRAD_NORM="${GRPO_PPO_REPLAY_MAX_GRAD_NORM:-1.0}"
STAGE3_GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE="${GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE:-1e-6}"
STAGE3_GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE="${GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE:-true}"
STAGE3_GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH="${GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH:-true}"
STAGE3_GRPO_PPO_REPLAY_BC_UPDATE="${GRPO_PPO_REPLAY_BC_UPDATE:-true}"
STAGE3_GRPO_PPO_REPLAY_LOGPROB_MODE="${GRPO_PPO_REPLAY_LOGPROB_MODE:-trajectory}"
STAGE3_GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE="${GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE:-trajectory_all_steps}"
STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN="${GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN:--5.0}"
STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX="${GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX:-2.0}"
STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE="${GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE:-constant}"
STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_BASE="${GRPO_PPO_REPLAY_STEP_CLIP_BASE:-0.001}"
STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_RATE="${GRPO_PPO_REPLAY_STEP_CLIP_RATE:-3.0}"
STAGE3_TRAINER_GRADIENT_CLIP_VAL="${TRAINER_GRADIENT_CLIP_VAL:-}"
STAGE3_GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${STAGE3_MAX_EPOCHS}}"
STAGE3_GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}"
STAGE3_GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}"
STAGE3_LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
STAGE3_LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
STAGE3_LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-}"
STAGE3_CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
STAGE3_CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-0}"
STAGE3_OFFLINE_RL_ENABLED="${OFFLINE_RL_ENABLED:-false}"
STAGE3_ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-}"
STAGE3_OFFLINE_RL_MISSING_BUFFER_POLICY="${OFFLINE_RL_MISSING_BUFFER_POLICY:-error}"
STAGE3_OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY="${OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY:-false}"
STAGE3_GRPO_BUFFER_GUIDANCE_ENABLED="${GRPO_BUFFER_GUIDANCE_ENABLED:-false}"
STAGE3_GRPO_BUFFER_REWARD_BONUS_WEIGHT="${GRPO_BUFFER_REWARD_BONUS_WEIGHT:-0.0}"
STAGE3_GRPO_BUFFER_REWARD_BONUS_SCALE_M="${GRPO_BUFFER_REWARD_BONUS_SCALE_M:-4.0}"
STAGE3_GRPO_BUFFER_REWARD_BONUS_USE_MARGIN="${GRPO_BUFFER_REWARD_BONUS_USE_MARGIN:-true}"
STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT:-0.0}"
STAGE3_GRPO_BUFFER_DISTILL_LOSS_SCHEDULE="${GRPO_BUFFER_DISTILL_LOSS_SCHEDULE:-linear_warmup}"
STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START:-0.0}"
STAGE3_GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH="${GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH:-0}"
STAGE3_GRPO_BUFFER_DISTILL_WARMUP_EPOCHS="${GRPO_BUFFER_DISTILL_WARMUP_EPOCHS:-3}"
STAGE3_GRPO_BUFFER_DISTILL_TOP_K="${GRPO_BUFFER_DISTILL_TOP_K:-1}"
STAGE3_GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN="${GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN:-0.0}"
STAGE3_GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING="${GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING:-low_noise}"
STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT="${GRPO_SELF_IMITATION_LOSS_WEIGHT:-0.0}"
STAGE3_GRPO_SELF_IMITATION_LOSS_SCHEDULE="${GRPO_SELF_IMITATION_LOSS_SCHEDULE:-linear_warmup}"
STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT_START="${GRPO_SELF_IMITATION_LOSS_WEIGHT_START:-0.0}"
STAGE3_GRPO_SELF_IMITATION_WARMUP_START_EPOCH="${GRPO_SELF_IMITATION_WARMUP_START_EPOCH:-0}"
STAGE3_GRPO_SELF_IMITATION_WARMUP_EPOCHS="${GRPO_SELF_IMITATION_WARMUP_EPOCHS:-3}"
STAGE3_GRPO_SELF_IMITATION_TOP_K="${GRPO_SELF_IMITATION_TOP_K:-1}"
STAGE3_GRPO_SELF_IMITATION_MIN_REWARD="${GRPO_SELF_IMITATION_MIN_REWARD:-0.85}"
STAGE3_GRPO_SELF_IMITATION_MIN_REWARD_MARGIN="${GRPO_SELF_IMITATION_MIN_REWARD_MARGIN:-0.01}"
STAGE3_GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO="${GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO:-1.0}"
STAGE3_GRPO_SELF_IMITATION_BATCH_CAP_SCORE="${GRPO_SELF_IMITATION_BATCH_CAP_SCORE:-reward}"
STAGE3_GRPO_SELF_IMITATION_BASELINE_MODE="${GRPO_SELF_IMITATION_BASELINE_MODE:-buffer_or_group_mean}"
STAGE3_GRPO_SELF_IMITATION_TIMESTEP_SAMPLING="${GRPO_SELF_IMITATION_TIMESTEP_SAMPLING:-low_noise}"
STAGE3_GRPO_SELF_IMITATION_REQUIRE_NC="${GRPO_SELF_IMITATION_REQUIRE_NC:-true}"
STAGE3_GRPO_SELF_IMITATION_REQUIRE_DAC="${GRPO_SELF_IMITATION_REQUIRE_DAC:-true}"
STAGE3_GRPO_SELF_IMITATION_REQUIRE_TTC="${GRPO_SELF_IMITATION_REQUIRE_TTC:-true}"
STAGE3_GRPO_SELF_IMITATION_REQUIRE_DDC="${GRPO_SELF_IMITATION_REQUIRE_DDC:-true}"
STAGE3_GRPO_SELF_IMITATION_NC_MIN_ABSOLUTE="${GRPO_SELF_IMITATION_NC_MIN_ABSOLUTE:-1.0}"
STAGE3_GRPO_SELF_IMITATION_DAC_MIN_ABSOLUTE="${GRPO_SELF_IMITATION_DAC_MIN_ABSOLUTE:-1.0}"
STAGE3_GRPO_SELF_IMITATION_TTC_MIN_ABSOLUTE="${GRPO_SELF_IMITATION_TTC_MIN_ABSOLUTE:-0.95}"
STAGE3_GRPO_SELF_IMITATION_DDC_MIN_ABSOLUTE="${GRPO_SELF_IMITATION_DDC_MIN_ABSOLUTE:-0.99}"
STAGE3_OFFLINE_RL_USE_BATCHED_PDM_SCORING="${OFFLINE_RL_USE_BATCHED_PDM_SCORING:-true}"
STAGE3_OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
STAGE3_OFFLINE_RL_USE_FAST_PDM_SCORER="${OFFLINE_RL_USE_FAST_PDM_SCORER:-true}"
STAGE3_OFFLINE_RL_PDM_BATCH_CHUNK_SIZE="${OFFLINE_RL_PDM_BATCH_CHUNK_SIZE:-0}"
STAGE3_OFFLINE_RL_PDM_SHADOW_CHECK="${OFFLINE_RL_PDM_SHADOW_CHECK:-false}"
STAGE3_OFFLINE_RL_PDM_SHADOW_MAX_SAMPLES="${OFFLINE_RL_PDM_SHADOW_MAX_SAMPLES:-4}"
STAGE3_OFFLINE_RL_PDM_SHADOW_MAX_ABS_DIFF="${OFFLINE_RL_PDM_SHADOW_MAX_ABS_DIFF:-0.0}"
STAGE3_OFFLINE_RL_STRICT_REWARD_SUBMETRICS="${OFFLINE_RL_STRICT_REWARD_SUBMETRICS:-true}"
STAGE3_OFFLINE_RL_MISSING_SUBMETRIC_POLICY="${OFFLINE_RL_MISSING_SUBMETRIC_POLICY:-error}"
STAGE3_TRAIN_TEST_SPLIT="navtrain"
STAGE3_EXPERIMENT_NAME="training_recogdrive_agent"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUT_ROOT}/hydra}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "${CACHE_MODE}" != "online" && "${CACHE_MODE}" != "offline" ]]; then
  echo "CACHE_MODE must be online or offline, got: ${CACHE_MODE}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -f "${IL_CHECKPOINT}" ]]; then
  echo "Stage 2 IL checkpoint does not exist: ${IL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "VLM path does not exist: ${VLM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${NUPLAN_MAPS_ROOT}" ]]; then
  echo "NUPLAN_MAPS_ROOT does not exist: ${NUPLAN_MAPS_ROOT}" >&2
  exit 2
fi
if [[ "${STAGE3_OFFLINE_RL_ENABLED}" == "true" || "${STAGE3_GRPO_BUFFER_GUIDANCE_ENABLED}" == "true" ]]; then
  if [[ -z "${STAGE3_ELITE_BUFFER_DIR}" || ! -d "${STAGE3_ELITE_BUFFER_DIR}" ]]; then
    echo "ELITE_BUFFER_DIR must exist when OFFLINE_RL_ENABLED/GRPO_BUFFER_GUIDANCE_ENABLED is true: ${STAGE3_ELITE_BUFFER_DIR}" >&2
    exit 2
  fi
fi

"${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader

cache_path = Path("${METRIC_CACHE_DIR}")
try:
    loader = MetricCacheLoader(cache_path)
except Exception as exc:
    raise SystemExit(f"Metric cache is not ready at {cache_path}: {exc}")
count = len(loader)
if count <= 0:
    raise SystemExit(f"Metric cache is empty at {cache_path}")
print(f"metric_cache_count={count}")
PY

"${PYTHON_BIN}" - <<PY
import sys
import torch
required = int("${GPUS_PER_NODE}")
count = torch.cuda.device_count()
print(f"cuda_device_count={count}")
if count < required:
    raise SystemExit(f"Need at least {required} visible CUDA devices, got {count}")
PY

mkdir -p "${OUT_ROOT}"

HYDRA_ARGS=(
  "agent=recogdrive_agent"
  "agent.lr=${STAGE3_LR}"
  "agent.stage3_objective=${STAGE3_OBJECTIVE}"
  "agent.vlm_path=${VLM_PATH}"
  "agent.cam_type=single"
  "agent.grpo=True"
  "agent.grpo_sample_time=${STAGE3_GRPO_SAMPLE_TIME}"
  "agent.bc_anneal=${STAGE3_BC_ANNEAL}"
  "agent.bc_coeff_start=${STAGE3_BC_COEFF_START}"
  "agent.bc_coeff_end=${STAGE3_BC_COEFF_END}"
  "agent.bc_anneal_epochs=${STAGE3_BC_ANNEAL_EPOCHS}"
  "agent.reference_kl_coeff=${STAGE3_REFERENCE_KL_COEFF}"
  "agent.reference_kl_chunk_size=${STAGE3_REFERENCE_KL_CHUNK_SIZE}"
  "agent.grpo_use_gspo_ratio=${STAGE3_GRPO_USE_GSPO_RATIO}"
  "agent.grpo_gspo_clip_low=${STAGE3_GRPO_GSPO_CLIP_LOW}"
  "agent.grpo_gspo_clip_high=${STAGE3_GRPO_GSPO_CLIP_HIGH}"
  "agent.grpo_behavior_policy_sync_interval=${STAGE3_GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL}"
  "agent.grpo_behavior_policy_sample=${STAGE3_GRPO_BEHAVIOR_POLICY_SAMPLE}"
  "agent.grpo_normalize_advantage_batch=${STAGE3_GRPO_NORMALIZE_ADVANTAGE_BATCH}"
  "agent.grpo_advantage_clip_abs=${STAGE3_GRPO_ADVANTAGE_CLIP_ABS}"
  "agent.grpo_hard_gate_ttc=${STAGE3_GRPO_HARD_GATE_TTC}"
  "agent.grpo_hard_gate_ddc=${STAGE3_GRPO_HARD_GATE_DDC}"
  "agent.grpo_ttc_safe_threshold=${STAGE3_GRPO_TTC_SAFE_THRESHOLD}"
  "agent.grpo_ddc_safe_threshold=${STAGE3_GRPO_DDC_SAFE_THRESHOLD}"
  "agent.grpo_ppo_replay_inner_epochs=${STAGE3_GRPO_PPO_REPLAY_INNER_EPOCHS}"
  "agent.grpo_ppo_replay_minibatch_size=${STAGE3_GRPO_PPO_REPLAY_MINIBATCH_SIZE}"
  "agent.grpo_ppo_replay_max_grad_norm=${STAGE3_GRPO_PPO_REPLAY_MAX_GRAD_NORM}"
  "agent.grpo_ppo_replay_min_abs_advantage=${STAGE3_GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE}"
  "agent.grpo_ppo_replay_filter_zero_advantage=${STAGE3_GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE}"
  "agent.grpo_ppo_replay_sync_behavior_each_batch=${STAGE3_GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH}"
  "agent.grpo_ppo_replay_bc_update=${STAGE3_GRPO_PPO_REPLAY_BC_UPDATE}"
  "agent.grpo_ppo_replay_logprob_mode=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_MODE}"
  "agent.grpo_ppo_replay_step_minibatch_mode=${STAGE3_GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE}"
  "agent.grpo_ppo_replay_logprob_clamp_min=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN}"
  "agent.grpo_ppo_replay_logprob_clamp_max=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX}"
  "agent.grpo_ppo_replay_step_clip_schedule=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE}"
  "agent.grpo_ppo_replay_step_clip_base=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_BASE}"
  "agent.grpo_ppo_replay_step_clip_rate=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_RATE}"
  "agent.grpo_scheduler_epochs=${STAGE3_GRPO_SCHEDULER_EPOCHS}"
  "agent.grpo_scheduler_warmup_epochs=${STAGE3_GRPO_SCHEDULER_WARMUP_EPOCHS}"
  "agent.grpo_scheduler_min_lr=${STAGE3_GRPO_SCHEDULER_MIN_LR}"
  "agent.offline_rl_enabled=${STAGE3_OFFLINE_RL_ENABLED}"
  "agent.offline_rl_elite_buffer_path=${STAGE3_ELITE_BUFFER_DIR}"
  "agent.offline_rl_missing_buffer_policy=${STAGE3_OFFLINE_RL_MISSING_BUFFER_POLICY}"
  "agent.offline_rl_cache_elite_records_in_memory=${STAGE3_OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY}"
  "agent.offline_rl_strict_reward_submetrics=${STAGE3_OFFLINE_RL_STRICT_REWARD_SUBMETRICS}"
  "agent.offline_rl_missing_submetric_policy=${STAGE3_OFFLINE_RL_MISSING_SUBMETRIC_POLICY}"
  "agent.offline_rl_use_batched_pdm_scoring=${STAGE3_OFFLINE_RL_USE_BATCHED_PDM_SCORING}"
  "agent.offline_rl_use_exact_array_pdm_state_conversion=${STAGE3_OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}"
  "agent.offline_rl_use_fast_pdm_scorer=${STAGE3_OFFLINE_RL_USE_FAST_PDM_SCORER}"
  "agent.offline_rl_pdm_batch_chunk_size=${STAGE3_OFFLINE_RL_PDM_BATCH_CHUNK_SIZE}"
  "agent.offline_rl_pdm_shadow_check=${STAGE3_OFFLINE_RL_PDM_SHADOW_CHECK}"
  "agent.offline_rl_pdm_shadow_max_samples=${STAGE3_OFFLINE_RL_PDM_SHADOW_MAX_SAMPLES}"
  "agent.offline_rl_pdm_shadow_max_abs_diff=${STAGE3_OFFLINE_RL_PDM_SHADOW_MAX_ABS_DIFF}"
  "agent.offline_rl_grpo_buffer_guidance_enabled=${STAGE3_GRPO_BUFFER_GUIDANCE_ENABLED}"
  "agent.offline_rl_grpo_buffer_reward_bonus_weight=${STAGE3_GRPO_BUFFER_REWARD_BONUS_WEIGHT}"
  "agent.offline_rl_grpo_buffer_reward_bonus_scale_m=${STAGE3_GRPO_BUFFER_REWARD_BONUS_SCALE_M}"
  "agent.offline_rl_grpo_buffer_reward_bonus_use_margin=${STAGE3_GRPO_BUFFER_REWARD_BONUS_USE_MARGIN}"
  "agent.offline_rl_grpo_buffer_distill_loss_weight=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT}"
  "agent.offline_rl_grpo_buffer_distill_loss_schedule=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_SCHEDULE}"
  "agent.offline_rl_grpo_buffer_distill_loss_weight_start=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START}"
  "agent.offline_rl_grpo_buffer_distill_warmup_start_epoch=${STAGE3_GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH}"
  "agent.offline_rl_grpo_buffer_distill_warmup_epochs=${STAGE3_GRPO_BUFFER_DISTILL_WARMUP_EPOCHS}"
  "agent.offline_rl_grpo_buffer_distill_top_k=${STAGE3_GRPO_BUFFER_DISTILL_TOP_K}"
  "agent.offline_rl_grpo_buffer_distill_min_reward_margin=${STAGE3_GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN}"
  "agent.offline_rl_grpo_buffer_distill_timestep_sampling=${STAGE3_GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING}"
  "agent.offline_rl_grpo_self_imitation_loss_weight=${STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT}"
  "agent.offline_rl_grpo_self_imitation_loss_schedule=${STAGE3_GRPO_SELF_IMITATION_LOSS_SCHEDULE}"
  "agent.offline_rl_grpo_self_imitation_loss_weight_start=${STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT_START}"
  "agent.offline_rl_grpo_self_imitation_warmup_start_epoch=${STAGE3_GRPO_SELF_IMITATION_WARMUP_START_EPOCH}"
  "agent.offline_rl_grpo_self_imitation_warmup_epochs=${STAGE3_GRPO_SELF_IMITATION_WARMUP_EPOCHS}"
  "agent.offline_rl_grpo_self_imitation_top_k=${STAGE3_GRPO_SELF_IMITATION_TOP_K}"
  "agent.offline_rl_grpo_self_imitation_min_reward=${STAGE3_GRPO_SELF_IMITATION_MIN_REWARD}"
  "agent.offline_rl_grpo_self_imitation_min_reward_margin=${STAGE3_GRPO_SELF_IMITATION_MIN_REWARD_MARGIN}"
  "agent.offline_rl_grpo_self_imitation_max_target_scene_ratio=${STAGE3_GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO}"
  "agent.offline_rl_grpo_self_imitation_batch_cap_score=${STAGE3_GRPO_SELF_IMITATION_BATCH_CAP_SCORE}"
  "agent.offline_rl_grpo_self_imitation_baseline_mode=${STAGE3_GRPO_SELF_IMITATION_BASELINE_MODE}"
  "agent.offline_rl_grpo_self_imitation_timestep_sampling=${STAGE3_GRPO_SELF_IMITATION_TIMESTEP_SAMPLING}"
  "agent.offline_rl_grpo_self_imitation_require_nc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_NC}"
  "agent.offline_rl_grpo_self_imitation_require_dac=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_DAC}"
  "agent.offline_rl_grpo_self_imitation_require_ttc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_TTC}"
  "agent.offline_rl_grpo_self_imitation_require_ddc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_DDC}"
  "agent.offline_rl_grpo_self_imitation_nc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_NC_MIN_ABSOLUTE}"
  "agent.offline_rl_grpo_self_imitation_dac_min_absolute=${STAGE3_GRPO_SELF_IMITATION_DAC_MIN_ABSOLUTE}"
  "agent.offline_rl_grpo_self_imitation_ttc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_TTC_MIN_ABSOLUTE}"
  "agent.offline_rl_grpo_self_imitation_ddc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_DDC_MIN_ABSOLUTE}"
  "agent.cache_mode=False"
  "agent.vlm_type=internvl"
  "agent.checkpoint_path=${IL_CHECKPOINT}"
  "agent.dit_type=small"
  "agent.vlm_size=small"
  "agent.sampling_method=ddim"
  "agent.metric_cache_path=${METRIC_CACHE_DIR}"
  "agent.reference_policy_checkpoint=${IL_CHECKPOINT}"
  "trainer.params.max_epochs=${STAGE3_MAX_EPOCHS}"
  "trainer.params.num_nodes=${NODES}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "trainer.params.strategy=${DDP_STRATEGY}"
  "trainer.params.accumulate_grad_batches=${STAGE3_ACCUMULATE_GRAD_BATCHES}"
  "checkpoint.every_n_epochs=${STAGE3_CHECKPOINT_EVERY_N_EPOCHS}"
  "checkpoint.every_n_train_steps=${STAGE3_CHECKPOINT_EVERY_N_TRAIN_STEPS}"
  "dataloader.params.batch_size=${STAGE3_BATCH_SIZE}"
  "experiment_name=${STAGE3_EXPERIMENT_NAME}"
  "train_test_split=${STAGE3_TRAIN_TEST_SPLIT}"
  "force_cache_computation=False"
)
if [[ -n "${STAGE3_TRAINER_GRADIENT_CLIP_VAL}" ]]; then
  HYDRA_ARGS+=("trainer.params.gradient_clip_val=${STAGE3_TRAINER_GRADIENT_CLIP_VAL}")
elif [[ "${STAGE3_OBJECTIVE}" == "grpo_replay" ]]; then
  HYDRA_ARGS+=("trainer.params.gradient_clip_val=null")
fi
if [[ -n "${STAGE3_LIMIT_TRAIN_BATCHES}" ]]; then
  HYDRA_ARGS+=("trainer.params.limit_train_batches=${STAGE3_LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${STAGE3_LIMIT_VAL_BATCHES}" ]]; then
  HYDRA_ARGS+=("trainer.params.limit_val_batches=${STAGE3_LIMIT_VAL_BATCHES}")
fi
if [[ -n "${STAGE3_LOG_EVERY_N_STEPS}" ]]; then
  HYDRA_ARGS+=("trainer.params.log_every_n_steps=${STAGE3_LOG_EVERY_N_STEPS}")
fi

if [[ "${CACHE_MODE}" == "offline" ]]; then
  if [[ ! -d "${HIDDEN_CACHE_DIR}" ]]; then
    echo "HIDDEN_CACHE_DIR does not exist for offline mode: ${HIDDEN_CACHE_DIR}" >&2
    exit 2
  fi
  if ! find "${HIDDEN_CACHE_DIR}" -name 'internvl_feature.gz' -print -quit | grep -q .; then
    echo "No internvl_feature.gz files found under HIDDEN_CACHE_DIR: ${HIDDEN_CACHE_DIR}" >&2
    exit 2
  fi
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=True"
    "cache_path=${HIDDEN_CACHE_DIR}"
    "use_cache_without_dataset=True"
  )
else
  if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
    echo "NAVSIM_LOG_PATH does not exist for online mode: ${NAVSIM_LOG_PATH}" >&2
    exit 2
  fi
  if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
    echo "SENSOR_BLOBS_PATH does not exist for online mode: ${SENSOR_BLOBS_PATH}" >&2
    exit 2
  fi
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=False"
    "cache_path=null"
    "use_cache_without_dataset=False"
    "navsim_log_path=${NAVSIM_LOG_PATH}"
    "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  )
fi

if [[ -n "${MAX_SCENES:-}" ]]; then
  HYDRA_ARGS+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

CMD=(
  "${TORCHRUN_BIN}"
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py"
  "${HYDRA_ARGS[@]}"
)

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "cache_mode=${CACHE_MODE}"
  echo "il_checkpoint=${IL_CHECKPOINT}"
  echo "vlm_path=${VLM_PATH}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "hidden_cache_dir=${HIDDEN_CACHE_DIR}"
  echo "navsim_log_path=${NAVSIM_LOG_PATH}"
  echo "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "ddp_strategy=${DDP_STRATEGY}"
  echo "stage3_lr=${STAGE3_LR}"
  echo "stage3_objective=${STAGE3_OBJECTIVE}"
  echo "stage3_max_epochs=${STAGE3_MAX_EPOCHS}"
  echo "stage3_batch_size=${STAGE3_BATCH_SIZE}"
  echo "stage3_accumulate_grad_batches=${STAGE3_ACCUMULATE_GRAD_BATCHES}"
  echo "stage3_grpo_sample_time=${STAGE3_GRPO_SAMPLE_TIME}"
  echo "stage3_bc_anneal=${STAGE3_BC_ANNEAL}"
  echo "stage3_bc_coeff_start=${STAGE3_BC_COEFF_START}"
  echo "stage3_bc_coeff_end=${STAGE3_BC_COEFF_END}"
  echo "stage3_bc_anneal_epochs=${STAGE3_BC_ANNEAL_EPOCHS}"
  echo "stage3_reference_kl_coeff=${STAGE3_REFERENCE_KL_COEFF}"
  echo "stage3_reference_kl_chunk_size=${STAGE3_REFERENCE_KL_CHUNK_SIZE}"
  echo "stage3_grpo_use_gspo_ratio=${STAGE3_GRPO_USE_GSPO_RATIO}"
  echo "stage3_grpo_gspo_clip_low=${STAGE3_GRPO_GSPO_CLIP_LOW}"
  echo "stage3_grpo_gspo_clip_high=${STAGE3_GRPO_GSPO_CLIP_HIGH}"
  echo "stage3_grpo_behavior_policy_sync_interval=${STAGE3_GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL}"
  echo "stage3_grpo_behavior_policy_sample=${STAGE3_GRPO_BEHAVIOR_POLICY_SAMPLE}"
  echo "stage3_grpo_normalize_advantage_batch=${STAGE3_GRPO_NORMALIZE_ADVANTAGE_BATCH}"
  echo "stage3_grpo_advantage_clip_abs=${STAGE3_GRPO_ADVANTAGE_CLIP_ABS}"
  echo "stage3_grpo_hard_gate_ttc=${STAGE3_GRPO_HARD_GATE_TTC}"
  echo "stage3_grpo_hard_gate_ddc=${STAGE3_GRPO_HARD_GATE_DDC}"
  echo "stage3_grpo_ttc_safe_threshold=${STAGE3_GRPO_TTC_SAFE_THRESHOLD}"
  echo "stage3_grpo_ddc_safe_threshold=${STAGE3_GRPO_DDC_SAFE_THRESHOLD}"
  echo "stage3_grpo_ppo_replay_inner_epochs=${STAGE3_GRPO_PPO_REPLAY_INNER_EPOCHS}"
  echo "stage3_grpo_ppo_replay_minibatch_size=${STAGE3_GRPO_PPO_REPLAY_MINIBATCH_SIZE}"
  echo "stage3_grpo_ppo_replay_max_grad_norm=${STAGE3_GRPO_PPO_REPLAY_MAX_GRAD_NORM}"
  echo "stage3_grpo_ppo_replay_min_abs_advantage=${STAGE3_GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE}"
  echo "stage3_grpo_ppo_replay_filter_zero_advantage=${STAGE3_GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE}"
  echo "stage3_grpo_ppo_replay_sync_behavior_each_batch=${STAGE3_GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH}"
  echo "stage3_grpo_ppo_replay_bc_update=${STAGE3_GRPO_PPO_REPLAY_BC_UPDATE}"
  echo "stage3_grpo_ppo_replay_logprob_mode=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_MODE}"
  echo "stage3_grpo_ppo_replay_step_minibatch_mode=${STAGE3_GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE}"
  echo "stage3_grpo_ppo_replay_logprob_clamp_min=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN}"
  echo "stage3_grpo_ppo_replay_logprob_clamp_max=${STAGE3_GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX}"
  echo "stage3_grpo_ppo_replay_step_clip_schedule=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE}"
  echo "stage3_grpo_ppo_replay_step_clip_base=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_BASE}"
  echo "stage3_grpo_ppo_replay_step_clip_rate=${STAGE3_GRPO_PPO_REPLAY_STEP_CLIP_RATE}"
  if [[ -n "${STAGE3_TRAINER_GRADIENT_CLIP_VAL}" ]]; then
    echo "stage3_trainer_gradient_clip_val=${STAGE3_TRAINER_GRADIENT_CLIP_VAL}"
  elif [[ "${STAGE3_OBJECTIVE}" == "grpo_replay" ]]; then
    echo "stage3_trainer_gradient_clip_val=null"
  else
    echo "stage3_trainer_gradient_clip_val=config_default"
  fi
  echo "stage3_grpo_scheduler_epochs=${STAGE3_GRPO_SCHEDULER_EPOCHS}"
  echo "stage3_grpo_scheduler_warmup_epochs=${STAGE3_GRPO_SCHEDULER_WARMUP_EPOCHS}"
  echo "stage3_grpo_scheduler_min_lr=${STAGE3_GRPO_SCHEDULER_MIN_LR}"
  echo "stage3_limit_train_batches=${STAGE3_LIMIT_TRAIN_BATCHES}"
  echo "stage3_limit_val_batches=${STAGE3_LIMIT_VAL_BATCHES}"
  echo "stage3_log_every_n_steps=${STAGE3_LOG_EVERY_N_STEPS}"
  echo "stage3_checkpoint_every_n_epochs=${STAGE3_CHECKPOINT_EVERY_N_EPOCHS}"
  echo "stage3_checkpoint_every_n_train_steps=${STAGE3_CHECKPOINT_EVERY_N_TRAIN_STEPS}"
  echo "stage3_offline_rl_enabled=${STAGE3_OFFLINE_RL_ENABLED}"
  echo "stage3_elite_buffer_dir=${STAGE3_ELITE_BUFFER_DIR}"
  echo "stage3_offline_rl_cache_elite_records_in_memory=${STAGE3_OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY}"
  echo "stage3_grpo_buffer_guidance_enabled=${STAGE3_GRPO_BUFFER_GUIDANCE_ENABLED}"
  echo "stage3_grpo_buffer_reward_bonus_weight=${STAGE3_GRPO_BUFFER_REWARD_BONUS_WEIGHT}"
  echo "stage3_grpo_buffer_reward_bonus_scale_m=${STAGE3_GRPO_BUFFER_REWARD_BONUS_SCALE_M}"
  echo "stage3_grpo_buffer_reward_bonus_use_margin=${STAGE3_GRPO_BUFFER_REWARD_BONUS_USE_MARGIN}"
  echo "stage3_grpo_buffer_distill_loss_weight=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT}"
  echo "stage3_grpo_buffer_distill_loss_schedule=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_SCHEDULE}"
  echo "stage3_grpo_buffer_distill_loss_weight_start=${STAGE3_GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START}"
  echo "stage3_grpo_buffer_distill_warmup_start_epoch=${STAGE3_GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH}"
  echo "stage3_grpo_buffer_distill_warmup_epochs=${STAGE3_GRPO_BUFFER_DISTILL_WARMUP_EPOCHS}"
  echo "stage3_grpo_buffer_distill_top_k=${STAGE3_GRPO_BUFFER_DISTILL_TOP_K}"
  echo "stage3_grpo_buffer_distill_min_reward_margin=${STAGE3_GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN}"
  echo "stage3_grpo_buffer_distill_timestep_sampling=${STAGE3_GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING}"
  echo "stage3_grpo_self_imitation_loss_weight=${STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT}"
  echo "stage3_grpo_self_imitation_loss_schedule=${STAGE3_GRPO_SELF_IMITATION_LOSS_SCHEDULE}"
  echo "stage3_grpo_self_imitation_loss_weight_start=${STAGE3_GRPO_SELF_IMITATION_LOSS_WEIGHT_START}"
  echo "stage3_grpo_self_imitation_warmup_start_epoch=${STAGE3_GRPO_SELF_IMITATION_WARMUP_START_EPOCH}"
  echo "stage3_grpo_self_imitation_warmup_epochs=${STAGE3_GRPO_SELF_IMITATION_WARMUP_EPOCHS}"
  echo "stage3_grpo_self_imitation_top_k=${STAGE3_GRPO_SELF_IMITATION_TOP_K}"
  echo "stage3_grpo_self_imitation_min_reward=${STAGE3_GRPO_SELF_IMITATION_MIN_REWARD}"
  echo "stage3_grpo_self_imitation_min_reward_margin=${STAGE3_GRPO_SELF_IMITATION_MIN_REWARD_MARGIN}"
  echo "stage3_grpo_self_imitation_max_target_scene_ratio=${STAGE3_GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO}"
  echo "stage3_grpo_self_imitation_batch_cap_score=${STAGE3_GRPO_SELF_IMITATION_BATCH_CAP_SCORE}"
  echo "stage3_grpo_self_imitation_baseline_mode=${STAGE3_GRPO_SELF_IMITATION_BASELINE_MODE}"
  echo "stage3_grpo_self_imitation_timestep_sampling=${STAGE3_GRPO_SELF_IMITATION_TIMESTEP_SAMPLING}"
  echo "stage3_grpo_self_imitation_require_nc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_NC}"
  echo "stage3_grpo_self_imitation_require_dac=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_DAC}"
  echo "stage3_grpo_self_imitation_require_ttc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_TTC}"
  echo "stage3_grpo_self_imitation_require_ddc=${STAGE3_GRPO_SELF_IMITATION_REQUIRE_DDC}"
  echo "stage3_grpo_self_imitation_nc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_NC_MIN_ABSOLUTE}"
  echo "stage3_grpo_self_imitation_dac_min_absolute=${STAGE3_GRPO_SELF_IMITATION_DAC_MIN_ABSOLUTE}"
  echo "stage3_grpo_self_imitation_ttc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_TTC_MIN_ABSOLUTE}"
  echo "stage3_grpo_self_imitation_ddc_min_absolute=${STAGE3_GRPO_SELF_IMITATION_DDC_MIN_ABSOLUTE}"
  echo "max_scenes=${MAX_SCENES:-full}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${OUT_ROOT}/resolved_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_command.txt"
  exit 0
fi

if [[ "${KILL_GPU_STRESS}" == "1" ]]; then
  STRESS_PIDS="$(pgrep -f '^python /mnt/project/gpu_stress.py$' || true)"
  if [[ -n "${STRESS_PIDS}" ]]; then
    echo "Killing gpu stress parent pids: ${STRESS_PIDS}"
    kill ${STRESS_PIDS}
    sleep 5
  fi
  ORPHAN_STRESS_PIDS="$(
    "${PYTHON_BIN}" - <<'PY' || true
import os
import subprocess

try:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    raise SystemExit(0)

pids = []
for line in query.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        continue
    try:
        pid = int(parts[0])
        used_mib = int(parts[1])
    except ValueError:
        continue
    if used_mib < 50000:
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if "multiprocessing.spawn" in cmdline and "--multiprocessing-fork" in cmdline:
        pids.append(str(pid))

if pids:
    print(" ".join(sorted(set(pids))))
PY
  )"
  if [[ -n "${ORPHAN_STRESS_PIDS}" ]]; then
    echo "Killing orphan gpu stress worker pids: ${ORPHAN_STRESS_PIDS}"
    kill ${ORPHAN_STRESS_PIDS}
    sleep 5
  fi
fi

exec "${CMD[@]}"
