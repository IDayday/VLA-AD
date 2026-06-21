#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"

# Evidence basis:
# - Pure AWAC/IQL regression and the lightweight Buffer-DPO run did not transfer
#   the high-PDMS train-only buffer into better navtest policy quality.
# - DDPO/DPPO-style references frame diffusion fine-tuning as policy-gradient
#   optimization over sampled denoising trajectories, not reward-weighted
#   static regression alone.
# - This launcher therefore keeps the clean GRPO/reference-KL path as the main
#   optimizer and uses the train-only elite buffer only as a small reward
#   neighborhood prior. It intentionally disables buffer distillation, buffer
#   DPO, and self-imitation so the result isolates this mechanism.

export RUN_NAME="${RUN_NAME:-stage3_grpo_buffer_bonusonly_s16_lr1e4_b2acc4_20e_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

export RUN_TRAIN="${RUN_TRAIN:-0}"
export LAUNCH_EVAL_WATCHERS="${LAUNCH_EVAL_WATCHERS:-1}"
export CACHE_MODE="${CACHE_MODE:-online}"
export KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
export LR="${LR:-1e-4}"
export MAX_EPOCHS="${MAX_EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
export GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-16}"
export BC_ANNEAL="${BC_ANNEAL:-true}"
export BC_COEFF_START="${BC_COEFF_START:-0.10}"
export BC_COEFF_END="${BC_COEFF_END:-0.05}"
export BC_ANNEAL_EPOCHS="${BC_ANNEAL_EPOCHS:-5}"
export REFERENCE_KL_COEFF="${REFERENCE_KL_COEFF:-0.02}"
export REFERENCE_KL_CHUNK_SIZE="${REFERENCE_KL_CHUNK_SIZE:-16}"

# Match the strongest clean-GRPO comparison spine first. Enable GSPO explicitly
# from the environment only if a matched control also uses it.
export GRPO_USE_GSPO_RATIO="${GRPO_USE_GSPO_RATIO:-false}"
export GRPO_GSPO_CLIP_LOW="${GRPO_GSPO_CLIP_LOW:-0.05}"
export GRPO_GSPO_CLIP_HIGH="${GRPO_GSPO_CLIP_HIGH:-0.05}"
export GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL="${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-4}"
export GRPO_BEHAVIOR_POLICY_SAMPLE="${GRPO_BEHAVIOR_POLICY_SAMPLE:-true}"
export GRPO_NORMALIZE_ADVANTAGE_BATCH="${GRPO_NORMALIZE_ADVANTAGE_BATCH:-false}"
export GRPO_ADVANTAGE_CLIP_ABS="${GRPO_ADVANTAGE_CLIP_ABS:-0.0}"
export GRPO_HARD_GATE_TTC="${GRPO_HARD_GATE_TTC:-false}"
export GRPO_HARD_GATE_DDC="${GRPO_HARD_GATE_DDC:-false}"
export GRPO_SAFETY_ADVANTAGE_MODE="${GRPO_SAFETY_ADVANTAGE_MODE:-hard}"

export GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${MAX_EPOCHS}}"
export GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}"
export GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}"

export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-300}"
export EVAL_MIN_CHECKPOINT_STEP="${EVAL_MIN_CHECKPOINT_STEP:-300}"
export EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-300}"
export EVAL_ALWAYS_EPOCH_CHECKPOINTS="${EVAL_ALWAYS_EPOCH_CHECKPOINTS:-1}"

export START_EARLY_GATE_WATCHER="${START_EARLY_GATE_WATCHER:-1}"
export EARLY_GATE_THRESHOLD="${EARLY_GATE_THRESHOLD:-0.88}"
export EARLY_GATE_MARGIN="${EARLY_GATE_MARGIN:-0.005}"
export EARLY_GATE_STOP_ON_FAIL="${EARLY_GATE_STOP_ON_FAIL:-0}"
export EARLY_GATE_MIN_STOP_STEP="${EARLY_GATE_MIN_STOP_STEP:-3000}"

export OFFLINE_RL_ENABLED="${OFFLINE_RL_ENABLED:-true}"
export ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z}"
export OFFLINE_RL_MISSING_BUFFER_POLICY="${OFFLINE_RL_MISSING_BUFFER_POLICY:-error}"
export OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY="${OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY:-true}"
export OFFLINE_RL_STRICT_REWARD_SUBMETRICS="${OFFLINE_RL_STRICT_REWARD_SUBMETRICS:-true}"
export OFFLINE_RL_MISSING_SUBMETRIC_POLICY="${OFFLINE_RL_MISSING_SUBMETRIC_POLICY:-error}"
export OFFLINE_RL_USE_BATCHED_PDM_SCORING="${OFFLINE_RL_USE_BATCHED_PDM_SCORING:-true}"
export OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
export OFFLINE_RL_USE_FAST_PDM_SCORER="${OFFLINE_RL_USE_FAST_PDM_SCORER:-true}"
export OFFLINE_RL_PDM_SHADOW_CHECK="${OFFLINE_RL_PDM_SHADOW_CHECK:-false}"

export GRPO_BUFFER_GUIDANCE_ENABLED="${GRPO_BUFFER_GUIDANCE_ENABLED:-true}"
export GRPO_BUFFER_REWARD_BONUS_WEIGHT="${GRPO_BUFFER_REWARD_BONUS_WEIGHT:-0.01}"
export GRPO_BUFFER_REWARD_BONUS_SCALE_M="${GRPO_BUFFER_REWARD_BONUS_SCALE_M:-3.0}"
export GRPO_BUFFER_REWARD_BONUS_USE_MARGIN="${GRPO_BUFFER_REWARD_BONUS_USE_MARGIN:-true}"

# This margin is reused by _load_grpo_buffer_guidance_batch to decide which
# valid buffer candidates are eligible targets, even when distillation is off.
export GRPO_BUFFER_DISTILL_TOP_K="${GRPO_BUFFER_DISTILL_TOP_K:-1}"
export GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN="${GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN:-0.02}"
export GRPO_BUFFER_DISTILL_LOSS_WEIGHT="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT:-0.0}"
export GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT="${GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT:-0.0}"
export GRPO_SELF_IMITATION_LOSS_WEIGHT="${GRPO_SELF_IMITATION_LOSS_WEIGHT:-0.0}"

export TRAIN_WAIT_FOR_FREE_GPUS="${TRAIN_WAIT_FOR_FREE_GPUS:-1}"
export TRAIN_GPU_MAX_MEM_USED_MB="${TRAIN_GPU_MAX_MEM_USED_MB:-2000}"
export TRAIN_GPU_MAX_UTIL="${TRAIN_GPU_MAX_UTIL:-5}"
export TRAIN_GPU_WAIT_POLL_SECONDS="${TRAIN_GPU_WAIT_POLL_SECONDS:-120}"

export PRIMARY_EVAL_HOST="${PRIMARY_EVAL_HOST:-training-vla-zt2}"
export PRIMARY_EVAL_GPU_LIST="${PRIMARY_EVAL_GPU_LIST:-4,5,6,7}"
export SECONDARY_EVAL_HOST="${SECONDARY_EVAL_HOST:-}"
export EVAL_WAIT_FOR_FREE_GPUS="${EVAL_WAIT_FOR_FREE_GPUS:-1}"
export EVAL_GPU_MAX_MEM_USED_MB="${EVAL_GPU_MAX_MEM_USED_MB:-2000}"
export EVAL_GPU_MAX_UTIL="${EVAL_GPU_MAX_UTIL:-5}"

if [[ ! -d "${ELITE_BUFFER_DIR}" ]]; then
  echo "Elite buffer does not exist: ${ELITE_BUFFER_DIR}" >&2
  exit 2
fi

if [[ "${RUN_TRAIN}" != "1" ]]; then
  mkdir -p "${OUT_ROOT}"
  bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh" \
    > "${OUT_ROOT}/underlying_stable_dryrun.log"
  echo "Strict GSPO launch config written to ${OUT_ROOT}/strict_gspo_launch_config.txt"
  echo "Underlying stable dry-run log written to ${OUT_ROOT}/underlying_stable_dryrun.log"
  echo "Buffer-bonus-only wrapper dry run. Start with:"
  printf '  RUN_TRAIN=1 LAUNCH_EVAL_WATCHERS=1 RUN_NAME=%q bash %q\n' \
    "${RUN_NAME}" "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_buffer_bonus_only_2b_local.sh"
  exit 0
fi

exec bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh"
