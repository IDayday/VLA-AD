#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_grpo_dppo_step_s16_lr1e4_b2_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

# DPPO-style Stage3 policy optimization:
# collect a fixed diffusion rollout, score with train PDM cache, then do clipped
# old-policy-ratio updates on denoising steps/transitions.
export CACHE_MODE="${CACHE_MODE:-online}"
export KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
export GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export DDP_STRATEGY="${DDP_STRATEGY:-ddp}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

export LR="${LR:-1e-4}"
export STAGE3_OBJECTIVE="${STAGE3_OBJECTIVE:-grpo_replay}"
export MAX_EPOCHS="${MAX_EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
# Manual replay optimization performs explicit minibatch optimizer steps; keep
# Lightning accumulation at 1 to make the effective update schedule auditable.
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
export GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-16}"
export BC_ANNEAL="${BC_ANNEAL:-true}"
export BC_COEFF_START="${BC_COEFF_START:-0.10}"
export BC_COEFF_END="${BC_COEFF_END:-0.05}"
export BC_ANNEAL_EPOCHS="${BC_ANNEAL_EPOCHS:-5}"
export REFERENCE_KL_COEFF="${REFERENCE_KL_COEFF:-0.02}"
# Step-level replay evaluates old/reference transition distributions and is
# much heavier than trajectory-level GRPO. Keep reference-KL chunking enabled by
# default; the old unchunked b2 setting OOMed on 80GB A800.
export REFERENCE_KL_CHUNK_SIZE="${REFERENCE_KL_CHUNK_SIZE:-4}"

# Use behavior-policy rollouts and old/new log-prob ratios. This is the part
# that makes the update a clipped policy-gradient replay pass rather than plain
# reward-weighted regression.
export GRPO_USE_GSPO_RATIO="${GRPO_USE_GSPO_RATIO:-true}"
export GRPO_GSPO_CLIP_LOW="${GRPO_GSPO_CLIP_LOW:-0.05}"
export GRPO_GSPO_CLIP_HIGH="${GRPO_GSPO_CLIP_HIGH:-0.05}"
export GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL="${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-1}"
export GRPO_BEHAVIOR_POLICY_SAMPLE="${GRPO_BEHAVIOR_POLICY_SAMPLE:-true}"
export GRPO_NORMALIZE_ADVANTAGE_BATCH="${GRPO_NORMALIZE_ADVANTAGE_BATCH:-true}"
export GRPO_ADVANTAGE_CLIP_ABS="${GRPO_ADVANTAGE_CLIP_ABS:-5.0}"

# Keep the safety fix from the RPP-constrained run. 91.04 gains EP/TTC but loses
# DDC; do not give positive policy-gradient weight to TTC/DDC-unsafe samples.
export GRPO_HARD_GATE_TTC="${GRPO_HARD_GATE_TTC:-true}"
export GRPO_HARD_GATE_DDC="${GRPO_HARD_GATE_DDC:-true}"
export GRPO_TTC_SAFE_THRESHOLD="${GRPO_TTC_SAFE_THRESHOLD:-0.95}"
export GRPO_DDC_SAFE_THRESHOLD="${GRPO_DDC_SAFE_THRESHOLD:-0.99}"

# DPPO-style replay knobs. Transition minibatching avoids treating the whole
# denoising chain as one opaque scalar and lets early/late denoising steps use
# different clip widths.
export GRPO_PPO_REPLAY_INNER_EPOCHS="${GRPO_PPO_REPLAY_INNER_EPOCHS:-1}"
export GRPO_PPO_REPLAY_MINIBATCH_SIZE="${GRPO_PPO_REPLAY_MINIBATCH_SIZE:-64}"
export GRPO_PPO_REPLAY_MAX_GRAD_NORM="${GRPO_PPO_REPLAY_MAX_GRAD_NORM:-1.0}"
export GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE="${GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE:-1e-6}"
export GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE="${GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE:-true}"
export GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH="${GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH:-true}"
export GRPO_PPO_REPLAY_BC_UPDATE="${GRPO_PPO_REPLAY_BC_UPDATE:-true}"
export GRPO_PPO_REPLAY_LOGPROB_MODE="${GRPO_PPO_REPLAY_LOGPROB_MODE:-step}"
export GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE="${GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE:-transition}"
export GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN="${GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN:--5.0}"
export GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX="${GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX:-2.0}"
export GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE="${GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE:-dppo_exp}"
export GRPO_PPO_REPLAY_STEP_CLIP_BASE="${GRPO_PPO_REPLAY_STEP_CLIP_BASE:-0.001}"
export GRPO_PPO_REPLAY_STEP_CLIP_RATE="${GRPO_PPO_REPLAY_STEP_CLIP_RATE:-3.0}"

export GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${MAX_EPOCHS}}"
export GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}"
export GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
# Memory-safe b1/acc1/8GPU runs have effective batch 8. A 2400-step checkpoint
# is roughly exposure-matched to a 300-step checkpoint from b2/acc4/8GPU runs.
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-2400}"

# Keep buffer absorption disabled in this DPPO control. Buffer-DPO/distillation
# should be layered only after the policy-gradient baseline is understood.
export OFFLINE_RL_ENABLED="${OFFLINE_RL_ENABLED:-false}"
export GRPO_BUFFER_GUIDANCE_ENABLED="${GRPO_BUFFER_GUIDANCE_ENABLED:-false}"
export GRPO_BUFFER_REWARD_BONUS_WEIGHT="${GRPO_BUFFER_REWARD_BONUS_WEIGHT:-0.0}"
export GRPO_BUFFER_DISTILL_LOSS_WEIGHT="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT:-0.0}"
export GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT="${GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT:-0.0}"
export GRPO_SELF_IMITATION_LOSS_WEIGHT="${GRPO_SELF_IMITATION_LOSS_WEIGHT:-0.0}"

# DPPO replay is heavier than trajectory-level GRPO because each rollout can
# trigger multiple transition-minibatch optimizer steps. Default to waiting for
# genuinely free training GPUs so this launcher cannot accidentally stack on top
# of an active long-horizon run.
export TRAIN_WAIT_FOR_FREE_GPUS="${TRAIN_WAIT_FOR_FREE_GPUS:-1}"
export TRAIN_GPU_MAX_MEM_USED_MB="${TRAIN_GPU_MAX_MEM_USED_MB:-2000}"
export TRAIN_GPU_MAX_UTIL="${TRAIN_GPU_MAX_UTIL:-5}"
export START_EARLY_GATE_WATCHER="${START_EARLY_GATE_WATCHER:-1}"
export EARLY_GATE_THRESHOLD="${EARLY_GATE_THRESHOLD:-0.88}"
export EARLY_GATE_MARGIN="${EARLY_GATE_MARGIN:-0.005}"
export EARLY_GATE_STOP_ON_FAIL="${EARLY_GATE_STOP_ON_FAIL:-0}"
export EARLY_GATE_MIN_STOP_STEP="${EARLY_GATE_MIN_STOP_STEP:-3000}"

export RUN_TRAIN="${RUN_TRAIN:-1}"
export LAUNCH_EVAL_WATCHERS="${LAUNCH_EVAL_WATCHERS:-1}"
export PRIMARY_EVAL_HOST="${PRIMARY_EVAL_HOST:-training-vla-zt2}"
export SECONDARY_EVAL_HOST="${SECONDARY_EVAL_HOST:-}"
export EVAL_WAIT_FOR_FREE_GPUS="${EVAL_WAIT_FOR_FREE_GPUS:-1}"
export EVAL_GPU_MAX_MEM_USED_MB="${EVAL_GPU_MAX_MEM_USED_MB:-2000}"
export EVAL_GPU_MAX_UTIL="${EVAL_GPU_MAX_UTIL:-5}"
# With the memory-safe default, effective batch is 8, not the usual 64.
# Evaluate step checkpoints at roughly matched scene exposure: step2400 here is
# comparable to step300 for b2/acc4/8GPU controls. Epoch checkpoints are still
# evaluated regardless of this step filter.
export EVAL_MIN_CHECKPOINT_STEP="${EVAL_MIN_CHECKPOINT_STEP:-2400}"
export EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-2400}"
export EVAL_ALWAYS_EPOCH_CHECKPOINTS="${EVAL_ALWAYS_EPOCH_CHECKPOINTS:-1}"

exec "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh"
