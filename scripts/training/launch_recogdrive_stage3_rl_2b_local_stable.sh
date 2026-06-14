#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_rl_2b_safe_diffgrpo_online_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
STABLE_LAUNCHER="${STABLE_LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CACHE_MODE="${CACHE_MODE:-online}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-1}"
START_EARLY_GATE_WATCHER="${START_EARLY_GATE_WATCHER:-0}"
EARLY_GATE_THRESHOLD="${EARLY_GATE_THRESHOLD:-0.88}"
EARLY_GATE_MARGIN="${EARLY_GATE_MARGIN:-0.005}"
EARLY_GATE_POLL_SECONDS="${EARLY_GATE_POLL_SECONDS:-300}"
EARLY_GATE_STOP_ON_FAIL="${EARLY_GATE_STOP_ON_FAIL:-0}"

mkdir -p "${OUT_ROOT}"

JOB_CMD="cd $(printf '%q' "${REPO_ROOT}") && "
JOB_CMD+="OUT_ROOT=$(printf '%q' "${OUT_ROOT}/train") "
JOB_CMD+="CACHE_MODE=$(printf '%q' "${CACHE_MODE}") "
JOB_CMD+="KILL_GPU_STRESS=$(printf '%q' "${KILL_GPU_STRESS}") "
for name in GPUS GPUS_PER_NODE NODES NODE_RANK MASTER_ADDR MASTER_PORT DDP_STRATEGY DRY_RUN; do
  if [[ -n "${!name:-}" ]]; then
    JOB_CMD+="${name}=$(printf '%q' "${!name}") "
  fi
done
if [[ -n "${METRIC_CACHE_DIR:-}" ]]; then
  JOB_CMD+="METRIC_CACHE_DIR=$(printf '%q' "${METRIC_CACHE_DIR}") "
fi
if [[ -n "${HIDDEN_CACHE_DIR:-}" ]]; then
  JOB_CMD+="HIDDEN_CACHE_DIR=$(printf '%q' "${HIDDEN_CACHE_DIR}") "
fi
if [[ -n "${IL_CHECKPOINT:-}" ]]; then
  JOB_CMD+="IL_CHECKPOINT=$(printf '%q' "${IL_CHECKPOINT}") "
fi
if [[ -n "${VLM_PATH:-}" ]]; then
  JOB_CMD+="VLM_PATH=$(printf '%q' "${VLM_PATH}") "
fi
if [[ -n "${MAX_SCENES:-}" ]]; then
  JOB_CMD+="MAX_SCENES=$(printf '%q' "${MAX_SCENES}") "
fi
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  JOB_CMD+="MAX_EPOCHS=$(printf '%q' "${MAX_EPOCHS}") "
fi
if [[ -n "${LR:-}" ]]; then
  JOB_CMD+="LR=$(printf '%q' "${LR}") "
fi
if [[ -n "${STAGE3_OBJECTIVE:-}" ]]; then
  JOB_CMD+="STAGE3_OBJECTIVE=$(printf '%q' "${STAGE3_OBJECTIVE}") "
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  JOB_CMD+="BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") "
fi
if [[ -n "${ACCUMULATE_GRAD_BATCHES:-}" ]]; then
  JOB_CMD+="ACCUMULATE_GRAD_BATCHES=$(printf '%q' "${ACCUMULATE_GRAD_BATCHES}") "
fi
if [[ -n "${GRPO_SAMPLE_TIME:-}" ]]; then
  JOB_CMD+="GRPO_SAMPLE_TIME=$(printf '%q' "${GRPO_SAMPLE_TIME}") "
fi
if [[ -n "${BC_ANNEAL:-}" ]]; then
  JOB_CMD+="BC_ANNEAL=$(printf '%q' "${BC_ANNEAL}") "
fi
if [[ -n "${BC_COEFF_START:-}" ]]; then
  JOB_CMD+="BC_COEFF_START=$(printf '%q' "${BC_COEFF_START}") "
fi
if [[ -n "${BC_COEFF_END:-}" ]]; then
  JOB_CMD+="BC_COEFF_END=$(printf '%q' "${BC_COEFF_END}") "
fi
if [[ -n "${BC_ANNEAL_EPOCHS:-}" ]]; then
  JOB_CMD+="BC_ANNEAL_EPOCHS=$(printf '%q' "${BC_ANNEAL_EPOCHS}") "
fi
if [[ -n "${REFERENCE_KL_COEFF:-}" ]]; then
  JOB_CMD+="REFERENCE_KL_COEFF=$(printf '%q' "${REFERENCE_KL_COEFF}") "
fi
if [[ -n "${REFERENCE_KL_CHUNK_SIZE:-}" ]]; then
  JOB_CMD+="REFERENCE_KL_CHUNK_SIZE=$(printf '%q' "${REFERENCE_KL_CHUNK_SIZE}") "
fi
if [[ -n "${GRPO_USE_GSPO_RATIO:-}" ]]; then
  JOB_CMD+="GRPO_USE_GSPO_RATIO=$(printf '%q' "${GRPO_USE_GSPO_RATIO}") "
fi
if [[ -n "${GRPO_GSPO_CLIP_LOW:-}" ]]; then
  JOB_CMD+="GRPO_GSPO_CLIP_LOW=$(printf '%q' "${GRPO_GSPO_CLIP_LOW}") "
fi
if [[ -n "${GRPO_GSPO_CLIP_HIGH:-}" ]]; then
  JOB_CMD+="GRPO_GSPO_CLIP_HIGH=$(printf '%q' "${GRPO_GSPO_CLIP_HIGH}") "
fi
if [[ -n "${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-}" ]]; then
  JOB_CMD+="GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL=$(printf '%q' "${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL}") "
fi
if [[ -n "${GRPO_BEHAVIOR_POLICY_SAMPLE:-}" ]]; then
  JOB_CMD+="GRPO_BEHAVIOR_POLICY_SAMPLE=$(printf '%q' "${GRPO_BEHAVIOR_POLICY_SAMPLE}") "
fi
if [[ -n "${GRPO_NORMALIZE_ADVANTAGE_BATCH:-}" ]]; then
  JOB_CMD+="GRPO_NORMALIZE_ADVANTAGE_BATCH=$(printf '%q' "${GRPO_NORMALIZE_ADVANTAGE_BATCH}") "
fi
if [[ -n "${GRPO_ADVANTAGE_CLIP_ABS:-}" ]]; then
  JOB_CMD+="GRPO_ADVANTAGE_CLIP_ABS=$(printf '%q' "${GRPO_ADVANTAGE_CLIP_ABS}") "
fi
for name in GRPO_HARD_GATE_TTC GRPO_HARD_GATE_DDC GRPO_TTC_SAFE_THRESHOLD GRPO_DDC_SAFE_THRESHOLD; do
  if [[ -n "${!name:-}" ]]; then
    JOB_CMD+="${name}=$(printf '%q' "${!name}") "
  fi
done
if [[ -n "${GRPO_SCHEDULER_EPOCHS:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_EPOCHS=$(printf '%q' "${GRPO_SCHEDULER_EPOCHS}") "
fi
if [[ -n "${GRPO_SCHEDULER_WARMUP_EPOCHS:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_WARMUP_EPOCHS=$(printf '%q' "${GRPO_SCHEDULER_WARMUP_EPOCHS}") "
fi
if [[ -n "${GRPO_SCHEDULER_MIN_LR:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_MIN_LR=$(printf '%q' "${GRPO_SCHEDULER_MIN_LR}") "
fi
if [[ -n "${LIMIT_TRAIN_BATCHES:-}" ]]; then
  JOB_CMD+="LIMIT_TRAIN_BATCHES=$(printf '%q' "${LIMIT_TRAIN_BATCHES}") "
fi
if [[ -n "${LIMIT_VAL_BATCHES:-}" ]]; then
  JOB_CMD+="LIMIT_VAL_BATCHES=$(printf '%q' "${LIMIT_VAL_BATCHES}") "
fi
if [[ -n "${LOG_EVERY_N_STEPS:-}" ]]; then
  JOB_CMD+="LOG_EVERY_N_STEPS=$(printf '%q' "${LOG_EVERY_N_STEPS}") "
fi
for name in \
  OFFLINE_RL_ENABLED ELITE_BUFFER_DIR OFFLINE_RL_MISSING_BUFFER_POLICY \
  OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY \
  GRPO_PPO_REPLAY_INNER_EPOCHS GRPO_PPO_REPLAY_MINIBATCH_SIZE \
  GRPO_PPO_REPLAY_MAX_GRAD_NORM GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE \
  GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH \
  GRPO_PPO_REPLAY_BC_UPDATE GRPO_PPO_REPLAY_LOGPROB_MODE \
  GRPO_PPO_REPLAY_STEP_MINIBATCH_MODE GRPO_PPO_REPLAY_LOGPROB_CLAMP_MIN \
  GRPO_PPO_REPLAY_LOGPROB_CLAMP_MAX GRPO_PPO_REPLAY_STEP_CLIP_SCHEDULE \
  GRPO_PPO_REPLAY_STEP_CLIP_BASE GRPO_PPO_REPLAY_STEP_CLIP_RATE \
  TRAINER_GRADIENT_CLIP_VAL CHECKPOINT_EVERY_N_EPOCHS CHECKPOINT_EVERY_N_TRAIN_STEPS \
  GRPO_BUFFER_GUIDANCE_ENABLED GRPO_BUFFER_REWARD_BONUS_WEIGHT \
  GRPO_BUFFER_REWARD_BONUS_SCALE_M GRPO_BUFFER_REWARD_BONUS_USE_MARGIN \
  GRPO_BUFFER_DISTILL_LOSS_WEIGHT GRPO_BUFFER_DISTILL_LOSS_SCHEDULE \
  GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH \
  GRPO_BUFFER_DISTILL_WARMUP_EPOCHS GRPO_BUFFER_DISTILL_TOP_K \
  GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING \
  GRPO_SELF_IMITATION_LOSS_WEIGHT GRPO_SELF_IMITATION_LOSS_SCHEDULE \
  GRPO_SELF_IMITATION_LOSS_WEIGHT_START GRPO_SELF_IMITATION_WARMUP_START_EPOCH \
  GRPO_SELF_IMITATION_WARMUP_EPOCHS GRPO_SELF_IMITATION_TOP_K \
  GRPO_SELF_IMITATION_MIN_REWARD GRPO_SELF_IMITATION_MIN_REWARD_MARGIN \
  GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO GRPO_SELF_IMITATION_BATCH_CAP_SCORE \
  GRPO_SELF_IMITATION_BASELINE_MODE GRPO_SELF_IMITATION_TIMESTEP_SAMPLING \
  OFFLINE_RL_USE_BATCHED_PDM_SCORING OFFLINE_RL_USE_EXACT_ARRAY_PDM_STATE_CONVERSION \
  OFFLINE_RL_USE_FAST_PDM_SCORER OFFLINE_RL_PDM_BATCH_CHUNK_SIZE \
  OFFLINE_RL_PDM_SHADOW_CHECK OFFLINE_RL_PDM_SHADOW_MAX_SAMPLES \
  OFFLINE_RL_PDM_SHADOW_MAX_ABS_DIFF OFFLINE_RL_STRICT_REWARD_SUBMETRICS \
  OFFLINE_RL_MISSING_SUBMETRIC_POLICY; do
  if [[ -n "${!name:-}" ]]; then
    JOB_CMD+="${name}=$(printf '%q' "${!name}") "
  fi
done
JOB_CMD+="PYTHON_BIN=$(printf '%q' "${PYTHON_BIN}") "
JOB_CMD+="bash $(printf '%q' "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_rl_2b_local.sh")"

{
  printf 'name\tgpu\tcommand\n'
  printf 'stage3_rl_2b\t%s\t%s\n' "${GPU_LIST}" "${JOB_CMD}"
} > "${OUT_ROOT}/jobs.tsv"

{
  echo "run_root=${OUT_ROOT}"
  echo "start_early_gate_watcher=${START_EARLY_GATE_WATCHER}"
  echo "early_gate_threshold=${EARLY_GATE_THRESHOLD}"
  echo "early_gate_margin=${EARLY_GATE_MARGIN}"
  echo "early_gate_poll_seconds=${EARLY_GATE_POLL_SECONDS}"
  echo "early_gate_stop_on_fail=${EARLY_GATE_STOP_ON_FAIL}"
  echo "early_gate_rule=stop if PDMS < threshold - margin; watch if within margin; continue if PDMS >= threshold"
} > "${OUT_ROOT}/early_gate_config.txt"

setsid "${PYTHON_BIN}" "${STABLE_LAUNCHER}" \
  --jobs-tsv "${OUT_ROOT}/jobs.tsv" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds 15 \
  > "${OUT_ROOT}/launcher.log" 2>&1 < /dev/null &

echo "$!" > "${OUT_ROOT}/launcher.pid"
if [[ "${START_EARLY_GATE_WATCHER}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  setsid env \
    RUN_ROOT="${OUT_ROOT}" \
    THRESHOLD="${EARLY_GATE_THRESHOLD}" \
    MARGIN="${EARLY_GATE_MARGIN}" \
    POLL_SECONDS="${EARLY_GATE_POLL_SECONDS}" \
    STOP_ON_FAIL="${EARLY_GATE_STOP_ON_FAIL}" \
    bash "${REPO_ROOT}/scripts/training/watch_recogdrive_stage3_early_gate.sh" \
    > "${OUT_ROOT}/early_gate_watch.nohup.log" 2>&1 < /dev/null &
  echo "$!" > "${OUT_ROOT}/early_gate_watch.pid"
fi
echo "launched ${OUT_ROOT}"
