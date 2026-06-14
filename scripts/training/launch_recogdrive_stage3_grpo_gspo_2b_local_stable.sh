#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_grpo_gspo_refkl_s16_lr1e4_b2acc4_chunk16_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

# Conservative strict-GSPO defaults. Effective optimizer batch matches b4/acc2:
# 8 GPUs * micro-batch 2 * grad-accum 4 = 64 scenes/update.
export CACHE_MODE="${CACHE_MODE:-online}"
export KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
export GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export DDP_STRATEGY="${DDP_STRATEGY:-ddp}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export LR="${LR:-1e-4}"
export STAGE3_OBJECTIVE="${STAGE3_OBJECTIVE:-none}"
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
export GRPO_USE_GSPO_RATIO="${GRPO_USE_GSPO_RATIO:-true}"
export GRPO_GSPO_CLIP_LOW="${GRPO_GSPO_CLIP_LOW:-0.05}"
export GRPO_GSPO_CLIP_HIGH="${GRPO_GSPO_CLIP_HIGH:-0.05}"
export GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL="${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-4}"
export GRPO_BEHAVIOR_POLICY_SAMPLE="${GRPO_BEHAVIOR_POLICY_SAMPLE:-true}"
export GRPO_NORMALIZE_ADVANTAGE_BATCH="${GRPO_NORMALIZE_ADVANTAGE_BATCH:-false}"
export GRPO_ADVANTAGE_CLIP_ABS="${GRPO_ADVANTAGE_CLIP_ABS:-0.0}"
export GRPO_PPO_REPLAY_INNER_EPOCHS="${GRPO_PPO_REPLAY_INNER_EPOCHS:-1}"
export GRPO_PPO_REPLAY_MINIBATCH_SIZE="${GRPO_PPO_REPLAY_MINIBATCH_SIZE:-0}"
export GRPO_PPO_REPLAY_MAX_GRAD_NORM="${GRPO_PPO_REPLAY_MAX_GRAD_NORM:-1.0}"
export GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE="${GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE:-1e-6}"
export GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE="${GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE:-true}"
export GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH="${GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH:-true}"
export GRPO_PPO_REPLAY_BC_UPDATE="${GRPO_PPO_REPLAY_BC_UPDATE:-true}"
export GRPO_PPO_REPLAY_LOGPROB_MODE="${GRPO_PPO_REPLAY_LOGPROB_MODE:-trajectory}"
export GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${MAX_EPOCHS}}"
export GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}"
export GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"
export CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-300}"
export OFFLINE_RL_ENABLED="${OFFLINE_RL_ENABLED:-false}"
export ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-}"
export OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY="${OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY:-false}"
export GRPO_BUFFER_GUIDANCE_ENABLED="${GRPO_BUFFER_GUIDANCE_ENABLED:-false}"
export GRPO_BUFFER_REWARD_BONUS_WEIGHT="${GRPO_BUFFER_REWARD_BONUS_WEIGHT:-0.0}"
export GRPO_BUFFER_REWARD_BONUS_SCALE_M="${GRPO_BUFFER_REWARD_BONUS_SCALE_M:-4.0}"
export GRPO_BUFFER_REWARD_BONUS_USE_MARGIN="${GRPO_BUFFER_REWARD_BONUS_USE_MARGIN:-true}"
export GRPO_BUFFER_DISTILL_LOSS_WEIGHT="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT:-0.0}"
export GRPO_BUFFER_DISTILL_LOSS_SCHEDULE="${GRPO_BUFFER_DISTILL_LOSS_SCHEDULE:-linear_warmup}"
export GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START="${GRPO_BUFFER_DISTILL_LOSS_WEIGHT_START:-0.0}"
export GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH="${GRPO_BUFFER_DISTILL_WARMUP_START_EPOCH:-0}"
export GRPO_BUFFER_DISTILL_WARMUP_EPOCHS="${GRPO_BUFFER_DISTILL_WARMUP_EPOCHS:-3}"
export GRPO_BUFFER_DISTILL_TOP_K="${GRPO_BUFFER_DISTILL_TOP_K:-1}"
export GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN="${GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN:-0.0}"
export GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING="${GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING:-low_noise}"
export GRPO_SELF_IMITATION_LOSS_WEIGHT="${GRPO_SELF_IMITATION_LOSS_WEIGHT:-0.0}"
export GRPO_SELF_IMITATION_LOSS_SCHEDULE="${GRPO_SELF_IMITATION_LOSS_SCHEDULE:-linear_warmup}"
export GRPO_SELF_IMITATION_LOSS_WEIGHT_START="${GRPO_SELF_IMITATION_LOSS_WEIGHT_START:-0.0}"
export GRPO_SELF_IMITATION_WARMUP_START_EPOCH="${GRPO_SELF_IMITATION_WARMUP_START_EPOCH:-0}"
export GRPO_SELF_IMITATION_WARMUP_EPOCHS="${GRPO_SELF_IMITATION_WARMUP_EPOCHS:-3}"
export GRPO_SELF_IMITATION_TOP_K="${GRPO_SELF_IMITATION_TOP_K:-1}"
export GRPO_SELF_IMITATION_MIN_REWARD="${GRPO_SELF_IMITATION_MIN_REWARD:-0.85}"
export GRPO_SELF_IMITATION_MIN_REWARD_MARGIN="${GRPO_SELF_IMITATION_MIN_REWARD_MARGIN:-0.01}"
export GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO="${GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO:-1.0}"
export GRPO_SELF_IMITATION_BATCH_CAP_SCORE="${GRPO_SELF_IMITATION_BATCH_CAP_SCORE:-reward}"
export GRPO_SELF_IMITATION_BASELINE_MODE="${GRPO_SELF_IMITATION_BASELINE_MODE:-buffer_or_group_mean}"
export GRPO_SELF_IMITATION_TIMESTEP_SAMPLING="${GRPO_SELF_IMITATION_TIMESTEP_SAMPLING:-low_noise}"
export TRAIN_WAIT_FOR_FREE_GPUS="${TRAIN_WAIT_FOR_FREE_GPUS:-0}"
export TRAIN_GPU_MAX_MEM_USED_MB="${TRAIN_GPU_MAX_MEM_USED_MB:-2000}"
export TRAIN_GPU_MAX_UTIL="${TRAIN_GPU_MAX_UTIL:-5}"
export TRAIN_GPU_WAIT_POLL_SECONDS="${TRAIN_GPU_WAIT_POLL_SECONDS:-120}"
export START_EARLY_GATE_WATCHER="${START_EARLY_GATE_WATCHER:-1}"
export EARLY_GATE_THRESHOLD="${EARLY_GATE_THRESHOLD:-0.88}"
export EARLY_GATE_MARGIN="${EARLY_GATE_MARGIN:-0.005}"
export EARLY_GATE_POLL_SECONDS="${EARLY_GATE_POLL_SECONDS:-300}"
export EARLY_GATE_STOP_ON_FAIL="${EARLY_GATE_STOP_ON_FAIL:-0}"

RUN_TRAIN="${RUN_TRAIN:-0}"
LAUNCH_EVAL_WATCHERS="${LAUNCH_EVAL_WATCHERS:-0}"

PRIMARY_EVAL_HOST="${PRIMARY_EVAL_HOST:-training-vla-zt2}"
SECONDARY_EVAL_HOST="${SECONDARY_EVAL_HOST:-training-rl-zt3}"
PRIMARY_EVAL_DIR="${PRIMARY_EVAL_DIR:-${OUT_ROOT}/unique_lock_watch_on_vla_zt2_4gpu}"
SECONDARY_EVAL_DIR="${SECONDARY_EVAL_DIR:-${OUT_ROOT}/secondary_watch_on_rl_zt3_memfit_4gpu}"
GLOBAL_EVAL_LOCK_DIR="${GLOBAL_EVAL_LOCK_DIR:-${OUT_ROOT}/global_checkpoint_eval_locks}"

EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh}"
EVAL_FAST_METRIC_CACHE_DIR="${EVAL_FAST_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_navtest_full_v1_fast_pickle}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-60}"
EVAL_MIN_CKPT_AGE_SECONDS="${EVAL_MIN_CKPT_AGE_SECONDS:-120}"
EVAL_WAIT_FOR_FREE_GPUS="${EVAL_WAIT_FOR_FREE_GPUS:-1}"
EVAL_GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-4}"
PRIMARY_EVAL_GPU_LIST="${PRIMARY_EVAL_GPU_LIST:-4,5,6,7}"
SECONDARY_EVAL_GPU_LIST="${SECONDARY_EVAL_GPU_LIST:-0,1,2,4}"
EVAL_GPU_MAX_MEM_USED_MB="${EVAL_GPU_MAX_MEM_USED_MB:-60000}"
EVAL_GPU_MAX_UTIL="${EVAL_GPU_MAX_UTIL:-101}"
EVAL_ASYNC_PDM_WORKERS="${EVAL_ASYNC_PDM_WORKERS:-2}"
EVAL_ASYNC_PDM_BACKEND="${EVAL_ASYNC_PDM_BACKEND:-process}"
EVAL_ASYNC_PDM_PROCESS_START_METHOD="${EVAL_ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
EVAL_ASYNC_PDM_QUEUE_SIZE="${EVAL_ASYNC_PDM_QUEUE_SIZE:-$((EVAL_ASYNC_PDM_WORKERS * 2))}"
EVAL_ASYNC_PDM_PROGRESS_EVERY="${EVAL_ASYNC_PDM_PROGRESS_EVERY:-100}"
EVAL_ASYNC_PDM_PROFILE="${EVAL_ASYNC_PDM_PROFILE:-0}"
EVAL_ASYNC_PDM_TASK_CHUNK_SIZE="${EVAL_ASYNC_PDM_TASK_CHUNK_SIZE:-1}"
EVAL_TOKEN_SHARD_COUNT="${EVAL_TOKEN_SHARD_COUNT:-1}"
EVAL_TOKEN_SHARD_INDEX="${EVAL_TOKEN_SHARD_INDEX:-0}"
EVAL_PDM_RUNNER="${EVAL_PDM_RUNNER:-exact_pool}"
EVAL_DISABLE_TQDM="${EVAL_DISABLE_TQDM:-1}"
EVAL_MAX_SCENES="${EVAL_MAX_SCENES:-0}"

mkdir -p "${OUT_ROOT}"

write_launch_summary() {
  {
    echo "repo_root=${REPO_ROOT}"
    echo "run_name=${RUN_NAME}"
    echo "out_root=${OUT_ROOT}"
    echo "run_train=${RUN_TRAIN}"
    echo "launch_eval_watchers=${LAUNCH_EVAL_WATCHERS}"
    echo "cache_mode=${CACHE_MODE}"
    echo "kill_gpu_stress=${KILL_GPU_STRESS}"
    echo "gpu_list=${GPU_LIST}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "ddp_strategy=${DDP_STRATEGY}"
    echo "lr=${LR}"
    echo "stage3_objective=${STAGE3_OBJECTIVE}"
    echo "max_epochs=${MAX_EPOCHS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
    echo "grpo_sample_time=${GRPO_SAMPLE_TIME}"
    echo "bc_anneal=${BC_ANNEAL}"
    echo "bc_coeff_start=${BC_COEFF_START}"
    echo "bc_coeff_end=${BC_COEFF_END}"
    echo "bc_anneal_epochs=${BC_ANNEAL_EPOCHS}"
    echo "reference_kl_coeff=${REFERENCE_KL_COEFF}"
    echo "reference_kl_chunk_size=${REFERENCE_KL_CHUNK_SIZE}"
    echo "grpo_use_gspo_ratio=${GRPO_USE_GSPO_RATIO}"
    echo "grpo_gspo_clip_low=${GRPO_GSPO_CLIP_LOW}"
    echo "grpo_gspo_clip_high=${GRPO_GSPO_CLIP_HIGH}"
    echo "grpo_behavior_policy_sync_interval=${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL}"
    echo "grpo_behavior_policy_sample=${GRPO_BEHAVIOR_POLICY_SAMPLE}"
    echo "grpo_normalize_advantage_batch=${GRPO_NORMALIZE_ADVANTAGE_BATCH}"
    echo "grpo_advantage_clip_abs=${GRPO_ADVANTAGE_CLIP_ABS}"
    echo "grpo_ppo_replay_inner_epochs=${GRPO_PPO_REPLAY_INNER_EPOCHS}"
    echo "grpo_ppo_replay_minibatch_size=${GRPO_PPO_REPLAY_MINIBATCH_SIZE}"
    echo "grpo_ppo_replay_max_grad_norm=${GRPO_PPO_REPLAY_MAX_GRAD_NORM}"
    echo "grpo_ppo_replay_min_abs_advantage=${GRPO_PPO_REPLAY_MIN_ABS_ADVANTAGE}"
    echo "grpo_ppo_replay_filter_zero_advantage=${GRPO_PPO_REPLAY_FILTER_ZERO_ADVANTAGE}"
    echo "grpo_ppo_replay_sync_behavior_each_batch=${GRPO_PPO_REPLAY_SYNC_BEHAVIOR_EACH_BATCH}"
    echo "grpo_ppo_replay_bc_update=${GRPO_PPO_REPLAY_BC_UPDATE}"
    echo "grpo_ppo_replay_logprob_mode=${GRPO_PPO_REPLAY_LOGPROB_MODE}"
    echo "grpo_scheduler_epochs=${GRPO_SCHEDULER_EPOCHS}"
    echo "grpo_scheduler_warmup_epochs=${GRPO_SCHEDULER_WARMUP_EPOCHS}"
    echo "grpo_scheduler_min_lr=${GRPO_SCHEDULER_MIN_LR}"
    echo "limit_train_batches=${LIMIT_TRAIN_BATCHES}"
    echo "limit_val_batches=${LIMIT_VAL_BATCHES}"
    echo "log_every_n_steps=${LOG_EVERY_N_STEPS}"
    echo "checkpoint_every_n_epochs=${CHECKPOINT_EVERY_N_EPOCHS}"
    echo "checkpoint_every_n_train_steps=${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
    echo "start_early_gate_watcher=${START_EARLY_GATE_WATCHER}"
    echo "early_gate_threshold=${EARLY_GATE_THRESHOLD}"
    echo "early_gate_margin=${EARLY_GATE_MARGIN}"
    echo "early_gate_poll_seconds=${EARLY_GATE_POLL_SECONDS}"
    echo "early_gate_stop_on_fail=${EARLY_GATE_STOP_ON_FAIL}"
    echo "offline_rl_enabled=${OFFLINE_RL_ENABLED}"
    echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
    echo "offline_rl_cache_elite_records_in_memory=${OFFLINE_RL_CACHE_ELITE_RECORDS_IN_MEMORY}"
    echo "grpo_buffer_guidance_enabled=${GRPO_BUFFER_GUIDANCE_ENABLED}"
    echo "grpo_buffer_reward_bonus_weight=${GRPO_BUFFER_REWARD_BONUS_WEIGHT}"
    echo "grpo_buffer_reward_bonus_scale_m=${GRPO_BUFFER_REWARD_BONUS_SCALE_M}"
    echo "grpo_buffer_distill_loss_weight=${GRPO_BUFFER_DISTILL_LOSS_WEIGHT}"
    echo "grpo_buffer_distill_loss_schedule=${GRPO_BUFFER_DISTILL_LOSS_SCHEDULE}"
    echo "grpo_buffer_distill_warmup_epochs=${GRPO_BUFFER_DISTILL_WARMUP_EPOCHS}"
    echo "grpo_buffer_distill_top_k=${GRPO_BUFFER_DISTILL_TOP_K}"
    echo "grpo_buffer_distill_min_reward_margin=${GRPO_BUFFER_DISTILL_MIN_REWARD_MARGIN}"
    echo "grpo_buffer_distill_timestep_sampling=${GRPO_BUFFER_DISTILL_TIMESTEP_SAMPLING}"
    echo "grpo_self_imitation_loss_weight=${GRPO_SELF_IMITATION_LOSS_WEIGHT}"
    echo "grpo_self_imitation_loss_schedule=${GRPO_SELF_IMITATION_LOSS_SCHEDULE}"
    echo "grpo_self_imitation_loss_weight_start=${GRPO_SELF_IMITATION_LOSS_WEIGHT_START}"
    echo "grpo_self_imitation_warmup_start_epoch=${GRPO_SELF_IMITATION_WARMUP_START_EPOCH}"
    echo "grpo_self_imitation_warmup_epochs=${GRPO_SELF_IMITATION_WARMUP_EPOCHS}"
    echo "grpo_self_imitation_top_k=${GRPO_SELF_IMITATION_TOP_K}"
    echo "grpo_self_imitation_min_reward=${GRPO_SELF_IMITATION_MIN_REWARD}"
    echo "grpo_self_imitation_min_reward_margin=${GRPO_SELF_IMITATION_MIN_REWARD_MARGIN}"
    echo "grpo_self_imitation_max_target_scene_ratio=${GRPO_SELF_IMITATION_MAX_TARGET_SCENE_RATIO}"
    echo "grpo_self_imitation_batch_cap_score=${GRPO_SELF_IMITATION_BATCH_CAP_SCORE}"
    echo "grpo_self_imitation_baseline_mode=${GRPO_SELF_IMITATION_BASELINE_MODE}"
    echo "grpo_self_imitation_timestep_sampling=${GRPO_SELF_IMITATION_TIMESTEP_SAMPLING}"
    echo "train_wait_for_free_gpus=${TRAIN_WAIT_FOR_FREE_GPUS}"
    echo "train_gpu_max_mem_used_mb=${TRAIN_GPU_MAX_MEM_USED_MB}"
    echo "train_gpu_max_util=${TRAIN_GPU_MAX_UTIL}"
    echo "train_gpu_wait_poll_seconds=${TRAIN_GPU_WAIT_POLL_SECONDS}"
    echo "primary_eval_host=${PRIMARY_EVAL_HOST}"
    echo "secondary_eval_host=${SECONDARY_EVAL_HOST}"
    echo "primary_eval_gpu_list=${PRIMARY_EVAL_GPU_LIST}"
    echo "secondary_eval_gpu_list=${SECONDARY_EVAL_GPU_LIST}"
    echo "eval_gpu_max_mem_used_mb=${EVAL_GPU_MAX_MEM_USED_MB}"
    echo "eval_gpu_max_util=${EVAL_GPU_MAX_UTIL}"
    echo "eval_script=${EVAL_SCRIPT}"
    echo "eval_fast_metric_cache_dir=${EVAL_FAST_METRIC_CACHE_DIR}"
    echo "eval_async_pdm_workers=${EVAL_ASYNC_PDM_WORKERS}"
    echo "eval_async_pdm_backend=${EVAL_ASYNC_PDM_BACKEND}"
    echo "eval_async_pdm_process_start_method=${EVAL_ASYNC_PDM_PROCESS_START_METHOD}"
    echo "eval_async_pdm_queue_size=${EVAL_ASYNC_PDM_QUEUE_SIZE}"
    echo "eval_async_pdm_progress_every=${EVAL_ASYNC_PDM_PROGRESS_EVERY}"
    echo "eval_async_pdm_profile=${EVAL_ASYNC_PDM_PROFILE}"
    echo "eval_async_pdm_task_chunk_size=${EVAL_ASYNC_PDM_TASK_CHUNK_SIZE}"
    echo "eval_token_shard_count=${EVAL_TOKEN_SHARD_COUNT}"
    echo "eval_token_shard_index=${EVAL_TOKEN_SHARD_INDEX}"
    echo "eval_pdm_runner=${EVAL_PDM_RUNNER}"
    echo "eval_disable_tqdm=${EVAL_DISABLE_TQDM}"
    echo "eval_max_scenes=${EVAL_MAX_SCENES}"
  } > "${OUT_ROOT}/strict_gspo_launch_config.txt"
}

remote_start_watcher() {
  local host="$1"
  local eval_dir="$2"
  local gpu_list="$3"
  local external_summary_tsv="$4"
  local label="$5"

  local remote_cmd
  remote_cmd="$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${REPO_ROOT}")
mkdir -p $(printf '%q' "${eval_dir}") $(printf '%q' "${GLOBAL_EVAL_LOCK_DIR}")
if [[ -f $(printf '%q' "${eval_dir}/watcher.pid") ]] && kill -0 "\$(cat $(printf '%q' "${eval_dir}/watcher.pid"))" 2>/dev/null; then
  echo "watcher already alive host=${host} dir=${eval_dir} pid=\$(cat $(printf '%q' "${eval_dir}/watcher.pid"))"
  exit 0
fi
nohup env \
  TRAIN_OUT_ROOT=$(printf '%q' "${OUT_ROOT}") \
  CHECKPOINT_ROOT=$(printf '%q' "${OUT_ROOT}/train/hydra") \
  STATUS_FILE=$(printf '%q' "${OUT_ROOT}/status/stage3_rl_2b.json") \
  EVAL_OUT_ROOT=$(printf '%q' "${eval_dir}") \
  EVAL_RUN_NAME=$(printf '%q' "${RUN_NAME}_${label}") \
  POLL_SECONDS=$(printf '%q' "${EVAL_POLL_SECONDS}") \
  MIN_CKPT_AGE_SECONDS=$(printf '%q' "${EVAL_MIN_CKPT_AGE_SECONDS}") \
  WAIT_FOR_FREE_GPUS=$(printf '%q' "${EVAL_WAIT_FOR_FREE_GPUS}") \
  GPUS_PER_NODE=$(printf '%q' "${EVAL_GPUS_PER_NODE}") \
  GPU_LIST=$(printf '%q' "${gpu_list}") \
  GPU_MAX_MEM_USED_MB=$(printf '%q' "${EVAL_GPU_MAX_MEM_USED_MB}") \
  GPU_MAX_UTIL=$(printf '%q' "${EVAL_GPU_MAX_UTIL}") \
  EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY=0 \
  EVAL_SCRIPT=$(printf '%q' "${EVAL_SCRIPT}") \
  ASYNC_PDM_BACKEND=$(printf '%q' "${EVAL_ASYNC_PDM_BACKEND}") \
  ASYNC_PDM_WORKERS=$(printf '%q' "${EVAL_ASYNC_PDM_WORKERS}") \
  ASYNC_PDM_QUEUE_SIZE=$(printf '%q' "${EVAL_ASYNC_PDM_QUEUE_SIZE}") \
  ASYNC_PDM_PROCESS_START_METHOD=$(printf '%q' "${EVAL_ASYNC_PDM_PROCESS_START_METHOD}") \
  ASYNC_PDM_PROGRESS_EVERY=$(printf '%q' "${EVAL_ASYNC_PDM_PROGRESS_EVERY}") \
  ASYNC_PDM_PROFILE=$(printf '%q' "${EVAL_ASYNC_PDM_PROFILE}") \
  ASYNC_PDM_TASK_CHUNK_SIZE=$(printf '%q' "${EVAL_ASYNC_PDM_TASK_CHUNK_SIZE}") \
  EVAL_TOKEN_SHARD_COUNT=$(printf '%q' "${EVAL_TOKEN_SHARD_COUNT}") \
  EVAL_TOKEN_SHARD_INDEX=$(printf '%q' "${EVAL_TOKEN_SHARD_INDEX}") \
  PDM_EVAL_RUNNER=$(printf '%q' "${EVAL_PDM_RUNNER}") \
  FAST_METRIC_CACHE_DIR=$(printf '%q' "${EVAL_FAST_METRIC_CACHE_DIR}") \
  DISABLE_TQDM=$(printf '%q' "${EVAL_DISABLE_TQDM}") \
  MAX_SCENES=$(printf '%q' "${EVAL_MAX_SCENES}") \
  EXTERNAL_SUMMARY_TSV=$(printf '%q' "${external_summary_tsv}") \
  EXTERNAL_SUMMARY_SKIP_STATES=started,done \
  GLOBAL_EVAL_LOCK_DIR=$(printf '%q' "${GLOBAL_EVAL_LOCK_DIR}") \
  bash $(printf '%q' "${REPO_ROOT}/scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh") \
  > $(printf '%q' "${eval_dir}/watcher.log") 2>&1 < /dev/null &
echo "\$!" > $(printf '%q' "${eval_dir}/watcher.pid")
echo "started watcher host=${host} dir=${eval_dir} pid=\$(cat $(printf '%q' "${eval_dir}/watcher.pid"))"
EOF
)"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "bash -lc $(printf '%q' "${remote_cmd}")"
}

write_launch_summary

gpu_busy_report() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F, -v gpu_list="${GPU_LIST}" -v max_mem="${TRAIN_GPU_MAX_MEM_USED_MB}" -v max_util="${TRAIN_GPU_MAX_UTIL}" '
      BEGIN {
        n = split(gpu_list, wanted, ",");
        for (i = 1; i <= n; ++i) allow[wanted[i] + 0] = 1;
      }
      {
        idx = $1 + 0;
        mem = $2 + 0;
        util = $3 + 0;
        if (idx in allow && (mem > max_mem || util > max_util)) {
          printf("gpu=%d mem=%dMB util=%d%%\n", idx, mem, util);
        }
      }'
}

wait_for_train_gpus() {
  if [[ "${TRAIN_WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot wait for training GPUs." >&2
    exit 3
  fi
  mkdir -p "${OUT_ROOT}"
  local wait_log="${OUT_ROOT}/train_gpu_wait.log"
  while true; do
    local blocked
    blocked="$(gpu_busy_report || true)"
    if [[ -z "${blocked}" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) target GPUs free; launching training." | tee -a "${wait_log}"
      return 0
    fi
    {
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for target GPUs: ${GPU_LIST}"
      echo "${blocked}"
    } | tee -a "${wait_log}"
    sleep "${TRAIN_GPU_WAIT_POLL_SECONDS}"
  done
}

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "Strict GSPO launch config written to ${OUT_ROOT}/strict_gspo_launch_config.txt"
  echo "Dry run. Start with:"
  printf '  RUN_TRAIN=1 LAUNCH_EVAL_WATCHERS=1 RUN_NAME=%q bash %q\n' \
    "${RUN_NAME}" "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh"
  exit 0
fi

wait_for_train_gpus

RUN_NAME="${RUN_NAME}" \
OUT_ROOT="${OUT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh"

if [[ "${LAUNCH_EVAL_WATCHERS}" == "1" ]]; then
  remote_start_watcher "${PRIMARY_EVAL_HOST}" "${PRIMARY_EVAL_DIR}" "${PRIMARY_EVAL_GPU_LIST}" "" "primary_ckpt_eval"
  remote_start_watcher \
    "${SECONDARY_EVAL_HOST}" \
    "${SECONDARY_EVAL_DIR}" \
    "${SECONDARY_EVAL_GPU_LIST}" \
    "${PRIMARY_EVAL_DIR}/checkpoint_eval_summary.tsv" \
    "secondary_ckpt_eval"
fi

echo "strict GSPO run root: ${OUT_ROOT}"
echo "monitor: python ${REPO_ROOT}/scripts/training/monitor_recogdrive_stage3_grpo_run.py --run-name ${RUN_NAME}"
