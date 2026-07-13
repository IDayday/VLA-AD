#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-training-vla-zt2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_awac_iql_dualhost_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train_v2_${RUN_NAME}}"

SHARD_COUNT="${SHARD_COUNT:-16}"
LOCAL_SHARD_INDICES="${LOCAL_SHARD_INDICES:-0,1,2,3,4,5,6,7}"
REMOTE_SHARD_INDICES="${REMOTE_SHARD_INDICES:-8,9,10,11,12,13,14,15}"

ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES:-16}"
ELITE_TOP_M="${ELITE_TOP_M:-8}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
SKIP_EXISTING_RECORDS="${SKIP_EXISTING_RECORDS:-0}"
VALIDATE_EXISTING_RECORDS="${VALIDATE_EXISTING_RECORDS:-1}"
PREFILTER_EXISTING_RECORDS="${PREFILTER_EXISTING_RECORDS:-1}"
MERGE_EXISTING_RECORDS="${MERGE_EXISTING_RECORDS:-0}"
MERGE_KEEP_TOP_K="${MERGE_KEEP_TOP_K:-${ELITE_TOP_M}}"
MERGE_KEEP_SUPPORT="${MERGE_KEEP_SUPPORT:-1}"
AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING:-true}"
AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER:-true}"
AWAC_PDM_BATCH_CHUNK_SIZE="${AWAC_PDM_BATCH_CHUNK_SIZE:-0}"
AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-true}"
AWAC_PDM_SHADOW_MAX_SAMPLES="${AWAC_PDM_SHADOW_MAX_SAMPLES:-4}"
AWAC_PDM_SHADOW_MAX_ABS_DIFF="${AWAC_PDM_SHADOW_MAX_ABS_DIFF:-0.0}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_ACCUMULATE_GRAD_BATCHES="${TRAIN_ACCUMULATE_GRAD_BATCHES:-1}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS:-20}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_SCHEDULER_MIN_LR="${TRAIN_SCHEDULER_MIN_LR:-1e-5}"
TRAIN_AWAC_BC_LOSS_WEIGHT="${TRAIN_AWAC_BC_LOSS_WEIGHT:-0.05}"
TRAIN_AWAC_BC_LOSS_SCHEDULE="${TRAIN_AWAC_BC_LOSS_SCHEDULE:-constant}"
TRAIN_AWAC_BC_LOSS_WEIGHT_START="${TRAIN_AWAC_BC_LOSS_WEIGHT_START:-${TRAIN_AWAC_BC_LOSS_WEIGHT}}"
TRAIN_AWAC_BC_LOSS_WEIGHT_END="${TRAIN_AWAC_BC_LOSS_WEIGHT_END:-${TRAIN_AWAC_BC_LOSS_WEIGHT}}"
TRAIN_AWAC_BC_LOSS_SCHEDULE_EPOCHS="${TRAIN_AWAC_BC_LOSS_SCHEDULE_EPOCHS:-1}"
TRAIN_AWAC_ADVANTAGE_TEMPERATURE="${TRAIN_AWAC_ADVANTAGE_TEMPERATURE:-0.03}"
TRAIN_AWAC_WEIGHT_MAX="${TRAIN_AWAC_WEIGHT_MAX:-20.0}"
TRAIN_AWAC_REQUIRE_TTC_GUARD="${TRAIN_AWAC_REQUIRE_TTC_GUARD:-false}"
TRAIN_AWAC_TTC_MIN_ABSOLUTE="${TRAIN_AWAC_TTC_MIN_ABSOLUTE:-0.95}"
TRAIN_AWAC_TTC_MAX_RELATIVE_DROP="${TRAIN_AWAC_TTC_MAX_RELATIVE_DROP:-0.02}"
TRAIN_AWAC_PRIOR_DISTANCE_WEIGHT="${TRAIN_AWAC_PRIOR_DISTANCE_WEIGHT:-0.02}"
TRAIN_AWAC_JERK_PENALTY_WEIGHT="${TRAIN_AWAC_JERK_PENALTY_WEIGHT:-0.005}"
TRAIN_AWAC_SELECT_BY="${TRAIN_AWAC_SELECT_BY:-pdms_minus_prior}"
TRAIN_AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD="${TRAIN_AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD:-false}"
LAUNCH_REMOTE_EVAL_WATCHER="${LAUNCH_REMOTE_EVAL_WATCHER:-1}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-300}"
EVAL_MIN_CKPT_AGE_SECONDS="${EVAL_MIN_CKPT_AGE_SECONDS:-120}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh}"
EVAL_WAIT_FOR_FREE_GPUS="${EVAL_WAIT_FOR_FREE_GPUS:-0}"
EVAL_ASYNC_PDM_WORKERS="${EVAL_ASYNC_PDM_WORKERS:-2}"
EVAL_ASYNC_PDM_BACKEND="${EVAL_ASYNC_PDM_BACKEND:-process}"
EVAL_ASYNC_PDM_PROCESS_START_METHOD="${EVAL_ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
EVAL_ASYNC_PDM_QUEUE_SIZE="${EVAL_ASYNC_PDM_QUEUE_SIZE:-$((EVAL_ASYNC_PDM_WORKERS * 2))}"
EVAL_TOKEN_SHARD_COUNT="${EVAL_TOKEN_SHARD_COUNT:-1}"
EVAL_TOKEN_SHARD_INDEX="${EVAL_TOKEN_SHARD_INDEX:-0}"
EVAL_FAST_METRIC_CACHE_DIR="${EVAL_FAST_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_navtest_full_v1_fast_pickle}"
EVAL_MIN_CHECKPOINT_STEP="${EVAL_MIN_CHECKPOINT_STEP:-0}"
EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-0}"
EVAL_ALWAYS_EPOCH_CHECKPOINTS="${EVAL_ALWAYS_EPOCH_CHECKPOINTS:-1}"

RUN_BUFFER="${RUN_BUFFER:-1}"
RUN_TRAIN_AFTER_BUFFER="${RUN_TRAIN_AFTER_BUFFER:-1}"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${OUT_ROOT}"

{
  echo "repo_root=${REPO_ROOT}"
  echo "remote_host=${REMOTE_HOST}"
  echo "out_root=${OUT_ROOT}"
  echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
  echo "shard_count=${SHARD_COUNT}"
  echo "local_shard_indices=${LOCAL_SHARD_INDICES}"
  echo "remote_shard_indices=${REMOTE_SHARD_INDICES}"
  echo "online_policy_samples=${ONLINE_POLICY_SAMPLES}"
  echo "elite_top_m=${ELITE_TOP_M}"
  echo "skip_existing_records=${SKIP_EXISTING_RECORDS}"
  echo "validate_existing_records=${VALIDATE_EXISTING_RECORDS}"
  echo "merge_existing_records=${MERGE_EXISTING_RECORDS}"
  echo "merge_keep_top_k=${MERGE_KEEP_TOP_K}"
  echo "merge_keep_support=${MERGE_KEEP_SUPPORT}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "train_accumulate_grad_batches=${TRAIN_ACCUMULATE_GRAD_BATCHES}"
  echo "train_max_epochs=${TRAIN_MAX_EPOCHS}"
  echo "train_lr=${TRAIN_LR}"
  echo "train_scheduler_min_lr=${TRAIN_SCHEDULER_MIN_LR}"
  echo "train_awac_bc_loss_weight=${TRAIN_AWAC_BC_LOSS_WEIGHT}"
  echo "train_awac_bc_loss_schedule=${TRAIN_AWAC_BC_LOSS_SCHEDULE}"
  echo "train_awac_bc_loss_weight_start=${TRAIN_AWAC_BC_LOSS_WEIGHT_START}"
  echo "train_awac_bc_loss_weight_end=${TRAIN_AWAC_BC_LOSS_WEIGHT_END}"
  echo "train_awac_bc_loss_schedule_epochs=${TRAIN_AWAC_BC_LOSS_SCHEDULE_EPOCHS}"
  echo "train_awac_advantage_temperature=${TRAIN_AWAC_ADVANTAGE_TEMPERATURE}"
  echo "train_awac_weight_max=${TRAIN_AWAC_WEIGHT_MAX}"
  echo "train_awac_require_ttc_guard=${TRAIN_AWAC_REQUIRE_TTC_GUARD}"
  echo "train_awac_ttc_min_absolute=${TRAIN_AWAC_TTC_MIN_ABSOLUTE}"
  echo "train_awac_ttc_max_relative_drop=${TRAIN_AWAC_TTC_MAX_RELATIVE_DROP}"
  echo "train_awac_prior_distance_weight=${TRAIN_AWAC_PRIOR_DISTANCE_WEIGHT}"
  echo "train_awac_jerk_penalty_weight=${TRAIN_AWAC_JERK_PENALTY_WEIGHT}"
  echo "train_awac_select_by=${TRAIN_AWAC_SELECT_BY}"
  echo "train_awac_recompute_buffer_valid_mask_on_load=${TRAIN_AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD}"
  echo "launch_remote_eval_watcher=${LAUNCH_REMOTE_EVAL_WATCHER}"
  echo "eval_script=${EVAL_SCRIPT}"
  echo "eval_wait_for_free_gpus=${EVAL_WAIT_FOR_FREE_GPUS}"
  echo "eval_async_pdm_workers=${EVAL_ASYNC_PDM_WORKERS}"
  echo "eval_async_pdm_backend=${EVAL_ASYNC_PDM_BACKEND}"
  echo "eval_async_pdm_process_start_method=${EVAL_ASYNC_PDM_PROCESS_START_METHOD}"
  echo "eval_async_pdm_queue_size=${EVAL_ASYNC_PDM_QUEUE_SIZE}"
  echo "eval_token_shard_count=${EVAL_TOKEN_SHARD_COUNT}"
  echo "eval_token_shard_index=${EVAL_TOKEN_SHARD_INDEX}"
  echo "eval_fast_metric_cache_dir=${EVAL_FAST_METRIC_CACHE_DIR}"
  echo "run_buffer=${RUN_BUFFER}"
  echo "run_train_after_buffer=${RUN_TRAIN_AFTER_BUFFER}"
} > "${OUT_ROOT}/resolved_dualhost_pipeline.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_dualhost_pipeline.txt"
  echo "Dry run only. Set DRY_RUN=0 RUN_STAGE3=1 to launch."
  exit 0
fi

if [[ "${RUN_STAGE3:-0}" != "1" && "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "Dual-host AWAC/IQL pipeline is disabled by default. Set RUN_STAGE3=1 or RUN_TRAIN=1." >&2
  exit 2
fi

common_env=(
  "DRY_RUN=0"
  "RUN_STAGE3=1"
  "RUN_BUFFER=${RUN_BUFFER}"
  "RUN_TRAIN_AFTER_BUFFER=0"
  "VALIDATE_AFTER_BUFFER=0"
  "SHARD_COUNT=${SHARD_COUNT}"
  "ONLINE_POLICY_SAMPLES=${ONLINE_POLICY_SAMPLES}"
  "ELITE_TOP_M=${ELITE_TOP_M}"
  "BUFFER_BATCH_SIZE=${BUFFER_BATCH_SIZE}"
  "SKIP_EXISTING_RECORDS=${SKIP_EXISTING_RECORDS}"
  "VALIDATE_EXISTING_RECORDS=${VALIDATE_EXISTING_RECORDS}"
  "PREFILTER_EXISTING_RECORDS=${PREFILTER_EXISTING_RECORDS}"
  "MERGE_EXISTING_RECORDS=${MERGE_EXISTING_RECORDS}"
  "MERGE_KEEP_TOP_K=${MERGE_KEEP_TOP_K}"
  "MERGE_KEEP_SUPPORT=${MERGE_KEEP_SUPPORT}"
  "AWAC_USE_BATCHED_PDM_SCORING=${AWAC_USE_BATCHED_PDM_SCORING}"
  "AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION=${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}"
  "AWAC_USE_FAST_PDM_SCORER=${AWAC_USE_FAST_PDM_SCORER}"
  "AWAC_PDM_BATCH_CHUNK_SIZE=${AWAC_PDM_BATCH_CHUNK_SIZE}"
  "AWAC_PDM_SHADOW_CHECK=${AWAC_PDM_SHADOW_CHECK}"
  "AWAC_PDM_SHADOW_MAX_SAMPLES=${AWAC_PDM_SHADOW_MAX_SAMPLES}"
  "AWAC_PDM_SHADOW_MAX_ABS_DIFF=${AWAC_PDM_SHADOW_MAX_ABS_DIFF}"
  "ELITE_BUFFER_DIR=${ELITE_BUFFER_DIR}"
)

local_cmd=(
  env
  "OUT_ROOT=${OUT_ROOT}/local_buffer"
  "SHARD_INDICES=${LOCAL_SHARD_INDICES}"
  "${common_env[@]}"
  bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_awac_iql_full_pipeline_zt2.sh"
)

remote_cmd=(
  env
  "OUT_ROOT=${OUT_ROOT}/remote_buffer"
  "SHARD_INDICES=${REMOTE_SHARD_INDICES}"
  "${common_env[@]}"
  bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_awac_iql_full_pipeline_zt2.sh"
)

if [[ "${RUN_BUFFER}" == "1" ]]; then
  mkdir -p "${OUT_ROOT}/local_buffer" "${OUT_ROOT}/remote_buffer"
  printf '%q ' "${local_cmd[@]}" > "${OUT_ROOT}/local_buffer_command.txt"
  printf '\n' >> "${OUT_ROOT}/local_buffer_command.txt"
  printf 'cd %q && ' "${REPO_ROOT}" > "${OUT_ROOT}/remote_buffer_command.txt"
  printf '%q ' "${remote_cmd[@]}" >> "${OUT_ROOT}/remote_buffer_command.txt"
  printf '\n' >> "${OUT_ROOT}/remote_buffer_command.txt"

  (
    cd "${REPO_ROOT}"
    "${local_cmd[@]}"
  ) > "${OUT_ROOT}/local_buffer.log" 2>&1 &
  local_pid="$!"

  ssh "${REMOTE_HOST}" "cd ${REPO_ROOT@Q} && $(printf '%q ' "${remote_cmd[@]}")" \
    > "${OUT_ROOT}/remote_buffer.log" 2>&1 &
  remote_pid="$!"

  echo "${local_pid}" > "${OUT_ROOT}/local_buffer.pid"
  echo "${remote_pid}" > "${OUT_ROOT}/remote_buffer.pid"
  echo "launched local buffer pid=${local_pid}"
  echo "launched remote buffer ssh pid=${remote_pid}"

  failed=0
  if ! wait "${local_pid}"; then
    failed=1
  fi
  if ! wait "${remote_pid}"; then
    failed=1
  fi
  if [[ "${failed}" != "0" ]]; then
    echo "Local or remote AWAC buffer generation failed. See ${OUT_ROOT}/*buffer.log" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/training/validate_recogdrive_stage3_awac_elite_buffer.py" \
  --buffer-dir "${ELITE_BUFFER_DIR}" \
  --summary-json "${OUT_ROOT}/elite_buffer_validation_summary.json" \
  --summary-csv "${OUT_ROOT}/elite_buffer_validation_summary.csv" \
  --strict-v2

if [[ "${RUN_TRAIN_AFTER_BUFFER}" == "1" ]]; then
  if [[ "${LAUNCH_REMOTE_EVAL_WATCHER}" == "1" ]]; then
    eval_out="${OUT_ROOT}/remote_ckpt_eval"
    mkdir -p "${eval_out}"
    remote_watcher_cmd="$(
      cat <<EOF
cd ${REPO_ROOT@Q} && nohup env TRAIN_OUT_ROOT=${OUT_ROOT@Q}/train CHECKPOINT_ROOT=${OUT_ROOT@Q}/train/hydra EVAL_OUT_ROOT=${eval_out@Q} EVAL_RUN_NAME=${RUN_NAME@Q}_remote_ckpt_eval POLL_SECONDS=${EVAL_POLL_SECONDS@Q} MIN_CKPT_AGE_SECONDS=${EVAL_MIN_CKPT_AGE_SECONDS@Q} WAIT_FOR_FREE_GPUS=${EVAL_WAIT_FOR_FREE_GPUS@Q} GPU_MAX_MEM_USED_MB=2000 GPU_MAX_UTIL=10 EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY=0 EVAL_SCRIPT=${EVAL_SCRIPT@Q} ASYNC_PDM_BACKEND=${EVAL_ASYNC_PDM_BACKEND@Q} ASYNC_PDM_WORKERS=${EVAL_ASYNC_PDM_WORKERS@Q} ASYNC_PDM_QUEUE_SIZE=${EVAL_ASYNC_PDM_QUEUE_SIZE@Q} ASYNC_PDM_PROCESS_START_METHOD=${EVAL_ASYNC_PDM_PROCESS_START_METHOD@Q} EVAL_TOKEN_SHARD_COUNT=${EVAL_TOKEN_SHARD_COUNT@Q} EVAL_TOKEN_SHARD_INDEX=${EVAL_TOKEN_SHARD_INDEX@Q} FAST_METRIC_CACHE_DIR=${EVAL_FAST_METRIC_CACHE_DIR@Q} EVAL_MIN_CHECKPOINT_STEP=${EVAL_MIN_CHECKPOINT_STEP@Q} EVAL_CHECKPOINT_STEP_INTERVAL=${EVAL_CHECKPOINT_STEP_INTERVAL@Q} EVAL_ALWAYS_EPOCH_CHECKPOINTS=${EVAL_ALWAYS_EPOCH_CHECKPOINTS@Q} bash ${REPO_ROOT@Q}/scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh > ${eval_out@Q}/watcher.log 2>&1 < /dev/null & echo \$! > ${eval_out@Q}/watcher.pid
EOF
    )"
    ssh "${REMOTE_HOST}" "${remote_watcher_cmd}" \
      > "${OUT_ROOT}/remote_eval_watcher_launch.log" 2>&1 || {
        echo "Failed to launch remote eval watcher; see ${OUT_ROOT}/remote_eval_watcher_launch.log" >&2
        exit 1
      }
  fi
  cd "${REPO_ROOT}"
  OUT_ROOT="${OUT_ROOT}/train" \
  ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR}" \
  CACHE_MODE=online \
  ONLINE_AWAC_CANDIDATES=false \
  VALIDATE_ELITE_BUFFER=true \
  DRY_RUN=0 \
  RUN_STAGE3=1 \
  LR="${TRAIN_LR}" \
  MAX_EPOCHS="${TRAIN_MAX_EPOCHS}" \
  SCHEDULER_MIN_LR="${TRAIN_SCHEDULER_MIN_LR}" \
  BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  ACCUMULATE_GRAD_BATCHES="${TRAIN_ACCUMULATE_GRAD_BATCHES}" \
  AWAC_ELITE_TOP_M="${ELITE_TOP_M}" \
  AWAC_BC_LOSS_WEIGHT="${TRAIN_AWAC_BC_LOSS_WEIGHT}" \
  AWAC_BC_LOSS_SCHEDULE="${TRAIN_AWAC_BC_LOSS_SCHEDULE}" \
  AWAC_BC_LOSS_WEIGHT_START="${TRAIN_AWAC_BC_LOSS_WEIGHT_START}" \
  AWAC_BC_LOSS_WEIGHT_END="${TRAIN_AWAC_BC_LOSS_WEIGHT_END}" \
  AWAC_BC_LOSS_SCHEDULE_EPOCHS="${TRAIN_AWAC_BC_LOSS_SCHEDULE_EPOCHS}" \
  AWAC_ADVANTAGE_TEMPERATURE="${TRAIN_AWAC_ADVANTAGE_TEMPERATURE}" \
  AWAC_WEIGHT_MAX="${TRAIN_AWAC_WEIGHT_MAX}" \
  AWAC_REQUIRE_TTC_GUARD="${TRAIN_AWAC_REQUIRE_TTC_GUARD}" \
  AWAC_TTC_MIN_ABSOLUTE="${TRAIN_AWAC_TTC_MIN_ABSOLUTE}" \
  AWAC_TTC_MAX_RELATIVE_DROP="${TRAIN_AWAC_TTC_MAX_RELATIVE_DROP}" \
  AWAC_PRIOR_DISTANCE_WEIGHT="${TRAIN_AWAC_PRIOR_DISTANCE_WEIGHT}" \
  AWAC_JERK_PENALTY_WEIGHT="${TRAIN_AWAC_JERK_PENALTY_WEIGHT}" \
  AWAC_SELECT_BY="${TRAIN_AWAC_SELECT_BY}" \
  AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD="${TRAIN_AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD}" \
  bash "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh"
fi
