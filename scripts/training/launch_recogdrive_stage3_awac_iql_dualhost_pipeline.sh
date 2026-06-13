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
AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING:-true}"
AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER:-true}"
AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-false}"
AWAC_PDM_SHADOW_MAX_SAMPLES="${AWAC_PDM_SHADOW_MAX_SAMPLES:-4}"
AWAC_PDM_SHADOW_MAX_ABS_DIFF="${AWAC_PDM_SHADOW_MAX_ABS_DIFF:-0.0}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_ACCUMULATE_GRAD_BATCHES="${TRAIN_ACCUMULATE_GRAD_BATCHES:-1}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS:-20}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_SCHEDULER_MIN_LR="${TRAIN_SCHEDULER_MIN_LR:-1e-5}"
LAUNCH_REMOTE_EVAL_WATCHER="${LAUNCH_REMOTE_EVAL_WATCHER:-1}"
EVAL_POLL_SECONDS="${EVAL_POLL_SECONDS:-300}"
EVAL_MIN_CKPT_AGE_SECONDS="${EVAL_MIN_CKPT_AGE_SECONDS:-120}"

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
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "train_accumulate_grad_batches=${TRAIN_ACCUMULATE_GRAD_BATCHES}"
  echo "train_max_epochs=${TRAIN_MAX_EPOCHS}"
  echo "train_lr=${TRAIN_LR}"
  echo "train_scheduler_min_lr=${TRAIN_SCHEDULER_MIN_LR}"
  echo "launch_remote_eval_watcher=${LAUNCH_REMOTE_EVAL_WATCHER}"
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
  "AWAC_USE_BATCHED_PDM_SCORING=${AWAC_USE_BATCHED_PDM_SCORING}"
  "AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION=${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}"
  "AWAC_USE_FAST_PDM_SCORER=${AWAC_USE_FAST_PDM_SCORER}"
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
    ssh "${REMOTE_HOST}" "cd ${REPO_ROOT@Q} && TRAIN_OUT_ROOT=${OUT_ROOT@Q}/train CHECKPOINT_ROOT=${OUT_ROOT@Q}/train/hydra EVAL_OUT_ROOT=${eval_out@Q} EVAL_RUN_NAME=${RUN_NAME@Q}_remote_ckpt_eval POLL_SECONDS=${EVAL_POLL_SECONDS@Q} MIN_CKPT_AGE_SECONDS=${EVAL_MIN_CKPT_AGE_SECONDS@Q} WAIT_FOR_FREE_GPUS=1 GPU_MAX_MEM_USED_MB=2000 GPU_MAX_UTIL=10 EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY=0 setsid bash ${REPO_ROOT@Q}/scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh > ${eval_out@Q}/watcher.log 2>&1 < /dev/null & echo \\\$! > ${eval_out@Q}/watcher.pid" \
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
  bash "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh"
fi
