#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
RUN_NAME="${RUN_NAME:-stage3_awac_iql_full_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train_v2_${RUN_NAME}}"

IL_CHECKPOINT="${IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
VLM_PATH="${VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${NAVSIM_DATA_ROOT}/trainval_sensor_blobs/trainval}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_INDICES="${SHARD_INDICES:-}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES:-16}"
ELITE_TOP_M="${ELITE_TOP_M:-8}"
SKIP_EXISTING_RECORDS="${SKIP_EXISTING_RECORDS:-0}"
VALIDATE_EXISTING_RECORDS="${VALIDATE_EXISTING_RECORDS:-1}"
AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING:-true}"
AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER:-true}"
AWAC_PDM_BATCH_CHUNK_SIZE="${AWAC_PDM_BATCH_CHUNK_SIZE:-0}"
AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-false}"
AWAC_PDM_SHADOW_MAX_SAMPLES="${AWAC_PDM_SHADOW_MAX_SAMPLES:-4}"
AWAC_PDM_SHADOW_MAX_ABS_DIFF="${AWAC_PDM_SHADOW_MAX_ABS_DIFF:-0.0}"

RUN_BUFFER="${RUN_BUFFER:-1}"
RUN_TRAIN_AFTER_BUFFER="${RUN_TRAIN_AFTER_BUFFER:-1}"
VALIDATE_AFTER_BUFFER="${VALIDATE_AFTER_BUFFER:-1}"
DRY_RUN="${DRY_RUN:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_ACCUMULATE_GRAD_BATCHES="${TRAIN_ACCUMULATE_GRAD_BATCHES:-1}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS:-20}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_SCHEDULER_MIN_LR="${TRAIN_SCHEDULER_MIN_LR:-1e-5}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-8}"

AWAC_STRICT_REWARD_SUBMETRICS="${AWAC_STRICT_REWARD_SUBMETRICS:-true}"
AWAC_MISSING_SUBMETRIC_POLICY="${AWAC_MISSING_SUBMETRIC_POLICY:-error}"
AWAC_REQUIRE_BUFFER_VALID_MASK="${AWAC_REQUIRE_BUFFER_VALID_MASK:-true}"
AWAC_SELECT_VALID_TOPK_ONLY="${AWAC_SELECT_VALID_TOPK_ONLY:-true}"
AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES="${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES:-false}"
AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT="${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT:-0.0}"
AWAC_USE_FINAL_HEADING_GUARD="${AWAC_USE_FINAL_HEADING_GUARD:-true}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export PYTHONUNBUFFERED=1

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
  fi
  for path in "${IL_CHECKPOINT}" "${METRIC_CACHE_DIR}" "${NAVSIM_LOG_PATH}" "${SENSOR_BLOBS_PATH}" "${NUPLAN_MAPS_ROOT}" "${VLM_PATH}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required path does not exist: ${path}" >&2
      exit 2
    fi
  done
fi

mkdir -p "${OUT_ROOT}" "${ELITE_BUFFER_DIR}"

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
  echo "gpu_list=${GPU_LIST}"
  echo "shard_count=${SHARD_COUNT}"
  echo "shard_indices=${SHARD_INDICES:-all}"
  echo "buffer_batch_size=${BUFFER_BATCH_SIZE}"
  echo "online_policy_samples=${ONLINE_POLICY_SAMPLES}"
  echo "elite_top_m=${ELITE_TOP_M}"
  echo "skip_existing_records=${SKIP_EXISTING_RECORDS}"
  echo "validate_existing_records=${VALIDATE_EXISTING_RECORDS}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "train_lr=${TRAIN_LR}"
  echo "train_max_epochs=${TRAIN_MAX_EPOCHS}"
  echo "train_scheduler_min_lr=${TRAIN_SCHEDULER_MIN_LR}"
  echo "run_buffer=${RUN_BUFFER}"
  echo "run_train_after_buffer=${RUN_TRAIN_AFTER_BUFFER}"
  echo "validate_after_buffer=${VALIDATE_AFTER_BUFFER}"
} > "${OUT_ROOT}/resolved_full_pipeline.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_full_pipeline.txt"
  echo "Dry run only. Set DRY_RUN=0 RUN_STAGE3=1 to launch."
  exit 0
fi

if [[ "${RUN_STAGE3:-0}" != "1" && "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "Full AWAC/IQL pipeline is disabled by default. Set RUN_STAGE3=1 or RUN_TRAIN=1." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty." >&2
  exit 2
fi

if [[ -n "${SHARD_INDICES}" ]]; then
  IFS=',' read -r -a SHARDS <<< "${SHARD_INDICES}"
else
  mapfile -t SHARDS < <(seq 0 $((SHARD_COUNT - 1)))
fi
if [[ "${#SHARDS[@]}" -eq 0 ]]; then
  echo "No shards selected." >&2
  exit 2
fi
for shard_index in "${SHARDS[@]}"; do
  if [[ ! "${shard_index}" =~ ^[0-9]+$ ]]; then
    echo "Invalid shard index: ${shard_index}" >&2
    exit 2
  fi
  if (( shard_index < 0 || shard_index >= SHARD_COUNT )); then
    echo "Shard index must be in [0, SHARD_COUNT): ${shard_index}/${SHARD_COUNT}" >&2
    exit 2
  fi
done

if [[ "${RUN_BUFFER}" == "1" ]]; then
  declare -a PIDS=()
  launch_pos=0
  : > "${OUT_ROOT}/buffer_launch.log"
  for shard_index in "${SHARDS[@]}"; do
    gpu="${GPUS[$((launch_pos % ${#GPUS[@]}))]}"
    launch_pos=$((launch_pos + 1))
    shard_out="${OUT_ROOT}/buffer_shard_${shard_index}"
    mkdir -p "${shard_out}"
    (
      cd "${REPO_ROOT}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export RUN_STAGE3=1
      export DRY_RUN=0
      export OUT_ROOT="${shard_out}"
      export ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR}"
      export IL_CHECKPOINT="${IL_CHECKPOINT}"
      export VLM_PATH="${VLM_PATH}"
      export METRIC_CACHE_DIR="${METRIC_CACHE_DIR}"
      export CACHE_MODE=true
      export BATCH_SIZE="${BUFFER_BATCH_SIZE}"
      export ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES}"
      export ELITE_TOP_M="${ELITE_TOP_M}"
      export SKIP_EXISTING_RECORDS="${SKIP_EXISTING_RECORDS}"
      export VALIDATE_EXISTING_RECORDS="${VALIDATE_EXISTING_RECORDS}"
      export SHARD_INDEX="${shard_index}"
      export SHARD_COUNT="${SHARD_COUNT}"
      export AWAC_STRICT_REWARD_SUBMETRICS="${AWAC_STRICT_REWARD_SUBMETRICS}"
      export AWAC_MISSING_SUBMETRIC_POLICY="${AWAC_MISSING_SUBMETRIC_POLICY}"
      export AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING}"
      export AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}"
      export AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER}"
      export AWAC_PDM_BATCH_CHUNK_SIZE="${AWAC_PDM_BATCH_CHUNK_SIZE}"
      export AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK}"
      export AWAC_PDM_SHADOW_MAX_SAMPLES="${AWAC_PDM_SHADOW_MAX_SAMPLES}"
      export AWAC_PDM_SHADOW_MAX_ABS_DIFF="${AWAC_PDM_SHADOW_MAX_ABS_DIFF}"
      export AWAC_REQUIRE_BUFFER_VALID_MASK="${AWAC_REQUIRE_BUFFER_VALID_MASK}"
      export AWAC_SELECT_VALID_TOPK_ONLY="${AWAC_SELECT_VALID_TOPK_ONLY}"
      export AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES="${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES}"
      export AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT="${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT}"
      export AWAC_USE_FINAL_HEADING_GUARD="${AWAC_USE_FINAL_HEADING_GUARD}"
      "${PYTHON_BIN}" scripts/training/build_recogdrive_stage3_awac_elite_buffer.py \
        train_test_split=navtrain \
        navsim_log_path="${NAVSIM_LOG_PATH}" \
        sensor_blobs_path="${SENSOR_BLOBS_PATH}" \
        output_dir="${shard_out}/hydra" \
        cache_path=null \
        force_cache_computation=False \
        use_cache_without_dataset=False \
        dataloader.params.batch_size="${BUFFER_BATCH_SIZE}" \
        dataloader.params.num_workers=0 \
        agent=recogdrive_agent \
        agent.cache_hidden_state=True \
        agent.cache_mode=True \
        agent.vlm_path="${VLM_PATH}" \
        agent.checkpoint_path="${IL_CHECKPOINT}" \
        agent.reference_policy_checkpoint="${IL_CHECKPOINT}" \
        agent.metric_cache_path="${METRIC_CACHE_DIR}" \
        agent.dit_type=small \
        agent.vlm_size=small \
        agent.sampling_method=ddim
    ) > "${shard_out}/builder.log" 2>&1 &
    PIDS+=("$!")
    echo "launched shard=${shard_index} gpu=${gpu} pid=${PIDS[-1]} log=${shard_out}/builder.log" \
      | tee -a "${OUT_ROOT}/buffer_launch.log"
  done

  failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "At least one AWAC elite buffer shard failed. See ${OUT_ROOT}/buffer_shard_*/builder.log" >&2
    exit 1
  fi
fi

if [[ "${VALIDATE_AFTER_BUFFER}" == "1" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/training/validate_recogdrive_stage3_awac_elite_buffer.py" \
    --buffer-dir "${ELITE_BUFFER_DIR}" \
    --summary-json "${OUT_ROOT}/elite_buffer_validation_summary.json" \
    --summary-csv "${OUT_ROOT}/elite_buffer_validation_summary.csv" \
    --strict-v2
fi

if [[ "${RUN_TRAIN_AFTER_BUFFER}" == "1" ]]; then
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
  GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE}" \
  IL_CHECKPOINT="${IL_CHECKPOINT}" \
  VLM_PATH="${VLM_PATH}" \
  METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
  NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH}" \
  SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH}" \
  AWAC_ELITE_TOP_M="${ELITE_TOP_M}" \
  AWAC_STRICT_REWARD_SUBMETRICS="${AWAC_STRICT_REWARD_SUBMETRICS}" \
  AWAC_MISSING_SUBMETRIC_POLICY="${AWAC_MISSING_SUBMETRIC_POLICY}" \
  AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING}" \
  AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}" \
  AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER}" \
  AWAC_PDM_BATCH_CHUNK_SIZE="${AWAC_PDM_BATCH_CHUNK_SIZE}" \
  AWAC_REQUIRE_BUFFER_VALID_MASK="${AWAC_REQUIRE_BUFFER_VALID_MASK}" \
  AWAC_SELECT_VALID_TOPK_ONLY="${AWAC_SELECT_VALID_TOPK_ONLY}" \
  AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES="${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES}" \
  AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT="${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT}" \
  AWAC_USE_FINAL_HEADING_GUARD="${AWAC_USE_FINAL_HEADING_GUARD}" \
  bash "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh"
fi
