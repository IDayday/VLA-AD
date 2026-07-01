#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the Stage 3 checkpoint file to evaluate.}"
EVAL_SPLIT="${EVAL_SPLIT:-navtrain}"
case "${EVAL_SPLIT}" in
  navtrain)
    DEFAULT_METRIC_CACHE_DIR="${ARTIFACT_ROOT}/cache/metric_cache_train_full"
    DEFAULT_NAVSIM_LOG_PATH="${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval"
    DEFAULT_SENSOR_BLOBS_PATH="${NAVSIM_DATA_ROOT}/trainval_sensor_blobs/trainval"
    ;;
  navtest)
    DEFAULT_METRIC_CACHE_DIR="${ARTIFACT_ROOT}/cache/metric_cache_navtest_full_v1"
    DEFAULT_NAVSIM_LOG_PATH="${NAVSIM_DATA_ROOT}/test_navsim_logs/test"
    DEFAULT_SENSOR_BLOBS_PATH="${NAVSIM_DATA_ROOT}/test_sensor_blobs/test"
    ;;
  *)
    echo "EVAL_SPLIT must be navtrain or navtest, got: ${EVAL_SPLIT}" >&2
    exit 2
    ;;
esac
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/policy_diversity_k${DIVERSITY_K:-8}_${EVAL_SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}"
VLM_PATH="${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${DEFAULT_METRIC_CACHE_DIR}}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${DEFAULT_NAVSIM_LOG_PATH}}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${DEFAULT_SENSOR_BLOBS_PATH}}"

DIVERSITY_K="${DIVERSITY_K:-8}"
DIVERSITY_SEED="${DIVERSITY_SEED:-260306049}"
DIVERSITY_DETERMINISTIC="${DIVERSITY_DETERMINISTIC:-0}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LOCAL_EVAL_SHARD_COUNT="${LOCAL_EVAL_SHARD_COUNT:-${GPUS_PER_NODE}}"
LOCAL_EVAL_MAX_CONCURRENT="${LOCAL_EVAL_MAX_CONCURRENT:-${LOCAL_EVAL_SHARD_COUNT}}"
LOCAL_EVAL_MASTER_PORT_BASE="${LOCAL_EVAL_MASTER_PORT_BASE:-29800}"

ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE:-$((ASYNC_PDM_WORKERS * 2))}"
ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY:-50}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
MAX_SCENES="${MAX_SCENES:-1200}"
EVAL_TOKEN_FILE="${EVAL_TOKEN_FILE:-}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-600}"
DRY_RUN="${DRY_RUN:-0}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NAVSIM_DISABLE_TQDM="${DISABLE_TQDM}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-0}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUT_ROOT}/torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
for required_dir in "${VLM_PATH}" "${METRIC_CACHE_DIR}" "${NUPLAN_MAPS_ROOT}" "${NAVSIM_LOG_PATH}" "${SENSOR_BLOBS_PATH}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Required directory does not exist: ${required_dir}" >&2
    exit 2
  fi
done
if ! [[ "${DIVERSITY_K}" =~ ^[1-9][0-9]*$ ]]; then
  echo "DIVERSITY_K must be positive, got: ${DIVERSITY_K}" >&2
  exit 2
fi
if [[ -n "${EVAL_TOKEN_FILE}" && ! -f "${EVAL_TOKEN_FILE}" ]]; then
  echo "EVAL_TOKEN_FILE does not exist: ${EVAL_TOKEN_FILE}" >&2
  exit 2
fi
if ! [[ "${LOCAL_EVAL_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LOCAL_EVAL_SHARD_COUNT must be positive, got: ${LOCAL_EVAL_SHARD_COUNT}" >&2
  exit 2
fi
if ! [[ "${LOCAL_EVAL_MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LOCAL_EVAL_MAX_CONCURRENT must be positive, got: ${LOCAL_EVAL_MAX_CONCURRENT}" >&2
  exit 2
fi
if (( LOCAL_EVAL_MAX_CONCURRENT > LOCAL_EVAL_SHARD_COUNT )); then
  LOCAL_EVAL_MAX_CONCURRENT="${LOCAL_EVAL_SHARD_COUNT}"
fi
if ! [[ "${LOCAL_EVAL_MASTER_PORT_BASE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "LOCAL_EVAL_MASTER_PORT_BASE must be positive, got: ${LOCAL_EVAL_MASTER_PORT_BASE}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if (( ${#GPU_ARRAY[@]} < LOCAL_EVAL_SHARD_COUNT )); then
  echo "GPU_LIST has ${#GPU_ARRAY[@]} entries but LOCAL_EVAL_SHARD_COUNT=${LOCAL_EVAL_SHARD_COUNT}: ${GPU_LIST}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"
ORCH_LOG="${OUT_ROOT}/orchestrator.log"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${ORCH_LOG}"
}

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "eval_split=${EVAL_SPLIT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "vlm_path=${VLM_PATH}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "navsim_log_path=${NAVSIM_LOG_PATH}"
  echo "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  echo "diversity_k=${DIVERSITY_K}"
  echo "diversity_seed=${DIVERSITY_SEED}"
  echo "diversity_deterministic=${DIVERSITY_DETERMINISTIC}"
  echo "gpu_list=${GPU_LIST}"
  echo "local_eval_shard_count=${LOCAL_EVAL_SHARD_COUNT}"
  echo "local_eval_max_concurrent=${LOCAL_EVAL_MAX_CONCURRENT}"
  echo "async_pdm_workers=${ASYNC_PDM_WORKERS}"
  echo "async_pdm_backend=${ASYNC_PDM_BACKEND}"
  echo "async_pdm_process_start_method=${ASYNC_PDM_PROCESS_START_METHOD}"
  echo "async_pdm_queue_size=${ASYNC_PDM_QUEUE_SIZE}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "torchinductor_compile_threads=${TORCHINDUCTOR_COMPILE_THREADS}"
  echo "torchinductor_fx_graph_cache=${TORCHINDUCTOR_FX_GRAPH_CACHE}"
  echo "torchinductor_cache_dir=${TORCHINDUCTOR_CACHE_DIR}"
  echo "max_scenes=${MAX_SCENES}"
  echo "eval_token_file=${EVAL_TOKEN_FILE}"
  echo "distributed_timeout_seconds=${DISTRIBUTED_TIMEOUT_SECONDS}"
} > "${OUT_ROOT}/resolved_local_sharded_diversity_config.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_local_sharded_diversity_config.txt"
  exit 0
fi

run_shard() {
  local shard_index="$1"
  local gpu_id="$2"
  local master_port="$3"
  local shard_dir="${OUT_ROOT}/shard_${shard_index}"
  mkdir -p "${shard_dir}"
  log "shard_${shard_index} start gpu=${gpu_id} port=${master_port}"

  local cmd=(
    "${PYTHON_BIN}"
    "${REPO_ROOT}/navsim/planning/script/run_recogdrive_policy_diversity_diagnostics.py"
    "train_test_split=${EVAL_SPLIT}"
    "agent=recogdrive_agent"
    "agent.checkpoint_path='${CHECKPOINT}'"
    "agent.vlm_path=${VLM_PATH}"
    "agent.cam_type=single"
    "agent.grpo=False"
    "agent.cache_hidden_state=False"
    "agent.vlm_type=internvl"
    "agent.dit_type=small"
    "agent.vlm_size=small"
    "agent.sampling_method=ddim"
    "metric_cache_path=${METRIC_CACHE_DIR}"
    "navsim_log_path=${NAVSIM_LOG_PATH}"
    "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
    "experiment_name=policy_diversity_k${DIVERSITY_K}_${EVAL_SPLIT}_shard_${shard_index}"
    "+diversity_k=${DIVERSITY_K}"
    "+diversity_seed=${DIVERSITY_SEED}"
    "+diversity_deterministic=${DIVERSITY_DETERMINISTIC}"
    "+async_pdm_workers=${ASYNC_PDM_WORKERS}"
    "+async_pdm_backend=${ASYNC_PDM_BACKEND}"
    "+async_pdm_process_start_method=${ASYNC_PDM_PROCESS_START_METHOD}"
    "+async_pdm_queue_size=${ASYNC_PDM_QUEUE_SIZE}"
    "+async_pdm_progress_every=${ASYNC_PDM_PROGRESS_EVERY}"
    "+eval_token_shard_count=${LOCAL_EVAL_SHARD_COUNT}"
    "+eval_token_shard_index=${shard_index}"
  )
  if [[ "${MAX_SCENES}" -gt 0 ]]; then
    cmd+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
  fi
  if [[ -n "${EVAL_TOKEN_FILE}" ]]; then
    cmd+=("+eval_token_file=${EVAL_TOKEN_FILE}")
  fi

  {
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "${shard_dir}/resolved_command.txt"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    NAVSIM_EXP_ROOT="${shard_dir}/hydra" \
    TORCHINDUCTOR_CACHE_DIR="${shard_dir}/torchinductor" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${master_port}" \
    WORLD_SIZE=1 \
    RANK=0 \
    LOCAL_RANK=0 \
    RECOGDRIVE_ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}" \
    RECOGDRIVE_ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}" \
    RECOGDRIVE_ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD}" \
    RECOGDRIVE_ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE}" \
    RECOGDRIVE_ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY}" \
    RECOGDRIVE_DIVERSITY_K="${DIVERSITY_K}" \
    RECOGDRIVE_DIVERSITY_SEED="${DIVERSITY_SEED}" \
    RECOGDRIVE_DIVERSITY_DETERMINISTIC="${DIVERSITY_DETERMINISTIC}" \
    RECOGDRIVE_EVAL_TOKEN_FILE="${EVAL_TOKEN_FILE}" \
    RECOGDRIVE_EVAL_TOKEN_SHARD_COUNT="${LOCAL_EVAL_SHARD_COUNT}" \
    RECOGDRIVE_EVAL_TOKEN_SHARD_INDEX="${shard_index}" \
    RECOGDRIVE_EVAL_DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS}" \
    "${cmd[@]}" > "${shard_dir}/eval.log" 2>&1

  log "shard_${shard_index} done"
}

rc=0
pids=()

wait_for_batch() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  pids=()
}

for ((i = 0; i < LOCAL_EVAL_SHARD_COUNT; i++)); do
  gpu="${GPU_ARRAY[$i]}"
  port=$((LOCAL_EVAL_MASTER_PORT_BASE + i))
  run_shard "${i}" "${gpu}" "${port}" &
  pids+=("$!")
  if (( ${#pids[@]} >= LOCAL_EVAL_MAX_CONCURRENT )); then
    wait_for_batch
  fi
done
wait_for_batch

if (( rc != 0 )); then
  log "one or more local diversity diagnostics shards failed"
  exit "${rc}"
fi

aggregate_csv="${OUT_ROOT}/local_sharded_aggregated.csv"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluation/aggregate_exact_pdm_shards.py" \
  --eval-root "${OUT_ROOT}" \
  --output-csv "${aggregate_csv}" \
  --summary-json "${OUT_ROOT}/aggregate_summary.json" | tee -a "${ORCH_LOG}"

log "local sharded policy diversity diagnostics complete aggregate_csv=${aggregate_csv}"
