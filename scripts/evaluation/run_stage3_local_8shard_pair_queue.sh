#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage3 run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to a directory containing .ckpt files.}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${RUN_ROOT}/navtest_local_pair8_eval}"
GLOBAL_EVAL_LOCK_DIR="${GLOBAL_EVAL_LOCK_DIR:-${RUN_ROOT}/navtest_early_eval_global_locks}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_local_sharded_pdm.sh}"
WRAPPER_SCRIPT="${WRAPPER_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_checkpoint_eval_with_global_lock.sh}"

PAIR_SIZE="${PAIR_SIZE:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LOCAL_EVAL_SHARD_COUNT="${LOCAL_EVAL_SHARD_COUNT:-8}"
LOCAL_EVAL_MAX_CONCURRENT="${LOCAL_EVAL_MAX_CONCURRENT:-8}"
LOCAL_EVAL_MASTER_PORT_BASE="${LOCAL_EVAL_MASTER_PORT_BASE:-29600}"
LOCAL_EVAL_MASTER_PORT_STRIDE="${LOCAL_EVAL_MASTER_PORT_STRIDE:-100}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_MAX_MEM_USED_MB="${GPU_MAX_MEM_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MIN_EPOCH="${MIN_EPOCH:-0}"
MAX_EPOCH="${MAX_EPOCH:-999999}"
EXIT_WHEN_QUEUE_EMPTY="${EXIT_WHEN_QUEUE_EMPTY:-1}"

ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE:-$((ASYNC_PDM_WORKERS * 2))}"
ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY:-100}"
ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE:-0}"
ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE:-1}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle}"
VLM_PATH="${VLM_PATH:-}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
MAX_SCENES="${MAX_SCENES:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-600}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT does not exist: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "CHECKPOINT_ROOT does not exist: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "EVAL_SCRIPT does not exist: ${EVAL_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${WRAPPER_SCRIPT}" ]]; then
  echo "WRAPPER_SCRIPT does not exist: ${WRAPPER_SCRIPT}" >&2
  exit 2
fi

mkdir -p "${EVAL_OUT_ROOT}" "${GLOBAL_EVAL_LOCK_DIR}"
LOG_FILE="${EVAL_OUT_ROOT}/pair_queue.log"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG_FILE}"
}

checkpoint_id() {
  local name
  name="$(basename "$1" .ckpt)"
  echo "${name}" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/[=]+/_/g'
}

checkpoint_epoch() {
  local name
  name="$(basename "$1" .ckpt)"
  if [[ "${name}" =~ epoch[-_=]+([0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  echo ""
}

global_done_path() {
  local id="$1"
  printf '%s/%s.done\n' "${GLOBAL_EVAL_LOCK_DIR}" "${id}"
}

gpu_ready() {
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return 0
  fi

  "${PYTHON_BIN}" - <<PY
import subprocess
import sys

gpu_list = {item.strip() for item in "${GPU_LIST}".split(",") if item.strip()}
max_mem = int("${GPU_MAX_MEM_USED_MB}")
max_util = int("${GPU_MAX_UTIL}")
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
busy = []
for line in out.strip().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem_used, util = parts
    if idx not in gpu_list:
        continue
    if int(mem_used) > max_mem or int(util) > max_util:
        busy.append(f"{idx}:mem={mem_used}MB,util={util}%")
if busy:
    print("busy_gpus=" + ";".join(busy))
    sys.exit(1)
print("gpus_ready")
PY
}

collect_pending() {
  local ckpt id epoch
  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    id="$(checkpoint_id "${ckpt}")"
    epoch="$(checkpoint_epoch "${ckpt}")"
    if [[ -n "${epoch}" ]]; then
      if (( epoch < MIN_EPOCH || epoch > MAX_EPOCH )); then
        continue
      fi
    fi
    if [[ -f "$(global_done_path "${id}")" ]]; then
      continue
    fi
    printf '%s\n' "${ckpt}"
  done < <(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)
}

launch_eval() {
  local ckpt="$1"
  local slot="$2"
  local id out_dir port_base
  id="$(checkpoint_id "${ckpt}")"
  out_dir="${EVAL_OUT_ROOT}/${id}"
  port_base=$((LOCAL_EVAL_MASTER_PORT_BASE + slot * LOCAL_EVAL_MASTER_PORT_STRIDE))
  mkdir -p "${out_dir}"
  log "launch id=${id} slot=${slot} checkpoint=${ckpt} port_base=${port_base}"

  (
    set +e
    RUN_ROOT="${RUN_ROOT}" \
      CHECKPOINT="${ckpt}" \
      CHECKPOINT_ID="${id}" \
      EVAL_ROOT="${out_dir}" \
      GLOBAL_EVAL_LOCK_DIR="${GLOBAL_EVAL_LOCK_DIR}" \
      EVAL_SCRIPT="${EVAL_SCRIPT}" \
      GPU_LIST="${GPU_LIST}" \
      GPUS_PER_NODE="${GPUS_PER_NODE}" \
      LOCAL_EVAL_SHARD_COUNT="${LOCAL_EVAL_SHARD_COUNT}" \
      LOCAL_EVAL_MAX_CONCURRENT="${LOCAL_EVAL_MAX_CONCURRENT}" \
      LOCAL_EVAL_MASTER_PORT_BASE="${port_base}" \
      ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}" \
      ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}" \
      ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD}" \
      ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE}" \
      ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY}" \
      ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE}" \
      ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE}" \
      FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR}" \
      VLM_PATH="${VLM_PATH}" \
      METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH}" \
      SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH}" \
      DISABLE_TQDM="${DISABLE_TQDM}" \
      MAX_SCENES="${MAX_SCENES}" \
      DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS}" \
      bash "${WRAPPER_SCRIPT}" > "${out_dir}/wrapper.log" 2>&1
    rc=$?
    if [[ "${rc}" -eq 75 ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) skipped lock-busy id=${id}" >> "${out_dir}/wrapper.log"
      exit 0
    fi
    exit "${rc}"
  ) &
}

{
  echo "run_root=${RUN_ROOT}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "eval_out_root=${EVAL_OUT_ROOT}"
  echo "global_eval_lock_dir=${GLOBAL_EVAL_LOCK_DIR}"
  echo "eval_script=${EVAL_SCRIPT}"
  echo "wrapper_script=${WRAPPER_SCRIPT}"
  echo "pair_size=${PAIR_SIZE}"
  echo "gpu_list=${GPU_LIST}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "local_eval_shard_count=${LOCAL_EVAL_SHARD_COUNT}"
  echo "local_eval_max_concurrent=${LOCAL_EVAL_MAX_CONCURRENT}"
  echo "vlm_path=${VLM_PATH}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "navsim_log_path=${NAVSIM_LOG_PATH}"
  echo "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  echo "wait_for_free_gpus=${WAIT_FOR_FREE_GPUS}"
  echo "gpu_max_mem_used_mb=${GPU_MAX_MEM_USED_MB}"
  echo "gpu_max_util=${GPU_MAX_UTIL}"
  echo "min_epoch=${MIN_EPOCH}"
  echo "max_epoch=${MAX_EPOCH}"
  echo "max_scenes=${MAX_SCENES}"
} > "${EVAL_OUT_ROOT}/resolved_pair_queue_config.txt"

log "pair queue start"
while true; do
  mapfile -t pending < <(collect_pending)
  if (( ${#pending[@]} == 0 )); then
    log "no pending checkpoints"
    if [[ "${EXIT_WHEN_QUEUE_EMPTY}" == "1" ]]; then
      log "queue empty; exiting"
      exit 0
    fi
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! gpu_ready >> "${LOG_FILE}" 2>&1; then
    log "pending=${#pending[@]}; waiting for free local GPUs"
    sleep "${POLL_SECONDS}"
    continue
  fi

  pids=()
  launched=0
  for ckpt in "${pending[@]}"; do
    launch_eval "${ckpt}" "${launched}"
    pids+=("$!")
    launched=$((launched + 1))
    if (( launched >= PAIR_SIZE )); then
      break
    fi
  done

  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if (( rc != 0 )); then
    log "one or more eval jobs failed in this pair; continuing to remaining checkpoints"
  else
    log "pair finished launched=${launched}"
  fi
done
