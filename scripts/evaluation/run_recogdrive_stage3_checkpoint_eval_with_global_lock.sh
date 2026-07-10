#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the Stage3 training output root.}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the checkpoint file to evaluate.}"
CHECKPOINT_ID="${CHECKPOINT_ID:-}"
if [[ -z "${CHECKPOINT_ID}" ]]; then
  CHECKPOINT_ID="$(basename "${CHECKPOINT}" .ckpt | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/[=]+/_/g')"
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE:-$((ASYNC_PDM_WORKERS * 2))}"
ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY:-100}"
ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE:-0}"
ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE:-1}"
EVAL_TOKEN_SHARD_COUNT="${EVAL_TOKEN_SHARD_COUNT:-1}"
EVAL_TOKEN_SHARD_INDEX="${EVAL_TOKEN_SHARD_INDEX:-0}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle}"
VLM_PATH="${VLM_PATH:-}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
MAX_SCENES="${MAX_SCENES:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh}"

EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/local_foreground_eval_${CHECKPOINT_ID}_${GPUS_PER_NODE}gpu}"
EVAL_DIR="${EVAL_DIR:-${EVAL_ROOT}/eval_${CHECKPOINT_ID}}"
GLOBAL_EVAL_LOCK_DIR="${GLOBAL_EVAL_LOCK_DIR:-${RUN_ROOT}/global_checkpoint_eval_locks}"
LOCK="${GLOBAL_EVAL_LOCK_DIR}/${CHECKPOINT_ID}.lock"
DONE="${GLOBAL_EVAL_LOCK_DIR}/${CHECKPOINT_ID}.done"
SUMMARY_TSV="${EVAL_ROOT}/checkpoint_eval_summary.tsv"
SUBMETRIC_SUMMARY_TSV="${EVAL_ROOT}/checkpoint_eval_submetrics.tsv"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT does not exist: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "EVAL_SCRIPT does not exist: ${EVAL_SCRIPT}" >&2
  exit 2
fi

if [[ -z "${MASTER_PORT}" ]]; then
  MASTER_PORT="$("${PYTHON_BIN}" - <<'PY'
import random
print(random.randint(63700, 64500))
PY
)"
fi

mkdir -p "${EVAL_DIR}" "${GLOBAL_EVAL_LOCK_DIR}"

(
  flock -n 200 || {
    echo "checkpoint eval lock is busy: ${LOCK}" >&2
    exit 75
  }

  if [[ -f "${DONE}" ]]; then
    echo "done marker already exists: ${DONE}"
    exit 0
  fi

  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_root=${RUN_ROOT}"
    echo "checkpoint_id=${CHECKPOINT_ID}"
    echo "checkpoint=${CHECKPOINT}"
    echo "eval_root=${EVAL_ROOT}"
    echo "eval_dir=${EVAL_DIR}"
    echo "gpu_list=${GPU_LIST}"
    echo "gpus_per_node=${GPUS_PER_NODE}"
    echo "master_port=${MASTER_PORT}"
    echo "pdm_eval_runner=${PDM_EVAL_RUNNER}"
    echo "fast_metric_cache_dir=${FAST_METRIC_CACHE_DIR}"
  } | tee "${EVAL_DIR}/launcher_status.log"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    CHECKPOINT="${CHECKPOINT}" \
    OUT_ROOT="${EVAL_DIR}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    MASTER_PORT="${MASTER_PORT}" \
    ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}" \
    ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}" \
    ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD}" \
    ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE}" \
    ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY}" \
    ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE}" \
    ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE}" \
    EVAL_TOKEN_SHARD_COUNT="${EVAL_TOKEN_SHARD_COUNT}" \
    EVAL_TOKEN_SHARD_INDEX="${EVAL_TOKEN_SHARD_INDEX}" \
    PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER}" \
    FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR}" \
    VLM_PATH="${VLM_PATH}" \
    METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
    NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH}" \
    SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH}" \
    DISABLE_TQDM="${DISABLE_TQDM}" \
    MAX_SCENES="${MAX_SCENES}" \
    DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS}" \
    bash "${EVAL_SCRIPT}" > "${EVAL_DIR}/eval.log" 2>&1
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluation/summarize_recogdrive_eval_submetrics.py" \
      --eval-dir "${EVAL_DIR}" \
      --checkpoint-id "${CHECKPOINT_ID}" \
      --summary-tsv "${SUBMETRIC_SUMMARY_TSV}" >> "${EVAL_DIR}/eval.log" 2>&1

    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluation/analyze_recogdrive_stage3_navtest_pdms.py" \
      --run-root "${RUN_ROOT}" \
      --output-tsv "${RUN_ROOT}/navtest_pdms_analysis.tsv" \
      --output-md "${RUN_ROOT}/navtest_pdms_analysis.md" >> "${EVAL_DIR}/eval.log" 2>&1 || true

    if [[ ! -f "${SUMMARY_TSV}" ]]; then
      printf 'timestamp\tcheckpoint_id\tstate\tcheckpoint\tarchive\teval_dir\trc\n' > "${SUMMARY_TSV}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${CHECKPOINT_ID}" \
      "done" \
      "${CHECKPOINT}" \
      "${CHECKPOINT}" \
      "${EVAL_DIR}" \
      "${rc}" >> "${SUMMARY_TSV}"

    tmp="${DONE}.tmp.$$"
    {
      printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf 'checkpoint_id=%s\n' "${CHECKPOINT_ID}"
      printf 'checkpoint=%s\n' "${CHECKPOINT}"
      printf 'archive=%s\n' "${CHECKPOINT}"
      printf 'eval_out_root=%s\n' "${EVAL_ROOT}"
      printf 'eval_dir=%s\n' "${EVAL_DIR}"
    } > "${tmp}"
    mv "${tmp}" "${DONE}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) eval done checkpoint_id=${CHECKPOINT_ID} rc=${rc}" | tee -a "${EVAL_DIR}/launcher_status.log"
  else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) eval failed checkpoint_id=${CHECKPOINT_ID} rc=${rc}; see ${EVAL_DIR}/eval.log" | tee -a "${EVAL_DIR}/launcher_status.log"
    exit "${rc}"
  fi
) 200>"${LOCK}"
