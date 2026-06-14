#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:?Set TRAIN_OUT_ROOT to the Stage 3 training stable-launch output root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${TRAIN_OUT_ROOT}/train}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-stage3_safe_diffgrpo_ckpt_stream_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${EVAL_RUN_NAME}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu.sh}"

POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_CKPT_AGE_SECONDS="${MIN_CKPT_AGE_SECONDS:-120}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_MAX_MEM_USED_MB="${GPU_MAX_MEM_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
RETRY_FAILED="${RETRY_FAILED:-0}"
EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY="${EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY:-1}"
EXTERNAL_SUMMARY_TSV="${EXTERNAL_SUMMARY_TSV:-}"
EXTERNAL_SUMMARY_SKIP_STATES="${EXTERNAL_SUMMARY_SKIP_STATES:-started,done}"
GLOBAL_EVAL_LOCK_DIR="${GLOBAL_EVAL_LOCK_DIR:-}"

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
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
MAX_SCENES="${MAX_SCENES:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"

STATUS_FILE="${STATUS_FILE:-${TRAIN_OUT_ROOT}/status/stage3_rl_2b.json}"
ARCHIVE_DIR="${EVAL_OUT_ROOT}/checkpoint_archive"
STATE_DIR="${EVAL_OUT_ROOT}/state"
SUMMARY_TSV="${EVAL_OUT_ROOT}/checkpoint_eval_summary.tsv"
SUBMETRIC_SUMMARY_TSV="${EVAL_OUT_ROOT}/checkpoint_eval_submetrics.tsv"

mkdir -p "${ARCHIVE_DIR}" "${STATE_DIR}"
if [[ -n "${GLOBAL_EVAL_LOCK_DIR}" ]]; then
  mkdir -p "${GLOBAL_EVAL_LOCK_DIR}"
fi
exec 9>"${EVAL_OUT_ROOT}/watcher.lock"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another eval watcher already holds ${EVAL_OUT_ROOT}/watcher.lock; exiting"
  exit 0
fi

if [[ ! -f "${SUMMARY_TSV}" ]]; then
  printf 'timestamp\tcheckpoint_id\tstate\tcheckpoint\tarchive\teval_dir\trc\n' > "${SUMMARY_TSV}"
fi

{
  echo "train_out_root=${TRAIN_OUT_ROOT}"
  echo "checkpoint_root=${CHECKPOINT_ROOT}"
  echo "eval_out_root=${EVAL_OUT_ROOT}"
  echo "eval_script=${EVAL_SCRIPT}"
  echo "gpu_list=${GPU_LIST}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "wait_for_free_gpus=${WAIT_FOR_FREE_GPUS}"
  echo "gpu_max_mem_used_mb=${GPU_MAX_MEM_USED_MB}"
  echo "gpu_max_util=${GPU_MAX_UTIL}"
  echo "async_pdm_workers=${ASYNC_PDM_WORKERS}"
  echo "async_pdm_backend=${ASYNC_PDM_BACKEND}"
  echo "async_pdm_process_start_method=${ASYNC_PDM_PROCESS_START_METHOD}"
  echo "async_pdm_queue_size=${ASYNC_PDM_QUEUE_SIZE}"
  echo "async_pdm_progress_every=${ASYNC_PDM_PROGRESS_EVERY}"
  echo "async_pdm_profile=${ASYNC_PDM_PROFILE}"
  echo "async_pdm_task_chunk_size=${ASYNC_PDM_TASK_CHUNK_SIZE}"
  echo "eval_token_shard_count=${EVAL_TOKEN_SHARD_COUNT}"
  echo "eval_token_shard_index=${EVAL_TOKEN_SHARD_INDEX}"
  echo "pdm_eval_runner=${PDM_EVAL_RUNNER}"
  echo "fast_metric_cache_dir=${FAST_METRIC_CACHE_DIR}"
  echo "disable_tqdm=${DISABLE_TQDM}"
  echo "max_scenes=${MAX_SCENES}"
  echo "distributed_timeout_seconds=${DISTRIBUTED_TIMEOUT_SECONDS}"
  echo "external_summary_tsv=${EXTERNAL_SUMMARY_TSV}"
  echo "global_eval_lock_dir=${GLOBAL_EVAL_LOCK_DIR}"
} > "${EVAL_OUT_ROOT}/watcher_resolved_config.txt"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

checkpoint_id() {
  local name
  name="$(basename "$1" .ckpt)"
  echo "${name}" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/[=]+/_/g'
}

training_state() {
  local status_file="${STATUS_FILE}"
  if [[ ! -f "${status_file}" ]]; then
    status_file="$(find "${TRAIN_OUT_ROOT}/status" -maxdepth 1 -type f -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
  fi
  if [[ -z "${status_file}" || ! -f "${status_file}" ]]; then
    echo "unknown"
    return
  fi
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

try:
    payload = json.loads(Path("${status_file}").read_text())
except Exception:
    print("unknown")
else:
    print(payload.get("state", "unknown"))
PY
}

checkpoint_old_enough() {
  local ckpt="$1"
  local mtime now age
  mtime="$(stat -c '%Y' "${ckpt}")"
  now="$(date +%s)"
  age=$((now - mtime))
  [[ "${age}" -ge "${MIN_CKPT_AGE_SECONDS}" ]]
}

archive_checkpoint() {
  local ckpt="$1"
  local id archive
  id="$(checkpoint_id "${ckpt}")"
  archive="${ARCHIVE_DIR}/${id}.ckpt"

  if [[ ! -f "${archive}" ]]; then
    if ln "${ckpt}" "${archive}" 2>/dev/null; then
      log "archived checkpoint via hardlink id=${id} archive=${archive}"
    else
      cp -p "${ckpt}" "${archive}"
      log "archived checkpoint via copy id=${id} archive=${archive}"
    fi
  fi

  printf '%s\n%s\n' "${ckpt}" "${archive}" > "${STATE_DIR}/${id}.paths"
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
    mem_used_i = int(mem_used)
    util_i = int(util)
    if mem_used_i > max_mem or util_i > max_util:
        busy.append(f"{idx}:mem={mem_used_i}MB,util={util_i}%")

if busy:
    print("busy_gpus=" + ";".join(busy))
    sys.exit(1)

print("gpus_ready")
PY
}

record_summary() {
  local id="$1"
  local state="$2"
  local checkpoint="$3"
  local archive="$4"
  local eval_dir="$5"
  local rc="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${id}" \
    "${state}" \
    "${checkpoint}" \
    "${archive}" \
    "${eval_dir}" \
    "${rc}" >> "${SUMMARY_TSV}"
}

external_summary_has_checkpoint() {
  local id="$1"
  if [[ -z "${EXTERNAL_SUMMARY_TSV}" || ! -f "${EXTERNAL_SUMMARY_TSV}" ]]; then
    return 1
  fi

  awk -F '\t' -v id="${id}" -v states="${EXTERNAL_SUMMARY_SKIP_STATES}" '
    BEGIN {
      n = split(states, arr, ",")
      for (i = 1; i <= n; i++) {
        wanted[arr[i]] = 1
      }
    }
    $2 == id && wanted[$3] {
      found = 1
      exit
    }
    END {
      exit found ? 0 : 1
    }
  ' "${EXTERNAL_SUMMARY_TSV}"
}

eval_lock_path() {
  local id="$1"
  if [[ -z "${GLOBAL_EVAL_LOCK_DIR}" ]]; then
    return 1
  fi
  printf '%s/%s.lock\n' "${GLOBAL_EVAL_LOCK_DIR}" "${id}"
}

evaluate_archive() {
  local archive="$1"
  local id eval_dir checkpoint rc master_port lock_path eval_lock_fd
  id="$(checkpoint_id "${archive}")"
  eval_dir="${EVAL_OUT_ROOT}/eval_${id}"

  if [[ -f "${STATE_DIR}/${id}.done" ]]; then
    return 0
  fi
  if [[ -f "${STATE_DIR}/${id}.running" ]]; then
    return 0
  fi
  if [[ -f "${STATE_DIR}/${id}.failed" && "${RETRY_FAILED}" != "1" ]]; then
    return 0
  fi
  if external_summary_has_checkpoint "${id}"; then
    log "external summary already has checkpoint id=${id}; skipping"
    touch "${STATE_DIR}/${id}.external_skipped"
    return 0
  fi

  checkpoint="${archive}"
  if [[ -f "${STATE_DIR}/${id}.paths" ]]; then
    checkpoint="$(sed -n '1p' "${STATE_DIR}/${id}.paths")"
  fi

  mkdir -p "${eval_dir}"
  if lock_path="$(eval_lock_path "${id}")"; then
    exec {eval_lock_fd}>"${lock_path}"
    if ! flock -n "${eval_lock_fd}"; then
      log "another watcher holds global checkpoint lock id=${id} lock=${lock_path}; skipping"
      eval "exec ${eval_lock_fd}>&-"
      return 0
    fi
  fi
  if external_summary_has_checkpoint "${id}"; then
    log "external summary claimed checkpoint after lock id=${id}; skipping"
    touch "${STATE_DIR}/${id}.external_skipped"
    if [[ -n "${eval_lock_fd:-}" ]]; then
      flock -u "${eval_lock_fd}" || true
      eval "exec ${eval_lock_fd}>&-"
    fi
    return 0
  fi

  rm -f "${STATE_DIR}/${id}.failed"
  printf '%s\n' "${archive}" > "${STATE_DIR}/${id}.running"
  record_summary "${id}" "started" "${checkpoint}" "${archive}" "${eval_dir}" "-"
  log "starting eval id=${id} archive=${archive} out=${eval_dir}"

  master_port="$("${PYTHON_BIN}" - <<PY
import random
print(random.randint(63700, 64500))
PY
)"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    CHECKPOINT="${archive}" \
    OUT_ROOT="${eval_dir}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    MASTER_PORT="${master_port}" \
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
    DISABLE_TQDM="${DISABLE_TQDM}" \
    MAX_SCENES="${MAX_SCENES}" \
    DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS}" \
    bash "${EVAL_SCRIPT}" > "${eval_dir}/eval.log" 2>&1
  rc=$?
  set -e

  rm -f "${STATE_DIR}/${id}.running"
  if [[ "${rc}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluation/summarize_recogdrive_eval_submetrics.py" \
      --eval-dir "${eval_dir}" \
      --checkpoint-id "${id}" \
      --summary-tsv "${SUBMETRIC_SUMMARY_TSV}" \
      >> "${eval_dir}/eval.log" 2>&1 || log "submetric summary failed id=${id}; see ${eval_dir}/eval.log"
    touch "${STATE_DIR}/${id}.done"
    record_summary "${id}" "done" "${checkpoint}" "${archive}" "${eval_dir}" "${rc}"
    log "eval done id=${id}"
  else
    touch "${STATE_DIR}/${id}.failed"
    record_summary "${id}" "failed" "${checkpoint}" "${archive}" "${eval_dir}" "${rc}"
    log "eval failed id=${id} rc=${rc} log=${eval_dir}/eval.log"
  fi
  if [[ -n "${eval_lock_fd:-}" ]]; then
    flock -u "${eval_lock_fd}" || true
    eval "exec ${eval_lock_fd}>&-"
  fi
}

while true; do
  state="$(training_state)"

  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    if checkpoint_old_enough "${ckpt}"; then
      archive_checkpoint "${ckpt}"
    else
      log "checkpoint not old enough yet: ${ckpt}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)

  pending=()
  while IFS= read -r archive; do
    [[ -n "${archive}" ]] || continue
    id="$(checkpoint_id "${archive}")"
    if [[ -f "${STATE_DIR}/${id}.done" ]]; then
      continue
    fi
    if [[ -f "${STATE_DIR}/${id}.running" ]]; then
      continue
    fi
    if [[ -f "${STATE_DIR}/${id}.failed" && "${RETRY_FAILED}" != "1" ]]; then
      continue
    fi
    if [[ -f "${STATE_DIR}/${id}.external_skipped" ]]; then
      continue
    fi
    if external_summary_has_checkpoint "${id}"; then
      log "external summary already has checkpoint id=${id}; skipping pending archive"
      touch "${STATE_DIR}/${id}.external_skipped"
      continue
    fi
    pending+=("${archive}")
  done < <(find "${ARCHIVE_DIR}" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)

  if [[ "${#pending[@]}" -gt 0 ]]; then
    if gpu_ready; then
      evaluate_archive "${pending[0]}"
      continue
    fi
    log "pending_evals=${#pending[@]} training_state=${state}; waiting for free GPUs"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if [[ "${state}" == "done" && "${EXIT_WHEN_TRAINING_DONE_AND_QUEUE_EMPTY}" == "1" ]]; then
    log "training done and checkpoint eval queue empty; exiting"
    exit 0
  fi

  if [[ "${state}" == "failed" || "${state}" == "launch_failed" ]]; then
    log "training_state=${state}; no pending checkpoint evals"
  else
    log "no pending checkpoint evals; training_state=${state}"
  fi
  sleep "${POLL_SECONDS}"
done
