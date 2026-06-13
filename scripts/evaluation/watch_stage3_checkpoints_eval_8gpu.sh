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

STATUS_FILE="${TRAIN_OUT_ROOT}/status/stage3_rl_2b.json"
ARCHIVE_DIR="${EVAL_OUT_ROOT}/checkpoint_archive"
STATE_DIR="${EVAL_OUT_ROOT}/state"
SUMMARY_TSV="${EVAL_OUT_ROOT}/checkpoint_eval_summary.tsv"
SUBMETRIC_SUMMARY_TSV="${EVAL_OUT_ROOT}/checkpoint_eval_submetrics.tsv"

mkdir -p "${ARCHIVE_DIR}" "${STATE_DIR}"
exec 9>"${EVAL_OUT_ROOT}/watcher.lock"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another eval watcher already holds ${EVAL_OUT_ROOT}/watcher.lock; exiting"
  exit 0
fi

if [[ ! -f "${SUMMARY_TSV}" ]]; then
  printf 'timestamp\tcheckpoint_id\tstate\tcheckpoint\tarchive\teval_dir\trc\n' > "${SUMMARY_TSV}"
fi

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
}

checkpoint_id() {
  local name
  name="$(basename "$1" .ckpt)"
  echo "${name}" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/[=]+/_/g'
}

training_state() {
  if [[ ! -f "${STATUS_FILE}" ]]; then
    echo "unknown"
    return
  fi
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

try:
    payload = json.loads(Path("${STATUS_FILE}").read_text())
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

evaluate_archive() {
  local archive="$1"
  local id eval_dir checkpoint rc master_port
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

  checkpoint="${archive}"
  if [[ -f "${STATE_DIR}/${id}.paths" ]]; then
    checkpoint="$(sed -n '1p' "${STATE_DIR}/${id}.paths")"
  fi

  mkdir -p "${eval_dir}"
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
