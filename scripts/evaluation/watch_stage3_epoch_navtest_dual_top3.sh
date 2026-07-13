#!/usr/bin/env bash
set -Eeuo pipefail

: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the Lightning epoch checkpoint directory}"
: "${OUT_ROOT:?Set OUT_ROOT to the dual-protocol evaluation directory}"
: "${V1_CONFIG:?Set V1_CONFIG to the exact planner evaluation YAML used by the checkpoint}"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
V2_PYTHON_BIN="${V2_PYTHON_BIN:-/mnt/code/miniconda/envs/navsimv2-recogdrive/bin/python}"
V2_NAVSIM_ROOT="${V2_NAVSIM_ROOT:-/mnt/code/liushiqi/navsim}"
V2_METRIC_CACHE_PATH="${V2_METRIC_CACHE_PATH:-/mnt/code/dataset/navsim/metric_cache_v2/navtest_full_2026-03-07_15-40-49}"
V1_CHUNK_CACHE_ROOT="${V1_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
V1_METRIC_CACHE_DIR="${V1_METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-32}"
V2_WORKERS="${V2_WORKERS:-32}"
PRECISION="${PRECISION:-fp32}"
TOP_K="${TOP_K:-3}"
TOPK_BACKUP_MODE="${TOPK_BACKUP_MODE:-hardlink_or_copy}"
EPOCH_MIN="${EPOCH_MIN:-0}"
EPOCH_MAX="${EPOCH_MAX:-9}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
RETRY_SECONDS="${RETRY_SECONDS:-300}"

# Optional lifecycle control for a disposable GPU pressure process on the
# evaluation host. Defaults are inert so existing watchers are unchanged.
GPU_PRESSURE_PROCESS_PATTERN="${GPU_PRESSURE_PROCESS_PATTERN:-}"
GPU_PRESSURE_PYTHON_BIN="${GPU_PRESSURE_PYTHON_BIN:-${PYTHON_BIN}}"
GPU_PRESSURE_SCRIPT="${GPU_PRESSURE_SCRIPT:-}"
GPU_PRESSURE_RESTART_CWD="${GPU_PRESSURE_RESTART_CWD:-${PROJECT_ROOT}}"
GPU_PRESSURE_MEMORY_GB="${GPU_PRESSURE_MEMORY_GB:-}"
GPU_PRESSURE_STOP_TIMEOUT_SECONDS="${GPU_PRESSURE_STOP_TIMEOUT_SECONDS:-30}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/state"
WATCH_LOG="${OUT_ROOT}/logs/watcher.log"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
exec 9>"${OUT_ROOT}/state/watcher.lock"
if ! flock -n 9; then
  echo "A dual NAVTEST watcher already holds ${OUT_ROOT}/state/watcher.lock" >&2
  exit 3
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCH_LOG}"
}

gpu_pressure_enabled() {
  [[ -n "${GPU_PRESSURE_PROCESS_PATTERN}" ]]
}

gpu_pressure_candidate_leaders() {
  {
    pgrep -f -- "${GPU_PRESSURE_PROCESS_PATTERN}" 2>/dev/null || true
    [[ -f "${OUT_ROOT}/state/gpu_pressure.pid" ]] && cat "${OUT_ROOT}/state/gpu_pressure.pid"
    [[ -f "${OUT_ROOT}/state/gpu_pressure.stopped_pids" ]] && cat "${OUT_ROOT}/state/gpu_pressure.stopped_pids"
  } | awk '/^[0-9]+$/' | sort -nu
}

gpu_pressure_running() {
  local pid
  gpu_pressure_enabled || return 1
  while IFS= read -r pid; do
    if kill -0 "${pid}" 2>/dev/null || kill -0 -- "-${pid}" 2>/dev/null; then
      return 0
    fi
  done < <(gpu_pressure_candidate_leaders)
  return 1
}

stop_gpu_pressure() {
  local any_alive deadline pid
  local -a pids=()
  gpu_pressure_enabled || return 0
  mapfile -t pids < <(gpu_pressure_candidate_leaders)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi

  printf '%s\n' "${pids[@]}" > "${OUT_ROOT}/state/gpu_pressure.stopped_pids"
  log "stopping GPU pressure before evaluation: leaders=${pids[*]} pattern=${GPU_PRESSURE_PROCESS_PATTERN}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for pid in "${pids[@]}"; do
    # The pressure launcher uses setsid; workers share the leader's process
    # group and can survive if only the Python parent receives SIGTERM.
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  deadline=$(( $(date +%s) + GPU_PRESSURE_STOP_TIMEOUT_SECONDS ))
  while (( $(date +%s) < deadline )); do
    any_alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null || kill -0 -- "-${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    if (( any_alive == 0 )); then
      touch "${OUT_ROOT}/state/gpu_pressure.restart_required"
      return 0
    fi
    sleep 1
  done
  for pid in "${pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
    kill -KILL -- "-${pid}" 2>/dev/null || true
  done
  touch "${OUT_ROOT}/state/gpu_pressure.restart_required"
}

restart_gpu_pressure() {
  local pressure_pid
  [[ -f "${OUT_ROOT}/state/gpu_pressure.restart_required" ]] || return 0
  if gpu_pressure_running; then
    log "GPU pressure already running after evaluation; skipping duplicate restart"
    rm -f "${OUT_ROOT}/state/gpu_pressure.restart_required"
    return 0
  fi
  if [[ -z "${GPU_PRESSURE_SCRIPT}" || -z "${GPU_PRESSURE_MEMORY_GB}" ]]; then
    log "GPU pressure restart requested but script/memory configuration is incomplete"
    return 1
  fi

  mkdir -p "${OUT_ROOT}/logs"
  (
    cd "${GPU_PRESSURE_RESTART_CWD}"
    setsid "${GPU_PRESSURE_PYTHON_BIN}" "${GPU_PRESSURE_SCRIPT}" \
      --memory-gb "${GPU_PRESSURE_MEMORY_GB}" \
      >> "${OUT_ROOT}/logs/gpu_pressure.log" 2>&1 < /dev/null &
    pressure_pid=$!
    printf '%s\n' "${pressure_pid}" > "${OUT_ROOT}/state/gpu_pressure.pid"
  )
  rm -f "${OUT_ROOT}/state/gpu_pressure.restart_required"
  log "restarted GPU pressure after evaluation: pid=$(cat "${OUT_ROOT}/state/gpu_pressure.pid")"
}

trap 'restart_gpu_pressure || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_path() {
  local path="$1"
  [[ -e "${path}" ]] || { log "required path missing: ${path}"; exit 2; }
}

checkpoint_epoch_step() {
  local name
  name="$(basename "$1")"
  if [[ "${name}" =~ ^epoch=([0-9]+)-step=([0-9]+)\.ckpt$ ]]; then
    printf '%d %d\n' "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[2]}))"
    return 0
  fi
  if [[ "${name}" =~ ^epoch_([0-9]+)\.ckpt$ ]]; then
    printf '%d -1\n' "$((10#${BASH_REMATCH[1]}))"
    return 0
  fi
  return 1
}

checkpoint_is_stable() {
  local checkpoint="$1" age size_before size_after
  age=$(( $(date +%s) - $(stat -c %Y "${checkpoint}") ))
  (( age >= STABLE_SECONDS )) || return 1
  size_before="$(stat -c %s "${checkpoint}")"
  sleep 2
  size_after="$(stat -c %s "${checkpoint}")"
  [[ "${size_before}" == "${size_after}" && "${size_before}" -gt 0 ]]
}

write_epoch_metrics() {
  local epoch="$1" step="$2" checkpoint="$3" epoch_dir="$4"
  "${PYTHON_BIN}" - "${epoch}" "${step}" "${checkpoint}" "${epoch_dir}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

epoch, step = int(sys.argv[1]), int(sys.argv[2])
checkpoint, epoch_dir = Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve()
payload = {
    "epoch": epoch,
    "step": step,
    "checkpoint": str(checkpoint),
    "PDMS": None,
    "EPDMS": None,
    "v1_num_valid": None,
    "v2_num_valid": None,
    "v1_eval_dir": "",
    "v2_eval_dir": "",
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
v1_paths = sorted((epoch_dir / "navsim_v1").glob("*/aggregate_metrics.json"))
if v1_paths:
    metrics = json.loads(v1_paths[-1].read_text())
    payload.update(
        PDMS=metrics.get("PDMS"),
        v1_num_valid=metrics.get("num_pdm_valid"),
        v1_eval_dir=str(v1_paths[-1].parent),
    )
v2_path = epoch_dir / "navsim_v2" / "summary.json"
if v2_path.is_file():
    metrics = json.loads(v2_path.read_text())
    payload.update(
        EPDMS=metrics.get("EPDMS", metrics.get("score_mean")),
        v2_num_valid=metrics.get("successful"),
        v2_eval_dir=str(v2_path.parent),
    )
temporary = epoch_dir / "epoch_metrics.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(epoch_dir / "epoch_metrics.json")
PY
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluation/update_stage3_navtest_dual_topk.py" \
    --eval-root "${OUT_ROOT}" \
    --top-k "${TOP_K}" \
    --backup-mode "${TOPK_BACKUP_MODE}"
}

run_v1() {
  local checkpoint="$1" epoch_dir="$2"
  local v1_root="${epoch_dir}/navsim_v1"
  mkdir -p "${v1_root}"
  printf '[%s] v1 checkpoint=%q out=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${checkpoint}" "${v1_root}" >> "${COMMANDS_LOG}"
  OUT_ROOT="${v1_root}" \
    CHECKPOINTS="${checkpoint}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    GPUS_CSV="${GPU_LIST}" \
    PRECISION="${PRECISION}" \
    SKIP_COMPLETED=1 \
    CONFIG="${V1_CONFIG}" \
    CHUNK_CACHE_ROOT="${V1_CHUNK_CACHE_ROOT}" \
    METRIC_CACHE_DIR="${V1_METRIC_CACHE_DIR}" \
    bash "${PROJECT_ROOT}/scripts/evaluation/run_recogdrive_stage2_navtest_sharded_ckpts.sh" \
    >> "${epoch_dir}/v1_eval.log" 2>&1
  touch "${epoch_dir}/v1.done"
}

run_v2() {
  local epoch_dir="$1" v1_eval_dir
  v1_eval_dir="$(find "${epoch_dir}/navsim_v1" -mindepth 2 -maxdepth 2 -type f -name aggregate_metrics.json -printf '%h\n' | head -n 1)"
  [[ -n "${v1_eval_dir}" ]] || { log "cannot locate v1 predictions under ${epoch_dir}"; return 1; }
  mkdir -p "${epoch_dir}/navsim_v2"
  printf '[%s] v2 predictions=%q out=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${v1_eval_dir}" "${epoch_dir}/navsim_v2" >> "${COMMANDS_LOG}"
  CUDA_VISIBLE_DEVICES="" "${V2_PYTHON_BIN}" "${PROJECT_ROOT}/scripts/evaluation/score_recogdrive_predictions_navsim_v2.py" \
    --predictions-dir "${v1_eval_dir}" \
    --navsim-root "${V2_NAVSIM_ROOT}" \
    --metric-cache-path "${V2_METRIC_CACHE_PATH}" \
    --output-dir "${epoch_dir}/navsim_v2" \
    --workers "${V2_WORKERS}" \
    >> "${epoch_dir}/v2_eval.log" 2>&1
  touch "${epoch_dir}/v2.done"
}

for path in \
  "${PYTHON_BIN}" \
  "${V2_PYTHON_BIN}" \
  "${V2_NAVSIM_ROOT}/navsim/evaluate/pdm_score.py" \
  "${V2_METRIC_CACHE_PATH}" \
  "${V1_CONFIG}" \
  "${V1_CHUNK_CACHE_ROOT}" \
  "${V1_METRIC_CACHE_DIR}"; do
  require_path "${path}"
done

log "watcher started checkpoint_root=${CHECKPOINT_ROOT} out_root=${OUT_ROOT} epochs=${EPOCH_MIN}-${EPOCH_MAX} protocols=navsim_v1_pdms,navsim_v2_navtest_epdms top_k=${TOP_K}"

while true; do
  completed=0
  found=0
  while IFS= read -r checkpoint; do
    [[ -n "${checkpoint}" ]] || continue
    parsed="$(checkpoint_epoch_step "${checkpoint}" || true)"
    [[ -n "${parsed}" ]] || continue
    read -r epoch step <<< "${parsed}"
    (( epoch >= EPOCH_MIN && epoch <= EPOCH_MAX )) || continue
    found=$((found + 1))
    epoch_dir="${OUT_ROOT}/epoch_$(printf '%03d' "${epoch}")"
    mkdir -p "${epoch_dir}"
    printf '%s\n' "${checkpoint}" > "${epoch_dir}/checkpoint.path"
    if [[ -f "${epoch_dir}/v1.done" && -f "${epoch_dir}/v2.done" ]]; then
      completed=$((completed + 1))
      continue
    fi
    if ! checkpoint_is_stable "${checkpoint}"; then
      log "epoch=${epoch} checkpoint is not stable yet: ${checkpoint}"
      continue
    fi

    stop_gpu_pressure

    if [[ ! -f "${epoch_dir}/v1.done" ]]; then
      log "epoch=${epoch} step=${step} starting NAVSIM v1 PDMS"
      if ! run_v1 "${checkpoint}" "${epoch_dir}"; then
        log "epoch=${epoch} NAVSIM v1 PDMS failed; retrying in ${RETRY_SECONDS}s"
        restart_gpu_pressure || true
        sleep "${RETRY_SECONDS}"
        continue
      fi
      write_epoch_metrics "${epoch}" "${step}" "${checkpoint}" "${epoch_dir}"
      log "epoch=${epoch} NAVSIM v1 PDMS complete"
    fi

    if [[ ! -f "${epoch_dir}/v2.done" ]]; then
      log "epoch=${epoch} step=${step} starting NAVSIM v2 navtest EPDMS from identical trajectories"
      if ! run_v2 "${epoch_dir}"; then
        log "epoch=${epoch} NAVSIM v2 EPDMS failed; retrying in ${RETRY_SECONDS}s"
        restart_gpu_pressure || true
        sleep "${RETRY_SECONDS}"
        continue
      fi
      write_epoch_metrics "${epoch}" "${step}" "${checkpoint}" "${epoch_dir}"
      log "epoch=${epoch} NAVSIM v2 EPDMS complete"
    fi
    restart_gpu_pressure
    completed=$((completed + 1))
  done < <(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)

  if (( completed >= EXPECTED_EPOCHS )); then
    log "all ${completed} expected epochs completed under both protocols; exiting"
    exit 0
  fi
  log "queue status found=${found} fully_completed=${completed}/${EXPECTED_EPOCHS}; sleeping ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
