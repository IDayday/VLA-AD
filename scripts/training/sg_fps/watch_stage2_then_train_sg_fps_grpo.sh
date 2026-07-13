#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
STAGE2_OUT=${STAGE2_OUT:?set STAGE2_OUT to the Stage2 DPSI output directory}
SUPPORT_ARCHIVE_PATH=${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH to the SG-FPS v3 support archive}
STAGE2_PID_FILE=${STAGE2_PID_FILE:-}
STAGE3_OUT=${STAGE3_OUT:-${STAGE2_OUT%/}/../stage3_feasible_pareto_grpo}
POLL_SECONDS=${POLL_SECONDS:-300}
FS_NORM_STATS_PATH=${FS_NORM_STATS_PATH:-}
USE_FS_NORM=${USE_FS_NORM:-true}
FS_NORM_USE_ROBUST=${FS_NORM_USE_ROBUST:-true}
RUN_STAGE3=${RUN_STAGE3:-1}

mkdir -p "${STAGE3_OUT}/logs" "${STAGE3_OUT}/pids"

{
  echo "repo_root=${REPO_ROOT}"
  echo "stage2_out=${STAGE2_OUT}"
  echo "stage2_pid_file=${STAGE2_PID_FILE}"
  echo "support_archive_path=${SUPPORT_ARCHIVE_PATH}"
  echo "stage3_out=${STAGE3_OUT}"
  echo "fs_norm_stats_path=${FS_NORM_STATS_PATH}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STAGE3_OUT}/commands.log"

_stage2_alive() {
  if [[ -z "${STAGE2_PID_FILE}" || ! -f "${STAGE2_PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid=$(cat "${STAGE2_PID_FILE}" 2>/dev/null || true)
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  return 1
}

while _stage2_alive; do
  latest_ckpt=$(find "${STAGE2_OUT}" -name '*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
  echo "stage2_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) alive=true latest_ckpt=${latest_ckpt:-none}"
  sleep "${POLL_SECONDS}"
done

echo "stage2_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) alive=false"
latest_ckpt=$(find "${STAGE2_OUT}" -name '*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
if [[ -z "${latest_ckpt}" || ! -f "${latest_ckpt}" ]]; then
  echo "No Stage2 checkpoint found under ${STAGE2_OUT}" >&2
  exit 1
fi
echo "selected_stage2_checkpoint=${latest_ckpt}" | tee -a "${STAGE3_OUT}/commands.log"

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' env \
    SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}" \
    ELITE_BUFFER_DIR="${SUPPORT_ARCHIVE_PATH}" \
    IL_CHECKPOINT="${latest_ckpt}" \
    OUT_ROOT="${STAGE3_OUT}" \
    USE_FS_NORM="${USE_FS_NORM}" \
    FS_NORM_STATS_PATH="${FS_NORM_STATS_PATH}" \
    FS_NORM_USE_ROBUST="${FS_NORM_USE_ROBUST}" \
    RUN_STAGE3="${RUN_STAGE3}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_train_sg_fps_grpo.sh"
  printf '\n'
} >> "${STAGE3_OUT}/commands.log"

setsid env \
  SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}" \
  ELITE_BUFFER_DIR="${SUPPORT_ARCHIVE_PATH}" \
  IL_CHECKPOINT="${latest_ckpt}" \
  OUT_ROOT="${STAGE3_OUT}" \
  USE_FS_NORM="${USE_FS_NORM}" \
  FS_NORM_STATS_PATH="${FS_NORM_STATS_PATH}" \
  FS_NORM_USE_ROBUST="${FS_NORM_USE_ROBUST}" \
  RUN_STAGE3="${RUN_STAGE3}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${REPO_ROOT}/scripts/training/sg_fps/run_train_sg_fps_grpo.sh" \
  > "${STAGE3_OUT}/logs/stage3_feasible_pareto_grpo.log" 2>&1 < /dev/null &
echo $! > "${STAGE3_OUT}/pids/stage3_feasible_pareto_grpo.pid"
echo "stage3_pid=$(cat "${STAGE3_OUT}/pids/stage3_feasible_pareto_grpo.pid")"
