#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-metric_cache_navtrain_full_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval}"
FORCE_FEATURE_COMPUTATION="${FORCE_FEATURE_COMPUTATION:-false}"
RAY_THREADS_PER_NODE="${RAY_THREADS_PER_NODE:-null}"
MIN_FREE_GB="${MIN_FREE_GB:-250}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${ARTIFACT_ROOT}/cache}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${NUPLAN_MAPS_ROOT}" ]]; then
  echo "NUPLAN_MAPS_ROOT does not exist: ${NUPLAN_MAPS_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
  echo "NAVSIM_LOG_PATH does not exist: ${NAVSIM_LOG_PATH}" >&2
  exit 2
fi
if [[ -z "${MAX_SCENES:-}" && "${ALLOW_LOW_DISK}" != "1" ]]; then
  AVAILABLE_GB="$(df -BG "${ARTIFACT_ROOT}" | awk 'NR==2 {gsub(/G/, "", $4); print $4}')"
  if [[ -z "${AVAILABLE_GB}" || "${AVAILABLE_GB}" -lt "${MIN_FREE_GB}" ]]; then
    echo "Refusing full navtrain metric caching with only ${AVAILABLE_GB:-unknown}G free under ${ARTIFACT_ROOT}." >&2
    echo "Free at least ${MIN_FREE_GB}G, set MAX_SCENES for smoke, or set ALLOW_LOW_DISK=1 to override." >&2
    exit 2
  fi
fi

mkdir -p "${RUN_ROOT}" "${METRIC_CACHE_DIR}"

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/navsim/planning/script/run_metric_caching.py"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache.cache_path=${METRIC_CACHE_DIR}"
  "cache.force_feature_computation=${FORCE_FEATURE_COMPUTATION}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "worker=ray_distributed_no_torch"
  "worker.threads_per_node=${RAY_THREADS_PER_NODE}"
)

if [[ -n "${MAX_SCENES:-}" ]]; then
  CMD+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

{
  echo "repo_root=${REPO_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "navsim_log_path=${NAVSIM_LOG_PATH}"
  echo "train_test_split=${TRAIN_TEST_SPLIT}"
  echo "force_feature_computation=${FORCE_FEATURE_COMPUTATION}"
  echo "max_scenes=${MAX_SCENES:-full}"
  echo "min_free_gb=${MIN_FREE_GB}"
  echo "allow_low_disk=${ALLOW_LOW_DISK}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${RUN_ROOT}/resolved_command.txt"

exec "${CMD[@]}"
