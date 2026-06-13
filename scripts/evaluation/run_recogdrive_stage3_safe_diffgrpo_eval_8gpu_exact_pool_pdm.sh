#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/navsim/bin/torchrun}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_rl_2b_exact_pool_pdm_navtest_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the Stage 3 checkpoint file or directory to evaluate.}"
VLM_PATH="${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${RECOGDRIVE_NAVTEST_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_navtest_full_v1}}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/test_navsim_logs/test}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${NAVSIM_DATA_ROOT}/test_sensor_blobs/test}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63691}}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-16}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE:-$((ASYNC_PDM_WORKERS * 2))}"
ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY:-100}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUT_ROOT}/hydra}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RECOGDRIVE_ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}"
export RECOGDRIVE_ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}"
export RECOGDRIVE_ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD}"
export RECOGDRIVE_ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE}"
export RECOGDRIVE_ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY}"
export RECOGDRIVE_FAST_METRIC_CACHE_PATH="${FAST_METRIC_CACHE_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -e "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "VLM path does not exist: ${VLM_PATH}" >&2
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
if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
  echo "SENSOR_BLOBS_PATH does not exist: ${SENSOR_BLOBS_PATH}" >&2
  exit 2
fi
if [[ -n "${FAST_METRIC_CACHE_DIR}" && ! -d "${FAST_METRIC_CACHE_DIR}" ]]; then
  echo "FAST_METRIC_CACHE_DIR does not exist: ${FAST_METRIC_CACHE_DIR}" >&2
  exit 2
fi
if [[ "${ASYNC_PDM_BACKEND}" != "thread" && "${ASYNC_PDM_BACKEND}" != "process" ]]; then
  echo "ASYNC_PDM_BACKEND must be thread or process, got: ${ASYNC_PDM_BACKEND}" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<PY
import sys
import torch
from pathlib import Path
sys.path.insert(0, "${REPO_ROOT}")
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.metric_caching.fast_metric_cache_loader import FastMetricCacheLoader

loader = MetricCacheLoader(Path("${METRIC_CACHE_DIR}"))
count = len(loader)
if count <= 0:
    raise SystemExit("Metric cache is empty: ${METRIC_CACHE_DIR}")
print(f"metric_cache_count={count}")
fast_cache_dir = "${FAST_METRIC_CACHE_DIR}"
if fast_cache_dir:
    fast_loader = FastMetricCacheLoader(Path(fast_cache_dir))
    missing = set(loader.tokens) - set(fast_loader.tokens)
    if missing:
        sample = sorted(missing)[:5]
        raise SystemExit(f"FAST_METRIC_CACHE_DIR is missing {len(missing)} original metric-cache tokens; sample={sample}")
    print(f"fast_metric_cache_count={len(fast_loader)}")

required = int("${GPUS_PER_NODE}")
cuda_count = torch.cuda.device_count()
print(f"cuda_device_count={cuda_count}")
if cuda_count < required:
    raise SystemExit(f"Need at least {required} visible CUDA devices, got {cuda_count}")
PY

mkdir -p "${OUT_ROOT}"

CMD=(
  "${TORCHRUN_BIN}"
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${REPO_ROOT}/navsim/planning/script/run_pdm_score_recogdrive_async_pdm_exact_pool.py"
  "train_test_split=navtest"
  "agent=recogdrive_agent"
  "agent.checkpoint_path=${CHECKPOINT}"
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
  "experiment_name=stage3_safe_diffgrpo_eval_exact_pool_pdm"
  "+async_pdm_workers=${ASYNC_PDM_WORKERS}"
  "+async_pdm_backend=${ASYNC_PDM_BACKEND}"
  "+async_pdm_process_start_method=${ASYNC_PDM_PROCESS_START_METHOD}"
  "+async_pdm_queue_size=${ASYNC_PDM_QUEUE_SIZE}"
  "+async_pdm_progress_every=${ASYNC_PDM_PROGRESS_EVERY}"
)
if [[ -n "${FAST_METRIC_CACHE_DIR}" ]]; then
  CMD+=("+fast_metric_cache_path=${FAST_METRIC_CACHE_DIR}")
fi

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "vlm_path=${VLM_PATH}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "navsim_log_path=${NAVSIM_LOG_PATH}"
  echo "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  echo "gpus_per_node=${GPUS_PER_NODE}"
  echo "async_pdm_workers=${ASYNC_PDM_WORKERS}"
  echo "async_pdm_backend=${ASYNC_PDM_BACKEND}"
  echo "async_pdm_process_start_method=${ASYNC_PDM_PROCESS_START_METHOD}"
  echo "async_pdm_queue_size=${ASYNC_PDM_QUEUE_SIZE}"
  echo "fast_metric_cache_dir=${FAST_METRIC_CACHE_DIR}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${OUT_ROOT}/resolved_eval_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_eval_command.txt"
  exit 0
fi

exec "${CMD[@]}"
