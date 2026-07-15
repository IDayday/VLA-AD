#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:-/mnt/data/Bench2Drive}
DEFAULT_CARLA_ROOT=/mnt/project/CARLA_0.9.15
if [[ "$(id -u)" == "0" && -x /mnt/project/CARLA_0.9.15_nonroot/CarlaUE4.sh ]]; then
  DEFAULT_CARLA_ROOT=/mnt/project/CARLA_0.9.15_nonroot
fi
CARLA_ROOT=${CARLA_ROOT:-${DEFAULT_CARLA_ROOT}}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
B2D_ENV=${B2D_ENV:-b2d_eval}
NAVSIM_PYTHON=${NAVSIM_PYTHON:-/root/miniconda3/envs/navsim/bin/python}
TASK_NUM=${TASK_NUM:-16}
GPU_COUNT=${GPU_COUNT:-}
if [[ -z "${GPU_COUNT}" ]]; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
if (( GPU_COUNT <= 0 )); then
  echo "No GPUs detected" >&2
  exit 2
fi
if (( TASK_NUM <= 0 )); then
  echo "TASK_NUM must be positive" >&2
  exit 2
fi

DEFAULT_TASK_LIST=""
DEFAULT_GPU_RANK_LIST=""
for ((i=0; i<TASK_NUM; i++)); do
  DEFAULT_TASK_LIST+="${DEFAULT_TASK_LIST:+ }${i}"
  # Round-robin launch order separates CARLA cold starts that share a GPU.
  DEFAULT_GPU_RANK_LIST+="${DEFAULT_GPU_RANK_LIST:+ }$((i % GPU_COUNT))"
done
TASK_LIST=${TASK_LIST:-${DEFAULT_TASK_LIST}}
GPU_RANK_LIST=${GPU_RANK_LIST:-${DEFAULT_GPU_RANK_LIST}}
SERVER_GPU_RANK_LIST=${SERVER_GPU_RANK_LIST:-${GPU_RANK_LIST}}
RUN_ID=${RUN_ID:-closest_public_epoch200_${TASK_NUM}shard_$(date -u +%Y%m%dT%H%M%SZ)}

OUT=${OUT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/${RUN_ID}}
PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT:?Set PLANNER_CHECKPOINT to the formal Stage2 epoch-200 checkpoint}
VLM_PATH=${VLM_PATH:?Set VLM_PATH to the Stage1 checkpoint used to build the Stage2 cache}
TEAM_CONFIG=${TEAM_CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_closed_loop.closest_public.yaml}
SERVER_CONFIG=${SERVER_CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_stage2_closest_public_2b.yaml}

if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
  echo "Planner checkpoint not found: ${PLANNER_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${VLM_PATH}" ]]; then
  echo "VLM checkpoint directory not found: ${VLM_PATH}" >&2
  exit 2
fi
"${NAVSIM_PYTHON}" - "${PLANNER_CHECKPOINT}" "${VLM_PATH}" <<'PY'
import json
import sys
from pathlib import Path
import torch

checkpoint = Path(sys.argv[1])
vlm_path = Path(sys.argv[2]).resolve()
try:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(checkpoint, map_location="cpu")
config = payload.get("config", {}) if isinstance(payload, dict) else {}
metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
expected_contract = "recogdrive_b2d_closest_public_multiview_10hz_6x0p5s_v1"
if config.get("contract_id") != expected_contract or int(config.get("action_horizon", -1)) != 6:
    raise SystemExit(f"Wrong Stage2 checkpoint contract: {config.get('contract_id')!r}, horizon={config.get('action_horizon')!r}")
if int(metrics.get("completed_global_epochs", -1)) < 200:
    raise SystemExit(f"Checkpoint is not the completed epoch-200 artifact: {metrics}")
cache_summary_path = checkpoint.parent / "cache_summary.json"
if not cache_summary_path.is_file():
    raise SystemExit(f"Missing Stage2 cache provenance: {cache_summary_path}")
cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
if Path(cache_summary.get("source_vlm_path", "")).resolve() != vlm_path:
    raise SystemExit(
        f"Stage2/VLM provenance mismatch: cache={cache_summary.get('source_vlm_path')!r}, requested={vlm_path}"
    )
PY

mkdir -p "${OUT}"
python "${VLA_AD_ROOT}/scripts/bench2drive/check_recogdrive_b2d_reproduction_gate.py" \
  --target evaluation \
  --report "${OUT}/evaluation_gate.json"
BASE_ROUTES=${BASE_ROUTES:-${OUT}/routes/bench2drive220}
mkdir -p "$(dirname "${BASE_ROUTES}")"
cp "${BENCH2DRIVE_ROOT}/leaderboard/data/bench2drive220.xml" "${BASE_ROUTES}.xml"
DEFAULT_SPLIT_COST_JSON=${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/stage2_scratch_full1000_epoch200_220route_20260711T000033Z/route_json/merged.json
SPLIT_COST_JSON=${SPLIT_COST_JSON:-}
if [[ -z "${SPLIT_COST_JSON}" && -f "${DEFAULT_SPLIT_COST_JSON}" ]]; then
  SPLIT_COST_JSON=${DEFAULT_SPLIT_COST_JSON}
fi
if [[ -n "${SPLIT_COST_JSON}" ]]; then
  if [[ ! -f "${SPLIT_COST_JSON}" ]]; then
    echo "Split-cost reference not found: ${SPLIT_COST_JSON}" >&2
    exit 2
  fi
  cp "${SPLIT_COST_JSON}" "${OUT}/split_cost_reference.json"
  SPLIT_COST_JSON=${OUT}/split_cost_reference.json
fi
GIT_COMMIT=$(cd "${VLA_AD_ROOT}" && git rev-parse HEAD)
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=${GIT_COMMIT}"
  echo "vla_ad_root=${VLA_AD_ROOT}"
  echo "bench2drive_root=${BENCH2DRIVE_ROOT}"
  echo "carla_root=${CARLA_ROOT}"
  echo "out=${OUT}"
  echo "planner_checkpoint=${PLANNER_CHECKPOINT}"
  echo "vlm_path=${VLM_PATH}"
  echo "team_config=${TEAM_CONFIG}"
  echo "server_config=${SERVER_CONFIG}"
  echo "base_routes=${BASE_ROUTES}"
  echo "task_num=${TASK_NUM}"
  echo "task_list=${TASK_LIST}"
  echo "gpu_rank_list=${GPU_RANK_LIST}"
  echo "server_gpu_rank_list=${SERVER_GPU_RANK_LIST}"
  echo "split_strategy=${SPLIT_STRATEGY:-balanced}"
  echo "split_cost_json=${SPLIT_COST_JSON}"
  echo "omp_num_threads=${OMP_NUM_THREADS:-4}"
} > "${OUT}/launch_env.txt"

export BENCH2DRIVE_ROOT CARLA_ROOT VLA_AD_ROOT
export BASE_ROUTES
export TASK_NUM TASK_LIST GPU_RANK_LIST SERVER_GPU_RANK_LIST
export START_INFERENCE_SERVERS=${START_INFERENCE_SERVERS:-1}
export KEEP_INFERENCE_SERVERS=${KEEP_INFERENCE_SERVERS:-0}
export BASE_PORT=${BASE_PORT:-31000}
export BASE_TM_PORT=${BASE_TM_PORT:-51000}
export BASE_SERVER_PORT=${BASE_SERVER_PORT:-18765}
export SERVER_STARTUP_SECONDS=${SERVER_STARTUP_SECONDS:-300}
export SERVER_HEALTH_POLL_SECONDS=${SERVER_HEALTH_POLL_SECONDS:-2}
export SERVER_PROFILE_EVERY=${SERVER_PROFILE_EVERY:-200}
export WORKER_START_DELAY=${WORKER_START_DELAY:-3}
export WORKER_STALL_TIMEOUT=${WORKER_STALL_TIMEOUT:-240}
export FORCE_SPLIT=${FORCE_SPLIT:-1}
export SPLIT_STRATEGY=${SPLIT_STRATEGY:-balanced}
export SPLIT_COST_JSON
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-${OMP_NUM_THREADS}}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TEAM_CONFIG
export SERVER_CONFIG
export PLANNER_CHECKPOINT
export VLM_PATH
export SAVE_PATH=${SAVE_PATH:-${OUT}/metrics}
export RESULT_JSON_DIR=${RESULT_JSON_DIR:-${OUT}/route_json}
export WORKER_LOG_DIR=${WORKER_LOG_DIR:-${OUT}/worker_logs}
export SERVER_LOG_DIR=${SERVER_LOG_DIR:-${OUT}/server_logs}
export GENERATED_TEAM_CONFIG_DIR=${GENERATED_TEAM_CONFIG_DIR:-${OUT}/generated_team_configs}
export BASE_CHECKPOINT_ENDPOINT=${BASE_CHECKPOINT_ENDPOINT:-final_epoch200_b2d220}

cd "${VLA_AD_ROOT}"
exec "${CONDA_BIN}" run --no-capture-output -n "${B2D_ENV}" \
  bash scripts/bench2drive/run_recogdrive_closed_loop_multi.sh
