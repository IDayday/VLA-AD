#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:-/mnt/project/Bench2Drive}
CARLA_ROOT=${CARLA_ROOT:-/mnt/project/CARLA_0.9.15}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
B2D_ENV=${B2D_ENV:-b2d_eval}
RUN_ID=${RUN_ID:-best_step17136_8gpu_$(date -u +%Y%m%dT%H%M%SZ)}

OUT=${OUT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/${RUN_ID}}
PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_il_official_2b_20260709T210221Z/best_step17136_for_bench2drive220_20260710T0044.ckpt}
TEAM_CONFIG=${TEAM_CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_closed_loop.remote.yaml}

mkdir -p "${OUT}"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "vla_ad_root=${VLA_AD_ROOT}"
  echo "bench2drive_root=${BENCH2DRIVE_ROOT}"
  echo "carla_root=${CARLA_ROOT}"
  echo "out=${OUT}"
  echo "planner_checkpoint=${PLANNER_CHECKPOINT}"
  echo "team_config=${TEAM_CONFIG}"
  echo "gpu_rank_list=${GPU_RANK_LIST:-0 1 2 3 4 5 6 7}"
} > "${OUT}/launch_env.txt"

export BENCH2DRIVE_ROOT CARLA_ROOT VLA_AD_ROOT
export TASK_NUM=${TASK_NUM:-8}
export GPU_RANK_LIST=${GPU_RANK_LIST:-"0 1 2 3 4 5 6 7"}
export TASK_LIST=${TASK_LIST:-"0 1 2 3 4 5 6 7"}
export SERVER_GPU_RANK_LIST=${SERVER_GPU_RANK_LIST:-"${GPU_RANK_LIST}"}
export START_INFERENCE_SERVERS=${START_INFERENCE_SERVERS:-1}
export KEEP_INFERENCE_SERVERS=${KEEP_INFERENCE_SERVERS:-0}
export BASE_PORT=${BASE_PORT:-31000}
export BASE_TM_PORT=${BASE_TM_PORT:-51000}
export BASE_SERVER_PORT=${BASE_SERVER_PORT:-18765}
export SERVER_STARTUP_SECONDS=${SERVER_STARTUP_SECONDS:-90}
export SERVER_PROFILE_EVERY=${SERVER_PROFILE_EVERY:-200}
export FORCE_SPLIT=${FORCE_SPLIT:-1}
export TEAM_CONFIG
export PLANNER_CHECKPOINT
export SAVE_PATH=${SAVE_PATH:-${OUT}/metrics}
export RESULT_JSON_DIR=${RESULT_JSON_DIR:-${OUT}/route_json}
export WORKER_LOG_DIR=${WORKER_LOG_DIR:-${OUT}/worker_logs}
export SERVER_LOG_DIR=${SERVER_LOG_DIR:-${OUT}/server_logs}
export GENERATED_TEAM_CONFIG_DIR=${GENERATED_TEAM_CONFIG_DIR:-${OUT}/generated_team_configs}
export BASE_CHECKPOINT_ENDPOINT=${BASE_CHECKPOINT_ENDPOINT:-best_step17136_b2d220}

cd "${VLA_AD_ROOT}"
exec "${CONDA_BIN}" run --no-capture-output -n "${B2D_ENV}" \
  bash scripts/bench2drive/run_recogdrive_closed_loop_multi.sh
