#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT to the official Bench2Drive repo path}
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT to CARLA_0.9.15 root}

GPU_RANK=${GPU_RANK:-0}
PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}
ROUTES=${ROUTES:-leaderboard/data/drivetransformer_bench2drive_dev10.xml}
CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-eval_recogdrive_b2d_debug.json}
SAVE_PATH=${SAVE_PATH:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/debug}
TEAM_CONFIG=${TEAM_CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_closed_loop.remote.yaml}
TEAM_AGENT=${TEAM_AGENT:-${VLA_AD_ROOT}/bench2drive_eval/team_code/recogdrive_b2d_agent.py}
TIMEOUT=${TIMEOUT:-600}

export VLA_AD_ROOT
export BENCH2DRIVE_ROOT
export CARLA_ROOT
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/leaderboard/team_code:${BENCH2DRIVE_ROOT}/scenario_runner:${VLA_AD_ROOT}/bench2drive_eval/team_code:${VLA_AD_ROOT}"
export SCENARIO_RUNNER_ROOT="${BENCH2DRIVE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BENCH2DRIVE_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export SAVE_PATH

cd "${BENCH2DRIVE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_RANK}" python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --repetitions=1 \
  --track=SENSORS \
  --checkpoint="${CHECKPOINT_ENDPOINT}" \
  --agent="${TEAM_AGENT}" \
  --agent-config="${TEAM_CONFIG}" \
  --debug=0 \
  --resume=True \
  --port="${PORT}" \
  --traffic-manager-port="${TM_PORT}" \
  --gpu-rank="${GPU_RANK}" \
  --timeout="${TIMEOUT}"
