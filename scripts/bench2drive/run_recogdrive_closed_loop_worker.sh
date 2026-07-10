#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT}
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT}
TASK_ID=${TASK_ID:?Set TASK_ID}
GPU_RANK=${GPU_RANK:?Set GPU_RANK}
PORT=${PORT:?Set PORT}
TM_PORT=${TM_PORT:?Set TM_PORT}
TEAM_CONFIG=${TEAM_CONFIG:?Set TEAM_CONFIG}
CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:?Set CHECKPOINT_ENDPOINT}

ALGO=${ALGO:-recogdrive}
PLANNER_TYPE=${PLANNER_TYPE:-only_traj}
BASE_ROUTES=${BASE_ROUTES:-leaderboard/data/bench2drive220}
TEAM_AGENT=${TEAM_AGENT:-${VLA_AD_ROOT}/bench2drive_eval/team_code/recogdrive_b2d_agent.py}
TIMEOUT=${TIMEOUT:-600}

export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/leaderboard/team_code:${BENCH2DRIVE_ROOT}/scenario_runner:${VLA_AD_ROOT}/bench2drive_eval/team_code:${VLA_AD_ROOT}"
export SCENARIO_RUNNER_ROOT="${BENCH2DRIVE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BENCH2DRIVE_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export PLANNER_TYPE

cd "${BENCH2DRIVE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_RANK}" python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${BASE_ROUTES}_${TASK_ID}_${ALGO}_${PLANNER_TYPE}.xml" \
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
