#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT to the official Bench2Drive repo path}
CARLA_ROOT=${CARLA_ROOT:?Set CARLA_ROOT to CARLA_0.9.15 root}

BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}
TASK_NUM=${TASK_NUM:-8}
ALGO=${ALGO:-recogdrive}
PLANNER_TYPE=${PLANNER_TYPE:-only_traj}
BASE_ROUTES=${BASE_ROUTES:-leaderboard/data/bench2drive220}
TEAM_AGENT=${TEAM_AGENT:-${VLA_AD_ROOT}/bench2drive_eval/team_code/recogdrive_b2d_agent.py}
TEAM_CONFIG=${TEAM_CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_closed_loop.remote.yaml}
SAVE_PATH=${SAVE_PATH:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/bench2drive220}
RESULT_JSON_DIR=${RESULT_JSON_DIR:-${BENCH2DRIVE_ROOT}/${ALGO}_b2d_${PLANNER_TYPE}}
WORKER_LOG_DIR=${WORKER_LOG_DIR:-${SAVE_PATH}/worker_logs}
BASE_CHECKPOINT_ENDPOINT=${BASE_CHECKPOINT_ENDPOINT:-eval_recogdrive_b2d}
TIMEOUT=${TIMEOUT:-600}
FORCE_SPLIT=${FORCE_SPLIT:-0}
MAX_WORKER_RESTARTS=${MAX_WORKER_RESTARTS:-20}
WORKER_RETRY_DELAY=${WORKER_RETRY_DELAY:-10}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_CONDA_ENV=${NAVSIM_CONDA_ENV:-navsim}
START_INFERENCE_SERVERS=${START_INFERENCE_SERVERS:-0}
KEEP_INFERENCE_SERVERS=${KEEP_INFERENCE_SERVERS:-0}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
BASE_SERVER_PORT=${BASE_SERVER_PORT:-8765}
SERVER_STARTUP_SECONDS=${SERVER_STARTUP_SECONDS:-90}
SERVER_LOG_DIR=${SERVER_LOG_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/server_logs}
SERVER_PROFILE_EVERY=${SERVER_PROFILE_EVERY:-100}
PRECISION=${PRECISION:-bf16}
PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_vlm_il_train_full_epoch1/best.ckpt}
VLM_PATH=${VLM_PATH:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
GENERATED_TEAM_CONFIG_DIR=${GENERATED_TEAM_CONFIG_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop/generated_team_configs}
USE_PER_WORKER_SERVER_URLS=${USE_PER_WORKER_SERVER_URLS:-${START_INFERENCE_SERVERS}}

IFS=' ' read -r -a GPU_RANK_LIST <<< "${GPU_RANK_LIST:-0 1 2 3 4 5 6 7}"
IFS=' ' read -r -a TASK_LIST <<< "${TASK_LIST:-0 1 2 3 4 5 6 7}"
IFS=' ' read -r -a SERVER_GPU_RANK_LIST <<< "${SERVER_GPU_RANK_LIST:-${GPU_RANK_LIST[*]}}"
if [ -n "${SERVER_PORT_LIST:-}" ]; then
  IFS=' ' read -r -a SERVER_PORT_ARRAY <<< "${SERVER_PORT_LIST}"
else
  SERVER_PORT_ARRAY=()
fi

SERVER_PIDS=()
EVAL_PIDS=()

cleanup() {
  for pid in "${EVAL_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  if [ "${START_INFERENCE_SERVERS}" = "1" ] && [ "${KEEP_INFERENCE_SERVERS}" != "1" ]; then
    for pid in "${SERVER_PIDS[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    for ((i=0; i<${#GPU_RANK_LIST[@]}; i++)); do
      SERVER_PORT=$(server_port_for_worker "${i}")
      pkill -f "scripts/bench2drive/serve_recogdrive_b2d.py --host ${SERVER_HOST} --port ${SERVER_PORT}" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

server_port_for_worker() {
  local index=$1
  if [ "${#SERVER_PORT_ARRAY[@]}" -gt "${index}" ]; then
    echo "${SERVER_PORT_ARRAY[$index]}"
  else
    echo $((BASE_SERVER_PORT + index))
  fi
}

write_worker_config() {
  local source_config=$1
  local target_config=$2
  local server_url=$3
  local task_id=$4
  python - "${source_config}" "${target_config}" "${server_url}" "${task_id}" <<'PY'
from pathlib import Path
import sys
import yaml

source, target, server_url, task_id = sys.argv[1:]
with open(source, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}
data["server_url"] = server_url
data["save_name"] = f"{data.get('save_name', 'recogdrive_b2d')}_task{task_id}"
Path(target).parent.mkdir(parents=True, exist_ok=True)
with open(target, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

check_server_health() {
  local host=$1
  local port=$2
  python - "${host}" "${port}" <<'PY'
import json
import sys
from urllib.request import urlopen

host, port = sys.argv[1:]
with urlopen(f"http://{host}:{port}/health", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
if not payload.get("ok"):
    raise SystemExit(f"unhealthy server response: {payload}")
PY
}

checkpoint_progress_complete() {
  local checkpoint_endpoint=$1
  python - "${checkpoint_endpoint}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    current, total = payload.get("_checkpoint", {}).get("progress", [0, 0])
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
raise SystemExit(0 if int(total) > 0 and int(current) >= int(total) else 1)
PY
}

run_eval_worker() {
  local task_id=$1
  local gpu_rank=$2
  local port=$3
  local tm_port=$4
  local routes=$5
  local checkpoint_endpoint=$6
  local worker_team_config=$7
  local log_path=$8
  local restarts=0
  local rc=0
  local evaluator_pid=""
  local log_offset=1

  cleanup_worker_processes() {
    if [[ -n "${evaluator_pid}" ]] && kill -0 "${evaluator_pid}" 2>/dev/null; then
      kill "${evaluator_pid}" 2>/dev/null || true
      for _ in {1..10}; do
        kill -0 "${evaluator_pid}" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "${evaluator_pid}" 2>/dev/null || true
    fi
    pkill -KILL -f "${CARLA_ROOT}/.*-graphicsadapter=${gpu_rank}" 2>/dev/null || true
  }
  trap 'cleanup_worker_processes; exit 130' INT
  trap 'cleanup_worker_processes; exit 143' TERM

  while true; do
    {
      echo "===== worker task=${task_id} attempt=$((restarts + 1)) started $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      echo "routes=${routes} checkpoint=${checkpoint_endpoint} gpu=${gpu_rank} port=${port} tm_port=${tm_port}"
    } >> "${log_path}"
    log_offset=$(( $(stat -c %s "${log_path}") + 1 ))

    CUDA_VISIBLE_DEVICES="${gpu_rank}" python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
      --routes="${routes}" \
      --repetitions=1 \
      --track=SENSORS \
      --checkpoint="${checkpoint_endpoint}" \
      --agent="${TEAM_AGENT}" \
      --agent-config="${worker_team_config}" \
      --debug=0 \
      --resume=True \
      --port="${port}" \
      --traffic-manager-port="${tm_port}" \
      --gpu-rank="${gpu_rank}" \
      --timeout="${TIMEOUT}" >> "${log_path}" 2>&1 &
    evaluator_pid=$!

    while kill -0 "${evaluator_pid}" 2>/dev/null; do
      if tail -c +"${log_offset}" "${log_path}" 2>/dev/null | \
        rg -q 'LowLevelFatalError|Segmentation fault \(core dumped\)|Engine crash handling finished'; then
        echo "===== worker task=${task_id} detected CARLA crash; terminating evaluator pid=${evaluator_pid} =====" >> "${log_path}"
        kill "${evaluator_pid}" 2>/dev/null || true
        break
      fi
      sleep 5
    done

    set +e
    wait "${evaluator_pid}"
    rc=$?
    set -e
    cleanup_worker_processes
    evaluator_pid=""

    if checkpoint_progress_complete "${checkpoint_endpoint}"; then
      echo "===== worker task=${task_id} complete $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "${log_path}"
      trap - INT TERM
      return 0
    fi

    if (( restarts >= MAX_WORKER_RESTARTS )); then
      echo "===== worker task=${task_id} incomplete after ${restarts} restarts; last_rc=${rc} =====" >> "${log_path}"
      trap - INT TERM
      return "$((rc == 0 ? 1 : rc))"
    fi
    restarts=$((restarts + 1))
    echo "===== worker task=${task_id} incomplete; retry=${restarts}/${MAX_WORKER_RESTARTS} in ${WORKER_RETRY_DELAY}s (last_rc=${rc}) =====" >> "${log_path}"
    sleep "${WORKER_RETRY_DELAY}"
  done
}

export VLA_AD_ROOT BENCH2DRIVE_ROOT CARLA_ROOT SAVE_PATH
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${PYTHONPATH:-}:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${BENCH2DRIVE_ROOT}/leaderboard:${BENCH2DRIVE_ROOT}/leaderboard/team_code:${BENCH2DRIVE_ROOT}/scenario_runner:${VLA_AD_ROOT}/bench2drive_eval/team_code:${VLA_AD_ROOT}"
export SCENARIO_RUNNER_ROOT="${BENCH2DRIVE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${BENCH2DRIVE_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export PLANNER_TYPE

cd "${BENCH2DRIVE_ROOT}"

if [ "${START_INFERENCE_SERVERS}" = "1" ]; then
  mkdir -p "${SERVER_LOG_DIR}"
  worker_count=${#GPU_RANK_LIST[@]}
  server_gpu_count=${#SERVER_GPU_RANK_LIST[@]}
  for ((i=0; i<worker_count; i++)); do
    SERVER_PORT=$(server_port_for_worker "${i}")
    SERVER_GPU=${SERVER_GPU_RANK_LIST[$((i % server_gpu_count))]}
    SERVER_LOG="${SERVER_LOG_DIR}/server_${i}_gpu${SERVER_GPU}_port${SERVER_PORT}.log"
    echo "Launching inference server worker=${i} gpu=${SERVER_GPU} port=${SERVER_PORT} log=${SERVER_LOG}"
    (
      cd "${VLA_AD_ROOT}"
      CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_CONDA_ENV}" \
        python scripts/bench2drive/serve_recogdrive_b2d.py \
          --host "${SERVER_HOST}" \
          --port "${SERVER_PORT}" \
          --config configs/bench2drive_recogdrive_il.yaml \
          --planner-checkpoint "${PLANNER_CHECKPOINT}" \
          --vlm-path "${VLM_PATH}" \
          --precision "${PRECISION}" \
          --profile-every "${SERVER_PROFILE_EVERY}" > "${SERVER_LOG}" 2>&1
    ) &
    SERVER_PIDS+=("$!")
  done
  echo "Waiting ${SERVER_STARTUP_SECONDS}s for inference servers to load models before health checks"
  sleep "${SERVER_STARTUP_SECONDS}"
  for ((i=0; i<worker_count; i++)); do
    SERVER_PORT=$(server_port_for_worker "${i}")
    if ! check_server_health "${SERVER_HOST}" "${SERVER_PORT}"; then
      echo "Inference server health check failed for port=${SERVER_PORT}. Last log lines:"
      tail -n 80 "${SERVER_LOG_DIR}/server_${i}_gpu${SERVER_GPU_RANK_LIST[$((i % ${#SERVER_GPU_RANK_LIST[@]}))]}_port${SERVER_PORT}.log" || true
      exit 1
    fi
  done
fi

SPLIT_FLAG="${BASE_ROUTES}_${TASK_NUM}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
MISSING_SPLIT=0
for task_id in "${TASK_LIST[@]}"; do
  if [ ! -f "${BASE_ROUTES}_${task_id}_${ALGO}_${PLANNER_TYPE}.xml" ]; then
    MISSING_SPLIT=1
  fi
done
if [ "${FORCE_SPLIT}" = "1" ] || [ ! -f "${SPLIT_FLAG}" ] || [ "${MISSING_SPLIT}" = "1" ]; then
  python tools/split_xml.py "${BASE_ROUTES}" "${TASK_NUM}" "${ALGO}" "${PLANNER_TYPE}"
  touch "${SPLIT_FLAG}"
fi

mkdir -p "${RESULT_JSON_DIR}" "${SAVE_PATH}" "${GENERATED_TEAM_CONFIG_DIR}" "${WORKER_LOG_DIR}"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<length; i++)); do
  PORT=$((BASE_PORT + i * 150))
  TM_PORT=$((BASE_TM_PORT + i * 150))
  ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
  CHECKPOINT_ENDPOINT="${RESULT_JSON_DIR}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
  GPU_RANK=${GPU_RANK_LIST[$i]}
  LOG_PATH="${WORKER_LOG_DIR}/task_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log"
  WORKER_TEAM_CONFIG="${TEAM_CONFIG}"
  if [ "${USE_PER_WORKER_SERVER_URLS}" = "1" ]; then
    SERVER_PORT=$(server_port_for_worker "${i}")
    WORKER_TEAM_CONFIG="${GENERATED_TEAM_CONFIG_DIR}/team_config_task${TASK_LIST[$i]}_port${SERVER_PORT}.yaml"
    write_worker_config "${TEAM_CONFIG}" "${WORKER_TEAM_CONFIG}" "http://${SERVER_HOST}:${SERVER_PORT}" "${TASK_LIST[$i]}"
  fi
  echo "Launching task=${TASK_LIST[$i]} gpu=${GPU_RANK} routes=${ROUTES} team_config=${WORKER_TEAM_CONFIG}"
  run_eval_worker \
    "${TASK_LIST[$i]}" \
    "${GPU_RANK}" \
    "${PORT}" \
    "${TM_PORT}" \
    "${ROUTES}" \
    "${CHECKPOINT_ENDPOINT}" \
    "${WORKER_TEAM_CONFIG}" \
    "${LOG_PATH}" &
  EVAL_PIDS+=("$!")
  sleep 10
done

WAIT_RC=0
for pid in "${EVAL_PIDS[@]}"; do
  if ! wait "${pid}"; then
    WAIT_RC=1
  fi
done
exit "${WAIT_RC}"
