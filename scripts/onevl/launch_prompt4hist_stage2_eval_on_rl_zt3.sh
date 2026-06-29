#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt3}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/envs/navsim/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_eval_${RUN_ID}}"
TRAIN_DIR="${TRAIN_DIR:-/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/train_full200}"

STAGE1_CKPT="${STAGE1_CKPT:-/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228}"
VAL_DATA_JSONL="${VAL_DATA_JSONL:-/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_val6000_from_metric_cache_20260628.jsonl}"
NAV_DATA_JSON="${NAV_DATA_JSON:-/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json}"
VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-${OUT_ROOT}/cache_val6000}"
NAV_CACHE_ROOT="${NAV_CACHE_ROOT:-${OUT_ROOT}/cache_navtest}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/eval_top5}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/project/VLA-AD/configs/onevl_ar_answer_stage2_small.yaml}"
VAL_METRIC_CACHE="${VAL_METRIC_CACHE:-/mnt/project/onevl_navsim_exp/metric_cache_val6000_latest}"
NAV_METRIC_CACHE="${NAV_METRIC_CACHE:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MIN_EPOCH="${MIN_EPOCH:-50}"
MIN_STEP="${MIN_STEP:-40000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
TOP_K="${TOP_K:-5}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_USED_MAX_MB="${GPU_USED_MAX_MB:-1024}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"
EVAL_WATCHER_HOST="${EVAL_WATCHER_HOST:-remote}"

mkdir -p "${OUT_ROOT}/logs"

for path in "${STAGE1_CKPT}" "${VAL_DATA_JSONL}" "${NAV_DATA_JSON}" "${TRAIN_DIR}" "${CONFIG_PATH}" "${VAL_METRIC_CACHE}" "${NAV_METRIC_CACHE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

echo "== local preflight =="
hostname
"${PYTHON_BIN}" - <<'PY'
import torch
print("local_cuda_count", torch.cuda.device_count())
PY

echo "== remote preflight ${REMOTE_HOST} =="
ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "
set -euo pipefail
hostname
cd /mnt/project/VLA-AD
${REMOTE_PYTHON} - <<'PY'
import torch
print('remote_cuda_count', torch.cuda.device_count())
PY
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
"

{
  echo "out_root=${OUT_ROOT}"
  echo "remote_host=${REMOTE_HOST}"
  echo "train_dir=${TRAIN_DIR}"
  echo "stage1_ckpt=${STAGE1_CKPT}"
  echo "val_data_jsonl=${VAL_DATA_JSONL}"
  echo "nav_data_json=${NAV_DATA_JSON}"
  echo "val_cache_root=${VAL_CACHE_ROOT}"
  echo "nav_cache_root=${NAV_CACHE_ROOT}"
  echo "eval_root=${EVAL_ROOT}"
  echo "config_path=${CONFIG_PATH}"
  echo "val_metric_cache=${VAL_METRIC_CACHE}"
  echo "nav_metric_cache=${NAV_METRIC_CACHE}"
  echo "min_epoch=${MIN_EPOCH}"
  echo "min_step=${MIN_STEP}"
  echo "wait_for_gpu_free=${WAIT_FOR_GPU_FREE}"
  echo "gpu_used_max_mb=${GPU_USED_MAX_MB}"
} > "${OUT_ROOT}/launch.env"

remote_cmd=$(printf '%q ' \
  env \
  PYTHON_BIN="${REMOTE_PYTHON}" \
  OUT_ROOT="${OUT_ROOT}" \
  STAGE1_CKPT="${STAGE1_CKPT}" \
  VAL_DATA_JSONL="${VAL_DATA_JSONL}" \
  NAV_DATA_JSON="${NAV_DATA_JSON}" \
  VAL_CACHE_ROOT="${VAL_CACHE_ROOT}" \
  NAV_CACHE_ROOT="${NAV_CACHE_ROOT}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  GPU_LIST="${GPU_LIST}" \
  WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE}" \
  GPU_USED_MAX_MB="${GPU_USED_MAX_MB}" \
  GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS}" \
  bash "${SCRIPT_DIR}/build_prompt4hist_stage2_eval_caches.sh")

ssh -o BatchMode=yes "${REMOTE_HOST}" "
set -euo pipefail
mkdir -p '${OUT_ROOT}/logs'
cd /mnt/project
setsid ${remote_cmd} > '${OUT_ROOT}/logs/eval_cache.remote.outer.log' 2>&1 < /dev/null &
echo \$! > '${OUT_ROOT}/eval_cache.remote.pid'
"

if [[ "${EVAL_WATCHER_HOST}" == "remote" ]]; then
  ssh -o BatchMode=yes "${REMOTE_HOST}" "
  set -euo pipefail
  mkdir -p '${OUT_ROOT}/logs'
  cd /mnt/project
  setsid env \
    PYTHON_BIN='${REMOTE_PYTHON}' \
    OUT_ROOT='${OUT_ROOT}' \
    TRAIN_DIR='${TRAIN_DIR}' \
    CONFIG_PATH='${CONFIG_PATH}' \
    EVAL_ROOT='${EVAL_ROOT}' \
    VAL_CACHE_ROOT='${VAL_CACHE_ROOT}' \
    NAV_CACHE_ROOT='${NAV_CACHE_ROOT}' \
    VAL_METRIC_CACHE='${VAL_METRIC_CACHE}' \
    NAV_METRIC_CACHE='${NAV_METRIC_CACHE}' \
    REMOTE_HOST=local \
    REMOTE_PYTHON='${REMOTE_PYTHON}' \
    MIN_EPOCH='${MIN_EPOCH}' \
    MIN_STEP='${MIN_STEP}' \
    POLL_SECONDS='${POLL_SECONDS}' \
    STABLE_SECONDS='${STABLE_SECONDS}' \
    NUM_SHARDS='${NUM_SHARDS}' \
    TOP_K='${TOP_K}' \
    bash '${SCRIPT_DIR}/watch_prompt4hist_stage2_eval_cache_then_eval.sh' \
    > '${OUT_ROOT}/logs/eval_watcher.remote.outer.log' 2>&1 < /dev/null &
  echo \$! > '${OUT_ROOT}/eval_watcher.remote.pid'
  "
else
  setsid env \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUT_ROOT="${OUT_ROOT}" \
    TRAIN_DIR="${TRAIN_DIR}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    EVAL_ROOT="${EVAL_ROOT}" \
    VAL_CACHE_ROOT="${VAL_CACHE_ROOT}" \
    NAV_CACHE_ROOT="${NAV_CACHE_ROOT}" \
    VAL_METRIC_CACHE="${VAL_METRIC_CACHE}" \
    NAV_METRIC_CACHE="${NAV_METRIC_CACHE}" \
    REMOTE_HOST="${REMOTE_HOST}" \
    REMOTE_PYTHON="${REMOTE_PYTHON}" \
    MIN_EPOCH="${MIN_EPOCH}" \
    MIN_STEP="${MIN_STEP}" \
    POLL_SECONDS="${POLL_SECONDS}" \
    STABLE_SECONDS="${STABLE_SECONDS}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    TOP_K="${TOP_K}" \
    bash "${SCRIPT_DIR}/watch_prompt4hist_stage2_eval_cache_then_eval.sh" \
    > "${OUT_ROOT}/logs/eval_watcher.outer.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/eval_watcher.pid"
fi

ln -sfn "${OUT_ROOT}" /mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_eval_latest

echo "launched_out_root=${OUT_ROOT}"
echo "remote_cache_pid=$(cat "${OUT_ROOT}/eval_cache.remote.pid")"
if [[ "${EVAL_WATCHER_HOST}" == "remote" ]]; then
  echo "remote_eval_watcher_pid=$(cat "${OUT_ROOT}/eval_watcher.remote.pid")"
else
  echo "local_eval_watcher_pid=$(cat "${OUT_ROOT}/eval_watcher.pid")"
fi
