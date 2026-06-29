#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/mnt/project/onevl_navsim_exp/answer_full}
LOG_DIR="${OUT_ROOT}/logs"
CMD_LOG="${OUT_ROOT}/commands.log"
SCRIPT=/mnt/project/OneVL_training/run_script/train/navsim/sft_distributed_qwen3vl_answer_bs64.sh

mkdir -p "${LOG_DIR}"

export MODEL_PATH="${MODEL_PATH:-/mnt/project/onevl_models/Qwen3-VL-4B-Instruct}"
export DATASET_PATH="${DATASET_PATH:-/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/swift_output}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((31500 + RANDOM % 1000))}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
export MAX_STEPS="${MAX_STEPS:--1}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export DEEPSPEED="${DEEPSPEED:-zero2}"
export PYTHONPATH="/mnt:${PYTHONPATH:-}"

GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-20}"
REQUIRE_FREE_GPUS="${REQUIRE_FREE_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if (( gpu_count < NPROC_PER_NODE )); then
  echo "need ${NPROC_PER_NODE} GPUs, found ${gpu_count}" | tee -a "${LOG_DIR}/launcher.status.log"
  exit 1
fi

if [[ ! -s "${DATASET_PATH}" ]]; then
  echo "missing dataset: ${DATASET_PATH}" | tee -a "${LOG_DIR}/launcher.status.log"
  exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  > "${LOG_DIR}/gpu_preflight.csv"

busy_gpus=$(
  awk -F, -v n="${NPROC_PER_NODE}" -v max_mem="${GPU_MAX_USED_MB}" -v max_util="${GPU_MAX_UTIL}" '
    {
      idx=$1 + 0
      mem=$3
      util=$5
      gsub(/ /, "", mem)
      gsub(/ /, "", util)
      if (idx < n && (mem + 0 > max_mem || util + 0 > max_util)) {
        printf "gpu=%s memory_used_mb=%s util_pct=%s\n", idx, mem, util
      }
    }
  ' "${LOG_DIR}/gpu_preflight.csv"
)
if [[ -n "${busy_gpus}" && "${REQUIRE_FREE_GPUS}" == "1" ]]; then
  {
    echo "GPUs are too busy for the answer full run:"
    printf '%s\n' "${busy_gpus}"
  } | tee -a "${LOG_DIR}/launcher.status.log"
  exit 75
fi

cmd=(bash "${SCRIPT}")

{
  printf '[%s] ' "$(date -Is)"
  printf 'OUT_ROOT=%q MODEL_PATH=%q DATASET_PATH=%q OUTPUT_DIR=%q NPROC_PER_NODE=%q MASTER_PORT=%q NUM_TRAIN_EPOCHS=%q MAX_STEPS=%q PER_DEVICE_TRAIN_BATCH_SIZE=%q GRADIENT_ACCUMULATION_STEPS=%q ' \
    "${OUT_ROOT}" "${MODEL_PATH}" "${DATASET_PATH}" "${OUTPUT_DIR}" "${NPROC_PER_NODE}" "${MASTER_PORT}" "${NUM_TRAIN_EPOCHS}" "${MAX_STEPS}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >> "${CMD_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "dry run complete; command not launched" | tee -a "${LOG_DIR}/launcher.status.log"
  exit 0
fi

set +e
"${cmd[@]}" > "${LOG_DIR}/train.log" 2>&1
status=$?
set -e
echo "answer full exited with status ${status}" | tee -a "${LOG_DIR}/launcher.status.log"
exit "${status}"
