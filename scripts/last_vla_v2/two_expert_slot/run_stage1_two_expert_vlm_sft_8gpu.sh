#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUTPUT_DIR MASTER_PORT NAVSIM_LOG_PATH SENSOR_BLOBS_PATH VLM_PATH TEACHER_CACHE_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/two_expert_stage1_vlm_sft.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=two_expert_slot_stage1_vlm_sft
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "train_test_split=${TRAIN_TEST_SPLIT:-navtrain}"
  "output_dir=${OUTPUT_DIR}"
  "agent.vlm_path=${VLM_PATH}"
  "teacher_cache_root=${TEACHER_CACHE_ROOT}"
  seed="${SEED:-0}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
