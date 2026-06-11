#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
MASTER_PORT="${MASTER_PORT:-29591}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/strict_last_vla_stage1_latent_alignment}"
SEED="${SEED:-0}"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  agent=recogdrive_agent
  +experiment=last_vla_v2/strict_last_vla/stage1_latent_alignment
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH:-}"
  agent.vlm_type=internvl
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  trainer.params.devices=8
  trainer.params.max_epochs="${STAGE1_EPOCHS:-20}"
  trainer.params.default_root_dir="${OUTPUT_DIR}"
  dataloader.params.batch_size="${BATCH_SIZE:-16}"
  dataloader.params.num_workers="${NUM_WORKERS:-8}"
  "train_test_split=${TRAIN_TEST_SPLIT:-}"
  "cache_path=${CACHE_PATH:-}"
  use_cache_without_dataset=False
  force_cache_computation=False
  seed="${SEED}"
  "output_dir=${OUTPUT_DIR}"
  experiment_name=strict_last_vla_stage1_latent_alignment
)

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  printf 'RUN_TRAIN is not 1; Stage1 strict latent alignment not launched.\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

for name in RECOGDRIVE_VLM_PATH TRAIN_TEST_SPLIT CACHE_PATH; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}/logs"
printf '%q ' "${cmd[@]}" >"${OUTPUT_DIR}/commands.log"
printf '\n' >>"${OUTPUT_DIR}/commands.log"
"${cmd[@]}" >"${OUTPUT_DIR}/logs/stage1_latent_alignment.train.log" 2>&1
