#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
MASTER_PORT="${MASTER_PORT:-29592}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/strict_last_vla_stage2_diffusion_sft}"
SEED="${SEED:-0}"
KEY_STEPS="${KEY_STEPS:-50000,60000,80000,100000,120000,140000,160000}"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  agent=recogdrive_agent
  +experiment=last_vla_v2/strict_last_vla/stage2_diffusion_sft
  agent.lr=1e-4
  agent.grpo=False
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH:-}"
  agent.cam_type=single
  agent.cache_hidden_state=True
  agent.vlm_type=internvl
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  "agent.checkpoint_path=${A0_INIT_CHECKPOINT:-}"
  "agent.last_vla_adapter_checkpoint=${STAGE1_ADAPTER_CHECKPOINT:-}"
  trainer.params.max_epochs=200
  trainer.params.devices=8
  trainer.params.accumulate_grad_batches=1
  trainer.params.default_root_dir="${OUTPUT_DIR}"
  dataloader.params.batch_size=16
  dataloader.params.num_workers=8
  dataloader.params.prefetch_factor=2
  "train_test_split=${TRAIN_TEST_SPLIT:-}"
  "cache_path=${STRICT_LATENT_CACHE_ROOT:-}"
  use_cache_without_dataset=True
  force_cache_computation=False
  seed="${SEED}"
  "output_dir=${OUTPUT_DIR}"
  experiment_name=strict_last_vla_stage2_diffusion_sft
)

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  printf 'RUN_TRAIN is not 1; Stage2 strict diffusion SFT not launched.\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

for name in RECOGDRIVE_VLM_PATH TRAIN_TEST_SPLIT STRICT_LATENT_CACHE_ROOT A0_INIT_CHECKPOINT STAGE1_ADAPTER_CHECKPOINT; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}/logs"
{
  printf '[%s] A0_INIT_CHECKPOINT=%q A0_BASELINE_PDMS=0.864891 A0_STAGE2_KEY_STEPS=%q ' "$(date -Is)" "${A0_INIT_CHECKPOINT}" "${KEY_STEPS}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${OUTPUT_DIR}/commands.log"
A0_STAGE2_KEY_STEPS="${KEY_STEPS}" "${cmd[@]}" >"${OUTPUT_DIR}/logs/stage2_diffusion_sft.train.log" 2>&1
