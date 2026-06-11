#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  RECOGDRIVE_VLM_PATH
  CACHE_PATH
  ANCHOR_CACHE_DIR
  TRAIN_TEST_SPLIT
  OUTPUT_DIR
  MASTER_PORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}. Set ALLOW_OVERWRITE=1 to reuse it." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
SEED="${SEED:-0}"
KEY_STEPS="${KEY_STEPS:-50000,60000,80000,100000,120000,140000,160000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-recogdrive_stage2_residual_anchor_2bbase}"

mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/${EXPERIMENT_NAME}.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  agent=recogdrive_agent
  agent.lr=1e-4
  agent.grpo=False
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH}"
  agent.cam_type=single
  agent.cache_hidden_state=True
  agent.vlm_type=internvl
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  agent.use_expert_features=False
  agent.use_jepa=False
  agent.use_vggt=False
  agent.use_last_rd=False
  agent.use_last_vla=False
  agent.last_vla_stage=disabled
  agent.last_vla_use_residual_diffusion=True
  agent.last_vla_residual_anchor_source=vlm_text_traj
  agent.last_vla_require_residual_anchor=True
  "agent.last_vla_residual_anchor_cache_dir=${ANCHOR_CACHE_DIR}"
  agent.last_vla_residual_alpha_start=1.0
  agent.last_vla_residual_alpha_end=1.0
  agent.last_vla_residual_alpha_warmup_epochs=0
  trainer.params.max_epochs=200
  trainer.params.devices=8
  trainer.params.accumulate_grad_batches=1
  trainer.params.default_root_dir="${OUTPUT_DIR}"
  dataloader.params.batch_size=16
  dataloader.params.num_workers=8
  dataloader.params.prefetch_factor=2
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache_path=${CACHE_PATH}"
  use_cache_without_dataset=True
  force_cache_computation=False
  seed="${SEED}"
  "output_dir=${OUTPUT_DIR}"
  "experiment_name=${EXPERIMENT_NAME}"
)

{
  printf '[%s] RECOGDRIVE_STAGE2_RESIDUAL_ANCHOR_KEY_STEPS=%q ' "$(date -Is)" "${KEY_STEPS}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "Starting ReCogDrive stage2 residual-anchor. Log: ${TRAIN_LOG}"
set +e
RECOGDRIVE_STAGE2_RESIDUAL_ANCHOR_KEY_STEPS="${KEY_STEPS}" "${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
echo "torchrun exited with status ${status}. Log: ${TRAIN_LOG}" >&2
exit "${status}"
