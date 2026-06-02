#!/usr/bin/env bash
set -Eeuo pipefail

required=(TRAIN_CHUNK_CACHE_ROOT OUTPUT_DIR MASTER_PORT COT_ALIGNMENT_CHECKPOINT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
[[ -d "${TRAIN_CHUNK_CACHE_ROOT}" ]] || { echo "TRAIN_CHUNK_CACHE_ROOT missing: ${TRAIN_CHUNK_CACHE_ROOT}" >&2; exit 2; }
[[ -e "${COT_ALIGNMENT_CHECKPOINT}" ]] || { echo "COT_ALIGNMENT_CHECKPOINT missing: ${COT_ALIGNMENT_CHECKPOINT}" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SEED="${SEED:-0}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/last_vla_progressive_bottleneck.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_progressive_bottleneck
  "cache_path=${TRAIN_CHUNK_CACHE_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "output_dir=${OUTPUT_DIR}"
  seed="${SEED}"
  agent.use_last_vla=true
  agent.last_vla_stage=progressive_sft_bottleneck
  agent.use_last_rd=false
  agent.last_vla_require_full_geometry="${LAST_VLA_REQUIRE_FULL_GEOMETRY:-false}"
  agent.last_vla_allow_patch_geometry_fallback="${LAST_VLA_ALLOW_PATCH_GEOMETRY_FALLBACK:-true}"
  agent.last_vla_use_residual_diffusion=true
  agent.last_vla_cot_bottleneck_mode=true
  agent.last_vla_raw_vlm_context_to_dit=false
  agent.policy_kd_loss_weight="${LAST_VLA_POLICY_KD_WEIGHT:-0.0}"
  agent.policy_kd_mode="${LAST_VLA_POLICY_KD_MODE:-none}"
  ++agent.last_vla_adapter_checkpoint="${COT_ALIGNMENT_CHECKPOINT}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
if [[ -n "${A0_INIT_CHECKPOINT}" ]]; then
  cmd+=(agent.checkpoint_path="${A0_INIT_CHECKPOINT}")
fi

{
  date -Is
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${LAST_VLA_DRY_RUN:-0}" == "1" ]]; then
  echo "LAST_VLA_DRY_RUN=1; not launching training."
  exit 0
fi

echo "Starting Last-VLA progressive bottleneck SFT. Log: ${TRAIN_LOG}"
set +e
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "${status}" >>"${COMMANDS_LOG}"
exit "${status}"
