#!/usr/bin/env bash
set -Eeuo pipefail

required=(TRAIN_CHUNK_CACHE_ROOT OUTPUT_DIR MASTER_PORT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
if [[ ! -d "${TRAIN_CHUNK_CACHE_ROOT}" ]]; then
  echo "TRAIN_CHUNK_CACHE_ROOT does not exist: ${TRAIN_CHUNK_CACHE_ROOT}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SEED="${SEED:-0}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/last_vla_cot_alignment.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_cot_alignment
  "cache_path=${TRAIN_CHUNK_CACHE_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "output_dir=${OUTPUT_DIR}"
  seed="${SEED}"
  agent.use_last_vla=true
  agent.last_vla_stage=cot_alignment
  agent.use_last_rd=false
  agent.use_expert_features=false
  agent.last_vla_require_full_geometry="${LAST_VLA_REQUIRE_FULL_GEOMETRY:-false}"
  agent.last_vla_allow_patch_geometry_fallback="${LAST_VLA_ALLOW_PATCH_GEOMETRY_FALLBACK:-true}"
  agent.allow_expert_target_features=true
  agent.diffusion_loss_weight=0.0
  agent.policy_kd_loss_weight=0.0
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
if [[ -n "${A0_INIT_CHECKPOINT}" ]]; then
  [[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }
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

echo "Starting Last-VLA CoT alignment. Log: ${TRAIN_LOG}"
set +e
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "${status}" >>"${COMMANDS_LOG}"
exit "${status}"
