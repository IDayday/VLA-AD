#!/usr/bin/env bash
set -Eeuo pipefail

required=(TEACHER_TRAJ_CHUNK_ROOT OUTPUT_DIR MASTER_PORT PROGRESSIVE_CHECKPOINT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
[[ -d "${TEACHER_TRAJ_CHUNK_ROOT}" ]] || { echo "TEACHER_TRAJ_CHUNK_ROOT missing: ${TEACHER_TRAJ_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${PROGRESSIVE_CHECKPOINT}" ]] || { echo "PROGRESSIVE_CHECKPOINT missing: ${PROGRESSIVE_CHECKPOINT}" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
SEED="${SEED:-0}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/last_vla_teacher_traj_sft.train.log"
MANIFEST="${OUTPUT_DIR}/last_vla_teacher_manifest.json"

"${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py \
  --cache-root "${TEACHER_TRAJ_CHUNK_ROOT}" \
  --output "${MANIFEST}" \
  --strict-teacher-traj-sft \
  --min-teacher-coverage "${MIN_TEACHER_TRAJ_COVERAGE:-0.99}"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_teacher_traj_sft
  "cache_path=${TEACHER_TRAJ_CHUNK_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "output_dir=${OUTPUT_DIR}"
  seed="${SEED}"
  agent.use_last_vla=true
  agent.last_vla_stage=teacher_traj_sft
  agent.last_vla_require_full_geometry="${LAST_VLA_REQUIRE_FULL_GEOMETRY:-false}"
  agent.last_vla_allow_patch_geometry_fallback="${LAST_VLA_ALLOW_PATCH_GEOMETRY_FALLBACK:-true}"
  agent.last_vla_teacher_traj_mode="${LAST_VLA_TEACHER_TRAJ_MODE:-teacher_if_better}"
  agent.policy_kd_loss_weight="${LAST_VLA_POLICY_KD_WEIGHT:-0.0}"
  agent.policy_kd_mode="${LAST_VLA_POLICY_KD_MODE:-none}"
  ++agent.last_vla_adapter_checkpoint="${PROGRESSIVE_CHECKPOINT}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)

{
  date -Is
  printf 'MANIFEST=%s\n' "${MANIFEST}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${LAST_VLA_DRY_RUN:-0}" == "1" ]]; then
  echo "LAST_VLA_DRY_RUN=1; not launching training."
  exit 0
fi

echo "Starting Last-VLA teacher trajectory SFT. Log: ${TRAIN_LOG}"
set +e
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
status=$?
set -e
printf '%s exit_code=%s\n' "$(date -Is)" "${status}" >>"${COMMANDS_LOG}"
exit "${status}"
