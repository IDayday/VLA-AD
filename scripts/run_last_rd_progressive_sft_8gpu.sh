#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_5_CHECKPOINT="${STAGE1_5_CHECKPOINT:?Set STAGE1_5_CHECKPOINT to the adapter pretraining checkpoint.}"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
BASE_CONFIG="${BASE_CONFIG:-last_rd_progressive_sft}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT:-}"
LAST_RD_POLICY_KD_WEIGHT="${LAST_RD_POLICY_KD_WEIGHT:-0.05}"
LAST_RD_POLICY_KD_MODE="${LAST_RD_POLICY_KD_MODE:-noise}"
LAST_RD_RISK_LOSS_WEIGHT="${LAST_RD_RISK_LOSS_WEIGHT:-0.0}"
LAST_RD_NUM_WORKERS="${LAST_RD_NUM_WORKERS:-12}"
LAST_RD_PREFETCH_FACTOR="${LAST_RD_PREFETCH_FACTOR:-4}"
LAST_RD_PERSISTENT_WORKERS="${LAST_RD_PERSISTENT_WORKERS:-true}"
export LAST_RD_RUNTIME_FINITE_CHECK="${LAST_RD_RUNTIME_FINITE_CHECK:-1}"
export LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS="${LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS:-262144}"
POLICY_KD_ENABLED="$(python - "${LAST_RD_POLICY_KD_WEIGHT}" "${LAST_RD_POLICY_KD_MODE}" <<'PY'
import sys
weight = float(sys.argv[1])
mode = sys.argv[2]
print("1" if weight > 0.0 and mode != "none" else "0")
PY
)"

if [[ "${POLICY_KD_ENABLED}" == "1" && -z "${A0_REFERENCE_CHECKPOINT}" ]]; then
  echo "Progressive SFT policy KD is enabled (weight=${LAST_RD_POLICY_KD_WEIGHT}, mode=${LAST_RD_POLICY_KD_MODE}) but A0_REFERENCE_CHECKPOINT is empty." >&2
  echo "Provide the A0-official-aligned reference checkpoint or set LAST_RD_POLICY_KD_WEIGHT=0." >&2
  exit 1
fi
if [[ "${POLICY_KD_ENABLED}" == "0" ]]; then
  LAST_RD_POLICY_KD_MODE="none"
fi

mkdir -p "${OUTPUT_DIR}"
{
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'BASE_CONFIG is a Hydra experiment config name under navsim/planning/script/config/experiment.\n'
  printf 'STAGE1_5_CHECKPOINT=%s\n' "${STAGE1_5_CHECKPOINT}"
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf 'A0_REFERENCE_CHECKPOINT=%s\n' "${A0_REFERENCE_CHECKPOINT}"
  printf 'LAST_RD_POLICY_KD_WEIGHT=%s\n' "${LAST_RD_POLICY_KD_WEIGHT}"
  printf 'LAST_RD_POLICY_KD_MODE=%s\n' "${LAST_RD_POLICY_KD_MODE}"
  printf 'LAST_RD_RISK_LOSS_WEIGHT=%s\n' "${LAST_RD_RISK_LOSS_WEIGHT}"
  printf 'LAST_RD_NUM_WORKERS=%s\n' "${LAST_RD_NUM_WORKERS}"
  printf 'LAST_RD_PREFETCH_FACTOR=%s\n' "${LAST_RD_PREFETCH_FACTOR}"
  printf 'LAST_RD_PERSISTENT_WORKERS=%s\n' "${LAST_RD_PERSISTENT_WORKERS}"
  printf 'LAST_RD_RUNTIME_FINITE_CHECK=%s\n' "${LAST_RD_RUNTIME_FINITE_CHECK}"
  printf 'LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS=%s\n' "${LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS}"
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'Recommended preflight: python scripts/audit_last_rd_cache_manifest.py --cache-root %s --output reports/last_rd_cache_manifest.json\n' "${TRAIN_CHUNK_CACHE_ROOT}"
} > "${OUTPUT_DIR}/commands.log"

CHECKPOINT_OVERRIDE=()
if [[ -n "${A0_INIT_CHECKPOINT}" ]]; then
  CHECKPOINT_OVERRIDE+=(agent.checkpoint_path="${A0_INIT_CHECKPOINT}")
fi
REFERENCE_OVERRIDE=()
if [[ -n "${A0_REFERENCE_CHECKPOINT}" ]]; then
  REFERENCE_OVERRIDE+=(++agent.reference_a0_checkpoint="${A0_REFERENCE_CHECKPOINT}")
fi

if [[ "${LAST_RD_DRY_RUN:-0}" == "1" ]]; then
  echo "LAST_RD_DRY_RUN=1; Progressive SFT launcher validated commands.log and will not start torchrun."
  echo "Would run: torchrun --nproc_per_node=8 --master_port=${MASTER_PORT} ${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py +experiment=${BASE_CONFIG} ..."
  exit 0
fi

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  +experiment="${BASE_CONFIG}" \
  cache_path="${TRAIN_CHUNK_CACHE_ROOT}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  agent.use_expert_features=true \
  agent.use_jepa=true \
  agent.use_vggt=true \
  agent.allow_expert_target_features=true \
  agent.use_last_rd=true \
  agent.last_rd_stage=progressive_sft \
  agent.diffusion_loss_weight=1.0 \
  agent.future_jepa_loss_weight=0.10 \
  agent.future_jepa_loss_floor=0.01 \
  agent.vggt_geometry_loss_weight=0.05 \
  agent.vggt_geometry_loss_floor=0.005 \
  agent.coarse_traj_loss_weight=0.20 \
  agent.coarse_traj_loss_floor=0.05 \
  agent.coarse_heading_loss_weight=0.05 \
  agent.risk_loss_weight="${LAST_RD_RISK_LOSS_WEIGHT}" \
  agent.risk_loss_floor=0.005 \
  agent.policy_kd_loss_weight="${LAST_RD_POLICY_KD_WEIGHT}" \
  agent.policy_kd_mode="${LAST_RD_POLICY_KD_MODE}" \
  agent.last_rd_context_scale=1.0 \
  agent.last_rd_horizon_condition_scale=1.0 \
  trainer.params.max_epochs=200 \
  trainer.params.devices=8 \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers="${LAST_RD_NUM_WORKERS}" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor="${LAST_RD_PREFETCH_FACTOR}" \
  ++dataloader.params.persistent_workers="${LAST_RD_PERSISTENT_WORKERS}" \
  agent.last_rd_adapter_checkpoint="${STAGE1_5_CHECKPOINT}" \
  "${CHECKPOINT_OVERRIDE[@]}" \
  "${REFERENCE_OVERRIDE[@]}"
