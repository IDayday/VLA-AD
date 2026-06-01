#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:?Set A0_INIT_CHECKPOINT.}"
A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT:?Set A0_REFERENCE_CHECKPOINT.}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
STAGE1_5_CHECKPOINT="${STAGE1_5_CHECKPOINT:-${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt}"
OUTPUT_DIR="${OUT_ROOT}/wave2_progressive/lastrd_only"
LAST_RD_NUM_WORKERS="${LAST_RD_NUM_WORKERS:-12}"
LAST_RD_PREFETCH_FACTOR="${LAST_RD_PREFETCH_FACTOR:-4}"
LAST_RD_PERSISTENT_WORKERS="${LAST_RD_PERSISTENT_WORKERS:-true}"
export LAST_RD_RUNTIME_FINITE_CHECK="${LAST_RD_RUNTIME_FINITE_CHECK:-1}"
export LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS="${LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS:-262144}"

[[ -f "${STAGE1_5_CHECKPOINT}" ]] || { echo "Missing STAGE1_5_CHECKPOINT=${STAGE1_5_CHECKPOINT}" >&2; exit 1; }
[[ -f "${A0_INIT_CHECKPOINT}" ]] || { echo "Missing A0_INIT_CHECKPOINT=${A0_INIT_CHECKPOINT}" >&2; exit 1; }
[[ -f "${A0_REFERENCE_CHECKPOINT}" ]] || { echo "Missing A0_REFERENCE_CHECKPOINT=${A0_REFERENCE_CHECKPOINT}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}"
{
  printf 'Round1 Wave2 Progressive SFT LastRD-only on Server 1\n'
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'STAGE1_5_CHECKPOINT=%s\n' "${STAGE1_5_CHECKPOINT}"
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf 'A0_REFERENCE_CHECKPOINT=%s\n' "${A0_REFERENCE_CHECKPOINT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
  printf 'LAST_RD_NUM_WORKERS=%s\n' "${LAST_RD_NUM_WORKERS}"
  printf 'LAST_RD_PREFETCH_FACTOR=%s\n' "${LAST_RD_PREFETCH_FACTOR}"
  printf 'LAST_RD_PERSISTENT_WORKERS=%s\n' "${LAST_RD_PERSISTENT_WORKERS}"
  printf 'LAST_RD_RUNTIME_FINITE_CHECK=%s\n' "${LAST_RD_RUNTIME_FINITE_CHECK}"
  printf 'LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS=%s\n' "${LAST_RD_RUNTIME_FINITE_CHECK_MAX_ELEMENTS}"
} > "${OUTPUT_DIR}/commands.log"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  +experiment=last_rd_progressive_sft \
  cache_path="${TRAIN_CHUNK_CACHE_ROOT}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}" \
  ++agent.reference_a0_checkpoint="${A0_REFERENCE_CHECKPOINT}" \
  ++agent.last_rd_adapter_checkpoint="${STAGE1_5_CHECKPOINT}" \
  agent.use_expert_features=false \
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
  agent.risk_loss_weight=0.0 \
  agent.risk_loss_floor=0.0 \
  agent.policy_kd_loss_weight=0.05 \
  agent.policy_kd_mode=noise \
  trainer.params.max_epochs=200 \
  trainer.params.devices=8 \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers="${LAST_RD_NUM_WORKERS}" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor="${LAST_RD_PREFETCH_FACTOR}" \
  ++dataloader.params.persistent_workers="${LAST_RD_PERSISTENT_WORKERS}"
