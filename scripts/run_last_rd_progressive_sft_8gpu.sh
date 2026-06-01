#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE1_5_CHECKPOINT="${STAGE1_5_CHECKPOINT:?Set STAGE1_5_CHECKPOINT to the adapter pretraining checkpoint.}"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
BASE_CONFIG="${BASE_CONFIG:-recogdrive_agent_a4_v2}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT:-}"

mkdir -p "${OUTPUT_DIR}"
{
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'STAGE1_5_CHECKPOINT=%s\n' "${STAGE1_5_CHECKPOINT}"
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf 'A0_REFERENCE_CHECKPOINT=%s\n' "${A0_REFERENCE_CHECKPOINT}"
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
} > "${OUTPUT_DIR}/commands.log"

CHECKPOINT_OVERRIDE=()
if [[ -n "${A0_INIT_CHECKPOINT}" ]]; then
  CHECKPOINT_OVERRIDE+=(agent.checkpoint_path="${A0_INIT_CHECKPOINT}")
fi
REFERENCE_OVERRIDE=()
if [[ -n "${A0_REFERENCE_CHECKPOINT}" ]]; then
  REFERENCE_OVERRIDE+=(agent.reference_a0_checkpoint="${A0_REFERENCE_CHECKPOINT}")
fi

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  +experiment="${BASE_CONFIG}" \
  cache_path="${TRAIN_CHUNK_CACHE_ROOT}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=true \
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
  agent.risk_loss_weight=0.03 \
  agent.risk_loss_floor=0.005 \
  agent.policy_kd_loss_weight=0.05 \
  agent.policy_kd_mode=noise \
  agent.last_rd_context_scale=1.0 \
  agent.last_rd_horizon_condition_scale=1.0 \
  trainer.params.max_epochs=200 \
  trainer.params.devices=8 \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=8 \
  agent.last_rd_adapter_checkpoint="${STAGE1_5_CHECKPOINT}" \
  "${CHECKPOINT_OVERRIDE[@]}" \
  "${REFERENCE_OVERRIDE[@]}"
