#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

OUTPUT_DIR="${OUT_ROOT}/wave1_stage1_5/jepa_only"
export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"
{
  printf 'Round1 Wave1 Stage1.5 JEPA-only adapter on Server 2\n'
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
} > "${OUTPUT_DIR}/commands.log"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  +experiment=last_rd_stage1_5 \
  cache_path="${TRAIN_CHUNK_CACHE_ROOT}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  agent.use_expert_features=false \
  agent.use_jepa=true \
  agent.use_vggt=false \
  agent.allow_expert_target_features=true \
  agent.use_last_rd=true \
  agent.last_rd_stage=stage1_5 \
  agent.use_future_jepa_prediction=true \
  agent.use_vggt_geometry_tokens=false \
  agent.use_ego_trajectory_tokens=true \
  agent.use_risk_tokens=false \
  agent.diffusion_loss_weight=0.0 \
  agent.expert_alignment_weight=0.0 \
  agent.jepa_alignment_weight=0.0 \
  agent.vggt_alignment_weight=0.0 \
  agent.future_jepa_loss_weight=0.30 \
  agent.vggt_geometry_loss_weight=0.0 \
  agent.coarse_traj_loss_weight=0.50 \
  agent.coarse_heading_loss_weight=0.10 \
  agent.risk_loss_weight=0.0 \
  agent.policy_kd_loss_weight=0.0 \
  agent.freeze_base_action_head=true \
  agent.train_expert_only=true \
  agent.freeze_expert=false \
  trainer.params.max_epochs=20 \
  trainer.params.devices=8 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=8

python - <<'PY'
import os
from pathlib import Path
import torch

out = Path(os.environ["OUTPUT_DIR"])
candidates = [out / "latest.ckpt", out / "last.ckpt", *out.rglob("*.ckpt")]
source = next((path for path in candidates if path.is_file()), None)
if source is None:
    raise SystemExit("No checkpoint found for LaST-RD adapter extraction")
ckpt = torch.load(source, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
adapter = {
    k: v for k, v in state.items()
    if "action_head.last_rd." in k or (k.startswith("last_rd.") and not k.startswith("action_head."))
}
if not adapter:
    raise SystemExit(f"No LaST-RD adapter keys found in {source}")
target = out / "last_rd_adapter.pt"
torch.save({"state_dict": adapter, "source_checkpoint": str(source)}, target)
print(f"Saved {len(adapter)} LaST-RD adapter tensors to {target}")
PY

python "${REPO_ROOT}/scripts/audit_last_rd_adapter_checkpoint.py" \
  --checkpoint "${OUTPUT_DIR}/last_rd_adapter.pt" \
  --output "${OUTPUT_DIR}/adapter_audit.json"
