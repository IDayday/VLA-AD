#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:?Set BASE_CONFIG to the Hydra training config or agent override base.}"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT to the official-aligned chunk cache root.}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:?Set TRAIN_CHUNK_NAME_PATTERN, e.g. navtrain_chunk_*.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"

mkdir -p "${OUTPUT_DIR}"
{
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_CHUNK_NAME_PATTERN=%s\n' "${TRAIN_CHUNK_NAME_PATTERN}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
} > "${OUTPUT_DIR}/commands.log"

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
  agent.last_rd_stage=stage1_5 \
  agent.diffusion_loss_weight=0.0 \
  agent.future_jepa_loss_weight=0.30 \
  agent.vggt_geometry_loss_weight=0.10 \
  agent.coarse_traj_loss_weight=0.50 \
  agent.coarse_heading_loss_weight=0.10 \
  agent.risk_loss_weight=0.05 \
  agent.policy_kd_loss_weight=0.0 \
  agent.freeze_base_action_head=true \
  trainer.params.max_epochs=20 \
  trainer.params.devices=8 \
  trainer.params.strategy='ddp_find_unused_parameters_true' \
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
adapter = {k: v for k, v in state.items() if "action_head.last_rd." in k or ".last_rd." in k}
if not adapter:
    raise SystemExit(f"No LaST-RD adapter keys found in {source}")
torch.save({"state_dict": adapter, "source_checkpoint": str(source)}, out / "last_rd_adapter.pt")
print(f"Saved {len(adapter)} LaST-RD adapter tensors to {out / 'last_rd_adapter.pt'}")
PY
