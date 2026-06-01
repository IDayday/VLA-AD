#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-last_rd_stage1_5}"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT to the official-aligned chunk cache root.}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:?Set TRAIN_CHUNK_NAME_PATTERN, e.g. navtrain_chunk_*.}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
LAST_RD_RISK_LOSS_WEIGHT="${LAST_RD_RISK_LOSS_WEIGHT:-0.0}"
LAST_RD_CACHE_MANIFEST="${LAST_RD_CACHE_MANIFEST:-}"
ALLOW_MISSING_RISK_LABELS="${ALLOW_MISSING_RISK_LABELS:-0}"
RISK_LOSS_ENABLED="$(python - "${LAST_RD_RISK_LOSS_WEIGHT}" <<'PY'
import sys
print("1" if float(sys.argv[1]) > 0.0 else "0")
PY
)"

if [[ "${RISK_LOSS_ENABLED}" == "1" ]]; then
  if [[ -z "${LAST_RD_CACHE_MANIFEST}" ]]; then
    echo "LAST_RD_RISK_LOSS_WEIGHT=${LAST_RD_RISK_LOSS_WEIGHT} requires LAST_RD_CACHE_MANIFEST unless ALLOW_MISSING_RISK_LABELS=1." >&2
    [[ "${ALLOW_MISSING_RISK_LABELS}" == "1" ]] || exit 1
  elif [[ "${ALLOW_MISSING_RISK_LABELS}" != "1" ]]; then
    python - "${LAST_RD_CACHE_MANIFEST}" <<'PY'
import json
import sys
path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
risk = data.get("risk_label_coverage", {})
if not any(float(item.get("coverage", 0.0)) > 0.0 for item in risk.values() if isinstance(item, dict)):
    raise SystemExit("Risk loss requested but cache manifest reports zero risk label coverage. Set ALLOW_MISSING_RISK_LABELS=1 to override.")
PY
  fi
fi

mkdir -p "${OUTPUT_DIR}"
{
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'BASE_CONFIG is a Hydra experiment config name under navsim/planning/script/config/experiment.\n'
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_CHUNK_NAME_PATTERN=%s\n' "${TRAIN_CHUNK_NAME_PATTERN}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'LAST_RD_RISK_LOSS_WEIGHT=%s\n' "${LAST_RD_RISK_LOSS_WEIGHT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'Recommended preflight: python scripts/audit_last_rd_cache_manifest.py --cache-root %s --output reports/last_rd_cache_manifest.json\n' "${TRAIN_CHUNK_CACHE_ROOT}"
} > "${OUTPUT_DIR}/commands.log"

torchrun --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  +experiment="${BASE_CONFIG}" \
  cache_path="${TRAIN_CHUNK_CACHE_ROOT}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  output_dir="${OUTPUT_DIR}" \
  use_cache_without_dataset=true \
  agent.use_expert_features=false \
  agent.use_jepa=true \
  agent.use_vggt=true \
  agent.allow_expert_target_features=true \
  agent.use_last_rd=true \
  agent.last_rd_stage=stage1_5 \
  agent.diffusion_loss_weight=0.0 \
  agent.expert_alignment_weight=0.0 \
  agent.jepa_alignment_weight=0.0 \
  agent.vggt_alignment_weight=0.0 \
  agent.future_jepa_loss_weight=0.30 \
  agent.vggt_geometry_loss_weight=0.10 \
  agent.coarse_traj_loss_weight=0.50 \
  agent.coarse_heading_loss_weight=0.10 \
  agent.risk_loss_weight="${LAST_RD_RISK_LOSS_WEIGHT}" \
  agent.policy_kd_loss_weight=0.0 \
  agent.freeze_base_action_head=true \
  agent.train_expert_only=true \
  agent.freeze_expert=false \
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
include_legacy = os.environ.get("LAST_RD_SAVE_LEGACY_A4", "0") == "1"
adapter = {
    k: v for k, v in state.items()
    if "action_head.last_rd." in k or (k.startswith("last_rd.") and not k.startswith("action_head."))
}
if include_legacy:
    legacy_markers = ("jepa_", "vggt_", "branch_logits")
    adapter.update({k: v for k, v in state.items() if any(marker in k for marker in legacy_markers)})
if not adapter:
    raise SystemExit(f"No LaST-RD adapter keys found in {source}")
torch.save({"state_dict": adapter, "source_checkpoint": str(source)}, out / "last_rd_adapter.pt")
print(f"Saved {len(adapter)} LaST-RD adapter tensors to {out / 'last_rd_adapter.pt'}")
PY
