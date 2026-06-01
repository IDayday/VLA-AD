#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from transformers.feature_extraction_utils import BatchFeature
except ModuleNotFoundError:
    from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

    _install_dependency_stubs()
    from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.expert_cache import iter_index, load_sample
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    DDIMConfig,
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


def build_planner(stage: str, diffusion_loss_weight: float) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 4,
            "head_dim": 96,
            "num_layers": 2,
            "output_dim": 384,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=384,
        planner_dim=384,
        hidden_size=384,
        action_dim=3,
        action_horizon=8,
        max_seq_len=8,
        sampling_method="ddim",
        num_inference_steps=2,
        vlm_size="small",
        ddim_cfg=DDIMConfig(num_train_timesteps=10),
        use_expert_features=True,
        use_jepa=True,
        use_vggt=True,
        expert_dropout=0.0,
        expert_stream_dropout=0.0,
        jepa_alignment_weight=0.0,
        vggt_alignment_weight=0.0,
        use_last_rd=True,
        last_rd_stage=stage,
        diffusion_loss_weight=diffusion_loss_weight,
        future_jepa_loss_weight=0.30,
        vggt_geometry_loss_weight=0.10,
        coarse_traj_loss_weight=0.50,
        coarse_heading_loss_weight=0.10,
        risk_loss_weight=0.0,
    )
    return ReCogDriveDiffusionPlanner(cfg)


def synthetic_sample() -> Dict[str, torch.Tensor]:
    return {
        "history_trajectory": torch.randn(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "last_hidden_state": torch.randn(16, 1536),
        "status_feature": torch.randn(8),
        "trajectory": torch.randn(8, 3),
        "jepa_context_tokens": torch.randn(12, 1024),
        "jepa_target_tokens": torch.randn(12, 1024),
        "vggt_context_tokens": torch.randn(12, 2048),
        "vggt_target_tokens": torch.randn(12, 2048),
    }


def load_one_chunk_sample(cache_root: Optional[str]) -> Dict[str, Any]:
    if not cache_root:
        return synthetic_sample()
    root = Path(cache_root)
    chunks = [root] if (root / "index.jsonl").is_file() else sorted(path for path in root.iterdir() if (path / "index.jsonl").is_file())
    for chunk in chunks:
        for record in iter_index(chunk):
            path = Path(record["path"])
            if not path.is_file() and not path.is_absolute():
                path = chunk / path
            sample = load_sample(path)
            required = {
                "history_trajectory",
                "high_command_one_hot",
                "last_hidden_state",
                "status_feature",
                "trajectory",
                "jepa_context_tokens",
                "vggt_context_tokens",
            }
            if required.issubset(sample):
                return sample
    return synthetic_sample()


def make_batch(sample: Dict[str, Any], include_targets: bool) -> tuple[torch.Tensor, BatchFeature]:
    vl = sample["last_hidden_state"].float().unsqueeze(0)
    data = {
        "his_traj": sample["history_trajectory"].float().view(1, -1),
        "history_trajectory": sample["history_trajectory"].float().unsqueeze(0),
        "status_feature": sample["status_feature"].float().unsqueeze(0),
        "high_command_one_hot": sample["high_command_one_hot"].float().unsqueeze(0),
        "jepa_context_tokens": sample["jepa_context_tokens"].float().unsqueeze(0),
        "vggt_context_tokens": sample["vggt_context_tokens"].float().unsqueeze(0),
    }
    for key in ("vggt_geometry_tokens", "risk_labels"):
        if key in sample:
            data[key] = sample[key].float().unsqueeze(0)
    if include_targets:
        data["action"] = sample["trajectory"].float().unsqueeze(0)
        for key in ("jepa_target_tokens", "vggt_target_tokens", "vggt_geometry_target_tokens"):
            if key in sample:
                data[key] = sample[key].float().unsqueeze(0)
    return vl, BatchFeature(data=data)


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().float().item())


def main() -> int:
    torch.manual_seed(19)
    sample = load_one_chunk_sample(os.environ.get("TRAIN_CHUNK_CACHE_ROOT") or os.environ.get("CHUNK_CACHE_ROOT"))
    report: Dict[str, Any] = {"source": "chunk_or_synthetic"}

    vl_train, action_train = make_batch(sample, include_targets=True)
    stage1_5 = build_planner("stage1_5", 0.0)
    stage1_5.train()
    out_stage1_5 = stage1_5(vl_train, action_train)
    report["stage1_5"] = {key: scalar(out_stage1_5[key]) for key in ("loss", "future_jepa_loss", "vggt_geometry_loss", "coarse_traj_loss")}

    progressive = build_planner("progressive_sft", 1.0)
    progressive.train()
    out_progressive = progressive(vl_train, action_train)
    report["progressive_sft"] = {key: scalar(out_progressive[key]) for key in ("loss", "diffusion_loss", "future_jepa_loss", "coarse_traj_loss")}

    progressive.eval()
    vl_eval, action_eval = make_batch(sample, include_targets=False)
    with torch.no_grad():
        pred = progressive.get_action(vl_eval, action_eval, init_actions=torch.zeros(1, 8, 3), deterministic=True)
    report["get_action"] = {
        "pred_traj_shape": list(pred["pred_traj"].shape),
        "pred_traj_finite": bool(torch.isfinite(pred["pred_traj"]).all().item()),
        "target_tokens_passed": False,
    }

    out_path = REPO_ROOT / "reports" / "last_rd_smoke_forward.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
