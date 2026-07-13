#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule  # noqa: E402


class MockTwoExpertBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 32, *, with_lora: bool = False) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        if with_lora:
            self.lora_A = nn.Parameter(torch.randn(hidden_dim, 4) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(4, hidden_dim) * 0.01)
        self.hidden_dim = int(hidden_dim)

    def forward_with_two_expert_slots(self, images, prompt_inputs, two_expert_slots, **kwargs) -> Dict[str, torch.Tensor]:
        batch = int(images.shape[0])
        dyn_slots, geo_slots, _ = two_expert_slots.get_slots(batch, images.device, images.dtype)
        image_seed = images.reshape(batch, -1).mean(dim=1, keepdim=True).unsqueeze(-1)
        raw_seed = torch.arange(6, device=images.device, dtype=images.dtype).view(1, 6, 1)
        raw = image_seed + raw_seed.repeat(batch, 1, self.hidden_dim) / 100.0
        raw = self.layer(raw.float()).to(images.dtype)
        image_hidden = raw[:, :5]
        h_dyn = self.layer(dyn_slots.float()).to(images.dtype).view(batch, 3, 12, self.hidden_dim)
        h_geo = self.layer(geo_slots.float()).to(images.dtype)
        return {
            "raw_vlm_hidden": raw,
            "image_hidden": image_hidden,
            "h_dyn": h_dyn,
            "h_geo": h_geo,
        }


def make_batch(batch: int = 2, hidden_dim: int = 32, vggt_dim: int = 64) -> Dict[str, Any]:
    return {
        "images": torch.randn(batch, 3, 8, 8),
        "prompt_inputs": ["<image>\nPredict trajectory."] * batch,
        "status_feature": torch.randn(batch, 8),
        "high_command_one_hot": torch.eye(3)[:batch],
        "history_trajectory": torch.randn(batch, 4, 3),
        "trajectory": torch.randn(batch, 8, 3),
        "jepa_dynamic_teacher_tokens": torch.randn(batch, 3, 12, 1024),
        "vggt_feature23_tokens": torch.randn(batch, 12, vggt_dim),
    }


def run_smoke(train_mode: str = "frozen", vggt_dim: int = 64) -> Dict[str, Any]:
    torch.manual_seed(7)
    hidden_dim = 32
    module = TwoExpertVLMSFTModule(
        MockTwoExpertBackbone(hidden_dim, with_lora=(train_mode == "lora")),
        TwoExpertVLMSFTConfig(
            vlm_hidden_dim=hidden_dim,
            adapter_dim=16,
            vggt_feature_dim=int(vggt_dim),
            train_mode=train_mode,
            top_layers=1,
            random_mask_ratio=0.0,
            hidden_anchor_weight=0.0,
        ),
    )
    module.train()
    out = module(make_batch(hidden_dim=hidden_dim, vggt_dim=vggt_dim))
    out["loss"].backward()

    slot_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for param in module.two_expert_slots.parameters()
        if param.grad is not None
    )
    adapter_grad = sum(
        float(param.grad.detach().abs().sum().item())
        for name, param in module.named_parameters()
        if name.startswith(("dynamic_adapter.", "geometry_adapter.")) and param.grad is not None
    )
    report = module.trainable_parameter_report()
    result = {
        "ok": bool(
            torch.isfinite(out["loss"])
            and torch.isfinite(out["dyn_loss"])
            and torch.isfinite(out["geo_loss"])
            and torch.isfinite(out["probe_loss"])
            and slot_grad > 0.0
            and adapter_grad > 0.0
        ),
        "loss": float(out["loss"].detach().cpu()),
        "dyn_loss": float(out["dyn_loss"].detach().cpu()),
        "geo_loss": float(out["geo_loss"].detach().cpu()),
        "probe_loss": float(out["probe_loss"].detach().cpu()),
        "slot_grad_l1": slot_grad,
        "adapter_grad_l1": adapter_grad,
        "trainable_parameter_report": report,
        "train_mode": train_mode,
        "vggt_feature_dim": int(vggt_dim),
    }
    if train_mode == "frozen" and report["vlm_trainable_param_count"] != 0:
        result["ok"] = False
        result["error"] = "frozen mode unexpectedly has trainable VLM params"
    if train_mode in {"lora", "top_layers"} and report["vlm_trainable_param_count"] <= 0:
        result["ok"] = False
        result["error"] = f"{train_mode} mode has no trainable VLM params"
    return result


def main() -> int:
    train_mode = sys.argv[1] if len(sys.argv) > 1 else "frozen"
    result = run_smoke(train_mode=train_mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
