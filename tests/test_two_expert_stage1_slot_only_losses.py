from __future__ import annotations

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule


class _Backbone(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward_with_two_expert_slots(self, images, prompt_inputs, two_expert_slots, **kwargs):
        batch = images.shape[0]
        return {
            "raw_vlm_hidden": torch.randn(batch, 6, self.hidden_dim),
            "image_hidden": torch.randn(batch, 5, self.hidden_dim),
            "h_dyn": torch.randn(batch, 3, 12, self.hidden_dim),
            "h_geo": torch.randn(batch, 12, self.hidden_dim),
        }


def test_slot_only_losses_are_reported_and_finite():
    module = TwoExpertVLMSFTModule(
        _Backbone(),
        TwoExpertVLMSFTConfig(
            vlm_hidden_dim=16,
            adapter_dim=8,
            vggt_feature_dim=32,
            train_mode="frozen",
            slot_only_dyn_loss_weight=0.2,
            slot_only_geo_loss_weight=0.2,
            random_mask_ratio=0.0,
            hidden_anchor_weight=0.0,
        ),
    )
    batch = {
        "images": torch.randn(2, 3),
        "prompt_inputs": ["q", "q"],
        "status_feature": torch.randn(2, 8),
        "high_command_one_hot": torch.eye(3)[:2],
        "history_trajectory": torch.randn(2, 4, 3),
        "trajectory": torch.randn(2, 8, 3),
        "jepa_dynamic_teacher_tokens": torch.randn(2, 3, 12, 1024),
        "vggt_feature23_tokens": torch.randn(2, 12, 32),
    }
    out = module(batch)
    assert torch.isfinite(out["slot_only_dyn_loss"])
    assert torch.isfinite(out["slot_only_geo_loss"])
