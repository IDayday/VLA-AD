from __future__ import annotations

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule


class _MockSlotBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.backbone_weight = nn.Parameter(torch.ones(1))
        self.hidden_dim = hidden_dim
        self.called = False

    def forward_with_two_expert_slots(self, images, prompt_inputs, two_expert_slots, **kwargs):
        self.called = True
        batch = images.shape[0]
        return {
            "raw_vlm_hidden": torch.randn(batch, 6, self.hidden_dim),
            "image_hidden": torch.randn(batch, 5, self.hidden_dim),
            "h_dyn": torch.randn(batch, 3, 12, self.hidden_dim),
            "h_geo": torch.randn(batch, 12, self.hidden_dim),
        }


def test_two_expert_stage1_loss_weights_and_no_dit_call():
    torch.manual_seed(21)
    module = TwoExpertVLMSFTModule(
        _MockSlotBackbone(),
        TwoExpertVLMSFTConfig(
            vlm_hidden_dim=16,
            adapter_dim=8,
            vggt_feature_dim=32,
            train_mode="frozen",
            dyn_loss_weight=1.0,
            geo_loss_weight=1.0,
            probe_traj_loss_weight=0.1,
            probe_heading_loss_weight=0.05,
            probe_progress_loss_weight=0.05,
            hidden_anchor_weight=0.05,
            random_mask_ratio=0.0,
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
    expected = (
        out["dyn_loss"]
        + out["geo_loss"]
        + 0.1 * out["probe_loss"]
        + 0.05 * out["probe_heading_loss"]
        + 0.05 * out["probe_progress_loss"]
    )

    assert module.backbone.called
    assert torch.allclose(out["loss"], expected, atol=1e-6)
    assert torch.isfinite(out["loss"])
    assert not hasattr(module, "dit")
    report = module.trainable_parameter_report()
    assert report["slots"] > 0
    assert report["adapters"] > 0
    assert report["probe"] > 0
    assert report["vlm"] == 0
