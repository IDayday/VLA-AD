from __future__ import annotations

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule


class _ReplayBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(1))
        self.hidden_dim = hidden_dim
        self.replay_called = False

    def forward_with_two_expert_slots(self, images, prompt_inputs, two_expert_slots, **kwargs):
        batch = images.shape[0]
        return {
            "raw_vlm_hidden": torch.randn(batch, 6, self.hidden_dim),
            "image_hidden": torch.randn(batch, 5, self.hidden_dim),
            "h_dyn": torch.randn(batch, 3, 12, self.hidden_dim),
            "h_geo": torch.randn(batch, 12, self.hidden_dim),
        }

    def forward_replay_ce(self, images, replay_inputs, **kwargs):
        self.replay_called = True
        return {
            "loss": self.lora_A.sum() * 0.0 + torch.tensor(0.25),
            "token_count": torch.tensor(12.0),
            "answer_length_mean": torch.tensor(40.0),
        }


def test_stage1_replay_ce_contributes_to_total_loss():
    module = TwoExpertVLMSFTModule(
        _ReplayBackbone(),
        TwoExpertVLMSFTConfig(
            vlm_hidden_dim=16,
            adapter_dim=8,
            vggt_feature_dim=32,
            train_mode="lora",
            recogdrive_replay_ce_loss_weight=0.2,
            random_mask_ratio=0.0,
            hidden_anchor_weight=0.0,
        ),
    )
    batch = {
        "images": torch.randn(2, 3),
        "prompt_inputs": ["q", "q"],
        "replay_prompt_inputs": {"prompts": ["p", "p"], "answers": ["a", "a"], "num_patches_list": [1, 1]},
        "replay_parse_ok": torch.ones(2),
        "replay_official_recogdrive_stage1": torch.ones(2),
        "status_feature": torch.randn(2, 8),
        "high_command_one_hot": torch.eye(3)[:2],
        "history_trajectory": torch.randn(2, 4, 3),
        "trajectory": torch.randn(2, 8, 3),
        "jepa_dynamic_teacher_tokens": torch.randn(2, 3, 12, 1024),
        "vggt_feature23_tokens": torch.randn(2, 12, 32),
    }
    out = module(batch)
    assert module.backbone.replay_called
    expected_without_replay = out["loss"] - 0.2 * out["recogdrive_replay_ce_loss"]
    assert torch.isfinite(expected_without_replay)
    assert torch.allclose(out["recogdrive_replay_ce_loss"], torch.tensor(0.25), atol=1e-6)
