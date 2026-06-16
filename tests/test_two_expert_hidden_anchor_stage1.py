from __future__ import annotations

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule


def test_hidden_anchor_loss_uses_frozen_raw_hidden():
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = nn.Parameter(torch.ones(1))

    module = TwoExpertVLMSFTModule(
        _Backbone(),
        TwoExpertVLMSFTConfig(vlm_hidden_dim=16, adapter_dim=8, vggt_feature_dim=32, train_mode="lora", hidden_anchor_weight=0.03),
    )
    current = torch.randn(2, 5, 16)
    frozen = current + 0.1
    loss = module._hidden_anchor_loss(current, frozen)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0
