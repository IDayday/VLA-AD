from __future__ import annotations

import torch
from torch import nn

from navsim.agents.recogdrive.two_expert_vlm_sft import TwoExpertVLMSFTConfig, TwoExpertVLMSFTModule


def test_contrastive_loss_batch_and_skip_paths():
    module = TwoExpertVLMSFTModule(
        nn.Module(),
        TwoExpertVLMSFTConfig(vlm_hidden_dim=16, adapter_dim=8, vggt_feature_dim=32, train_mode="frozen"),
    )
    pred = torch.randn(3, 3, 12, 32)
    target = torch.randn(3, 3, 12, 32)
    loss, diag = module._contrastive_loss(pred, target, kind="dyn")
    assert torch.isfinite(loss)
    assert float(diag["contrastive_dyn_skipped"]) == 0.0

    skipped, diag = module._contrastive_loss(pred[:1], target[:1], kind="dyn")
    assert float(skipped) == 0.0
    assert float(diag["contrastive_dyn_skipped"]) == 1.0
