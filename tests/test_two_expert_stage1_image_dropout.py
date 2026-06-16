from __future__ import annotations

import torch

from navsim.agents.recogdrive.two_expert_adapters import ImageHiddenDropout


def test_image_hidden_dropout_training_only():
    tokens = torch.randn(2, 4, 8)
    dropout = ImageHiddenDropout(dropout_prob=1.0, token_dim=8, use_learned_mask_token=False)
    dropout.eval()
    out_eval, info_eval = dropout(tokens)
    assert torch.allclose(out_eval, tokens)
    assert float(info_eval["image_dropout_active"]) == 0.0

    dropout.train()
    out_train, info_train = dropout(tokens)
    assert torch.allclose(out_train, torch.zeros_like(tokens))
    assert float(info_train["image_dropout_active"]) == 1.0
