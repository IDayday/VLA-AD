from __future__ import annotations

import torch

from navsim.agents.recogdrive.last_vla_teacher_adapters import RandomTokenMask


def test_strict_last_vla_random_mask_only_active_in_training():
    torch.manual_seed(7)
    mask = RandomTokenMask(mask_ratio=0.5, token_dim=4)
    tokens = torch.randn(3, 10, 4)

    mask.train()
    masked, info = mask(tokens)
    assert masked.shape == tokens.shape
    assert info["actual_mask_ratio"] > 0
    assert not torch.allclose(masked, tokens)

    mask.eval()
    unmasked, eval_info = mask(tokens)
    assert torch.allclose(unmasked, tokens)
    assert eval_info["actual_mask_ratio"] == 0
