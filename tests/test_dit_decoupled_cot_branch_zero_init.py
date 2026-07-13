from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_dit import LightningDiT


def test_dit_cot_branch_zero_init_forward_and_gradients():
    torch.manual_seed(19)
    model = LightningDiT(num_heads=2, head_dim=16, output_dim=32, num_layers=1, dropout=0.0, attention_bias=True)
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    raw_context = torch.randn(2, 5, 32)
    cot_context = torch.randn(2, 6, 32)
    conditioning = torch.randn(2, 32)
    t = torch.tensor([1, 2])

    without_cot = model(hidden, raw_context, conditioning, t)
    with_cot = model(hidden, raw_context, conditioning, t, cot_condition_tokens=cot_context)
    assert torch.allclose(without_cot, with_cot, atol=1e-6, rtol=1e-6)

    with_cot.square().mean().backward()
    proj_grad = sum(block.cot_out_proj.weight.grad.abs().sum().item() for block in model.transformer_blocks)
    attn_grad = sum(block.cot_cross_attn.to_q.weight.grad.abs().sum().item() for block in model.transformer_blocks)
    assert proj_grad > 0.0
    assert attn_grad > 0.0
    assert not any("gate" in name and "cot" in name for name, _ in model.named_parameters())
