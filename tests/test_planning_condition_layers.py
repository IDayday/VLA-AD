from __future__ import annotations

import torch

from navsim.agents.recogdrive.recogdrive_dit import LightningDiT


def _model() -> LightningDiT:
    model = LightningDiT(
        num_heads=2,
        head_dim=16,
        output_dim=32,
        num_layers=4,
        dropout=0.0,
        attention_bias=True,
        interleave_attention=True,
    )
    model.set_planning_gate_init(0.05)
    return model


def test_planning_condition_defaults_to_cross_attention_blocks(monkeypatch) -> None:
    torch.manual_seed(23)
    model = _model()
    calls = [0, 0, 0, 0]
    for index, block in enumerate(model.transformer_blocks):
        original = block.planning_cross_attn.forward

        def wrapped(*args, _index=index, _original=original, **kwargs):
            calls[_index] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(block.planning_cross_attn, "forward", wrapped)

    model(
        torch.randn(2, 8, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 32),
        torch.tensor([1, 2]),
        planning_condition_tokens=torch.randn(2, 6, 32),
        planning_condition_layers="cross_attention",
    )

    assert calls == [0, 1, 0, 1]
    assert torch.isclose(model.last_planning_layer_gate_mean, torch.tensor(0.05), atol=1e-6)
    assert all(
        torch.isclose(torch.sigmoid(block.planning_gate_logit), torch.tensor(0.05), atol=1e-6)
        for block in model.transformer_blocks
    )


def test_planning_condition_all_mode_is_available(monkeypatch) -> None:
    model = _model()
    calls = [0, 0, 0, 0]
    for index, block in enumerate(model.transformer_blocks):
        original = block.planning_cross_attn.forward

        def wrapped(*args, _index=index, _original=original, **kwargs):
            calls[_index] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(block.planning_cross_attn, "forward", wrapped)

    model(
        torch.randn(1, 8, 32),
        torch.randn(1, 5, 32),
        torch.randn(1, 32),
        torch.tensor([1]),
        planning_condition_tokens=torch.randn(1, 6, 32),
        planning_condition_layers="all",
    )
    assert calls == [1, 1, 1, 1]


def test_planning_tokens_never_enter_legacy_cot_branch(monkeypatch) -> None:
    model = _model()

    def unexpected_cot_call(*args, **kwargs):
        raise AssertionError("PTA called the legacy Last-VLA CoT branch")

    for block in model.transformer_blocks:
        monkeypatch.setattr(block.cot_cross_attn, "forward", unexpected_cot_call)

    model(
        torch.randn(1, 8, 32),
        torch.randn(1, 5, 32),
        torch.randn(1, 32),
        torch.tensor([1]),
        planning_condition_tokens=torch.randn(1, 6, 32),
    )
