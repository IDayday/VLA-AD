from __future__ import annotations

import inspect

import torch

from navsim.agents.recogdrive.planning_token_adapter import (
    PlanningTokenAdapter,
    PlanningTokenAdapterConfig,
)


def _inputs(batch_size: int = 4, dim: int = 384):
    return (
        torch.randn(batch_size, 7, dim),
        torch.randn(batch_size, 8),
        torch.eye(3)[torch.arange(batch_size) % 3].float(),
        torch.randn(batch_size, 4, 3),
    )


def test_planning_token_adapter_shape_and_input_contract() -> None:
    torch.manual_seed(11)
    adapter = PlanningTokenAdapter(PlanningTokenAdapterConfig(condition_dropout=0.0)).eval()
    tokens, diagnostics = adapter(*_inputs())

    assert tokens.shape == (4, 16, 384)
    assert all(value.ndim == 0 for value in diagnostics.values())
    forbidden = {"target", "action", "noisy_action", "timestep", "support"}
    parameters = set(inspect.signature(adapter.forward).parameters)
    assert not any(any(part in name for part in forbidden) for name in parameters)


def test_planning_condition_dropout_drops_whole_samples() -> None:
    torch.manual_seed(17)
    adapter = PlanningTokenAdapter(
        PlanningTokenAdapterConfig(planner_dim=32, hidden_dim=64, num_tokens=8, num_heads=4, condition_dropout=0.5)
    ).train()
    tokens, diagnostics = adapter(*_inputs(batch_size=64, dim=32))
    per_sample_nonzero = tokens.abs().sum(dim=(1, 2)) > 0

    assert per_sample_nonzero.any()
    assert (~per_sample_nonzero).any()
    assert torch.all(tokens[~per_sample_nonzero] == 0)
    assert torch.isclose(
        diagnostics["planning_condition_keep_ratio"],
        per_sample_nonzero.float().mean(),
    )


def test_planning_adapter_validates_heads() -> None:
    try:
        PlanningTokenAdapter(PlanningTokenAdapterConfig(planner_dim=30, num_heads=8))
    except ValueError as exc:
        assert "divide" in str(exc)
    else:
        raise AssertionError("Expected invalid planner_dim/num_heads to fail.")
