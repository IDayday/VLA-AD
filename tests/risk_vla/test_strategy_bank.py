import torch

from navsim.agents.recogdrive.risk_vla import (
    RiskConditionedStrategyBank,
    RiskState,
    StrategyWeights,
)


def test_strategy_bank_outputs_tokens_and_residuals_only():
    batch, tokens, dim, horizon = 2, 4, 16, 8
    bank = RiskConditionedStrategyBank(planner_dim=dim, action_horizon=horizon, tokens_per_strategy=2)
    risk_state = RiskState(
        risk_logits=torch.zeros(batch, 6),
        risk_probs=torch.full((batch, 6), 0.5),
        risk_embedding=torch.randn(batch, dim),
        uncertainty=None,
        diagnostics={},
    )
    weights = StrategyWeights(
        base=torch.full((batch, 1), 0.1),
        path_intent=torch.full((batch, 1), 0.8),
        interaction=torch.full((batch, 1), 0.2),
        progress=torch.full((batch, 1), 0.3),
        comfort=torch.full((batch, 1), 0.4),
        uncertainty=None,
        diagnostics={},
    )
    out = bank(
        vlm_tokens=torch.randn(batch, tokens, dim),
        risk_state=risk_state,
        strategy_weights=weights,
        status_feature=torch.randn(batch, 8),
        history_trajectory=torch.randn(batch, 4, 3),
        high_command_one_hot=torch.randn(batch, 3),
        bit_terminal=torch.randn(batch, 3),
        bit_path=torch.randn(batch, horizon, 3),
    )
    assert out.strategy_tokens.shape == (batch, 8, dim)
    assert out.horizon_residual.shape == (batch, horizon, dim)
    assert out.action_prior is not None
    assert out.action_prior.shape == (batch, horizon, 3)
    assert out.losses == {}
    assert "mean_weight_path_intent" in out.diagnostics


def test_strategy_bank_handles_missing_bit_inputs_with_zeros():
    batch, tokens, dim, horizon = 1, 2, 8, 8
    bank = RiskConditionedStrategyBank(planner_dim=dim, action_horizon=horizon, tokens_per_strategy=1)
    risk_state = RiskState(
        risk_logits=torch.zeros(batch, 6),
        risk_probs=torch.zeros(batch, 6),
        risk_embedding=torch.zeros(batch, dim),
        uncertainty=None,
        diagnostics={},
    )
    weights = StrategyWeights(
        base=torch.ones(batch, 1),
        path_intent=torch.zeros(batch, 1),
        interaction=torch.zeros(batch, 1),
        progress=torch.zeros(batch, 1),
        comfort=torch.zeros(batch, 1),
        uncertainty=None,
        diagnostics={},
    )
    out = bank(
        torch.randn(batch, tokens, dim),
        risk_state,
        weights,
        torch.randn(batch, 8),
        torch.randn(batch, 12),
        torch.randn(batch, 3),
    )
    assert out.strategy_tokens.shape == (batch, 4, dim)
    assert out.horizon_residual.shape == (batch, horizon, dim)
    assert torch.allclose(out.action_prior, torch.zeros_like(out.action_prior))
