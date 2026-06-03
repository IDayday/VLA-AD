import torch

from navsim.agents.recogdrive.risk_vla import RiskConditionedStrategyRouter, RiskState


def make_state(probs):
    risk_probs = torch.tensor(probs, dtype=torch.float32)
    return RiskState(
        risk_logits=torch.logit(risk_probs.clamp(1e-4, 1 - 1e-4)),
        risk_probs=risk_probs,
        risk_embedding=torch.randn(risk_probs.shape[0], 16),
        uncertainty=None,
        diagnostics={},
    )


def test_independent_router_weights_are_bounded_and_interpretable():
    router = RiskConditionedStrategyRouter(planner_dim=16, mode="independent")
    low_path = make_state([[0.1, 0.1, 0.0, 0.0, 0.2, 0.2]])
    high_path = make_state([[0.1, 0.9, 0.0, 0.0, 0.2, 0.2]])
    low_out = router(low_path)
    high_out = router(high_path)

    for tensor in (high_out.base, high_out.path_intent, high_out.interaction, high_out.progress, high_out.comfort):
        assert torch.all(tensor >= 0)
        assert torch.all(tensor <= 1)
    assert high_out.path_intent.item() > low_out.path_intent.item()
    assert "strategy_entropy" in high_out.diagnostics


def test_safety_risk_increases_interaction_and_suppresses_path_progress():
    router = RiskConditionedStrategyRouter(planner_dim=16, mode="independent")
    low_safety = make_state([[0.1, 0.8, 0.0, 0.0, 0.8, 0.2]])
    high_safety = make_state([[0.1, 0.8, 0.9, 0.8, 0.8, 0.2]])
    low = router(low_safety)
    high = router(high_safety)

    assert high.interaction.item() > low.interaction.item()
    assert high.path_intent.item() < low.path_intent.item()
    assert high.progress.item() < low.progress.item()


def test_learned_softmax_router_outputs_strategy_distribution():
    router = RiskConditionedStrategyRouter(planner_dim=16, mode="learned_softmax")
    out = router(make_state([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]))
    total = out.base + out.path_intent + out.interaction + out.progress + out.comfort
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
