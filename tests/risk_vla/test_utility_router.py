import torch

from navsim.agents.recogdrive.risk_vla.dataclasses import RiskState
from navsim.agents.recogdrive.risk_vla.utility_router import RiskVLAv2UtilityRouter


def _risk_state():
    return RiskState(
        risk_logits=torch.zeros(2, 6),
        risk_probs=torch.tensor([[0.1, 0.8, 0.1, 0.1, 0.2, 0.1], [0.2, 0.1, 0.9, 0.8, 0.1, 0.1]]),
        risk_embedding=torch.randn(2, 16),
        uncertainty=None,
        diagnostics={},
    )


def test_constrained_utility_router_suppresses_unsafe_candidate():
    router = RiskVLAv2UtilityRouter(planner_dim=16)
    out = router(
        _risk_state(),
        candidate_utility=torch.tensor([[1.0, 1.1], [1.0, 1.5]]),
        candidate_nc_risk=torch.tensor([[0.0, 0.9], [0.0, 1.0]]),
        candidate_ttc_risk=torch.zeros(2, 2),
        mode="scorer_select",
    )
    assert out.selected_candidate_id.tolist() == [0, 0]
    assert out.candidate_weights.shape == (2, 2)
