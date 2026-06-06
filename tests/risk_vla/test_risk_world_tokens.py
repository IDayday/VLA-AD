import torch

from navsim.agents.recogdrive.risk_vla.risk_world_tokens import RiskWorldTokenEncoder


def test_risk_world_token_encoder_outputs_v3_expert_tokens():
    encoder = RiskWorldTokenEncoder(planner_dim=16, hidden_dim=32)
    tokens, diagnostics = encoder(
        vlm_tokens=torch.randn(2, 5, 16),
        risk_probs=torch.rand(2, 6),
        status_feature=torch.randn(2, 8),
        history_trajectory=torch.randn(2, 4, 3),
        high_command_one_hot=torch.randn(2, 3),
    )
    assert tokens.shape == (2, 7, 16)
    assert "risk_world_token_uncertainty_norm" in diagnostics
    assert "risk_world_token_tail_risk_norm" in diagnostics
