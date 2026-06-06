import torch

from navsim.agents.recogdrive.risk_vla.risk_world_tokens import RiskWorldTokenEncoder


def test_world_tokens_include_required_expert_slots():
    encoder = RiskWorldTokenEncoder(planner_dim=16, hidden_dim=32)
    tokens, diagnostics = encoder(
        vlm_tokens=torch.randn(2, 4, 16),
        risk_probs=torch.rand(2, 6),
        status_feature=torch.randn(2, 8),
        history_trajectory=torch.randn(2, 4, 3),
        high_command_one_hot=torch.randn(2, 3),
        candidate_summary=torch.randn(2, 16),
    )
    assert tokens.shape == (2, 7, 16)
    assert RiskWorldTokenEncoder.TOKEN_NAMES == [
        "semantic_interaction",
        "drivable_geometry",
        "dynamic_ttc",
        "ego_progress",
        "comfort",
        "uncertainty",
        "tail_risk",
    ]
    for name in RiskWorldTokenEncoder.TOKEN_NAMES:
        assert f"risk_world_token_{name}_norm" in diagnostics
