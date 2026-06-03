import torch

from navsim.agents.recogdrive.risk_vla import RiskStateEncoder


def test_risk_state_encoder_cpu_import_and_shapes_without_bit_inputs():
    batch, tokens, dim = 2, 5, 16
    encoder = RiskStateEncoder(planner_dim=dim, hidden_dim=32, action_horizon=8)
    vlm_tokens = torch.randn(batch, tokens, dim)
    status = torch.randn(batch, 8)
    history = torch.randn(batch, 4, 3)
    command = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    risk_state = encoder(vlm_tokens, status, history, command)

    assert risk_state.risk_logits.shape == (batch, 6)
    assert risk_state.risk_probs.shape == (batch, 6)
    assert risk_state.risk_embedding.shape == (batch, dim)
    assert risk_state.uncertainty is not None
    assert risk_state.uncertainty.shape == (batch, 1)
    assert torch.all(risk_state.risk_probs >= 0)
    assert torch.all(risk_state.risk_probs <= 1)
    assert "risk_prob_path_dac" in risk_state.diagnostics
    assert "risk_embedding_norm" in risk_state.diagnostics


def test_risk_state_encoder_accepts_flat_history_and_bit_summary():
    batch, tokens, dim = 2, 3, 12
    encoder = RiskStateEncoder(planner_dim=dim, hidden_dim=24, action_horizon=8)
    out = encoder(
        torch.randn(batch, tokens, dim),
        torch.randn(batch, 8),
        torch.randn(batch, 12),
        torch.randn(batch, 3),
        bit_terminal=torch.randn(batch, 3),
        bit_path=torch.randn(batch, 8, 3),
    )
    assert out.risk_logits.shape == (batch, 6)
    assert out.risk_embedding.dtype == torch.float32
