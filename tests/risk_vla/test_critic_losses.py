import torch

from navsim.agents.recogdrive.risk_vla.critic_losses import (
    cvar_tail_risk_loss,
    critic_total_loss,
    pairwise_preference_loss,
)


def test_pairwise_preference_loss_prefers_positive_margin():
    utility = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    loss = pairwise_preference_loss(utility, torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert loss.item() < 0.25


def test_cvar_tail_risk_uses_worst_quantile():
    risk = torch.tensor([[0.1, 0.2, 0.9, 0.8]])
    assert torch.isclose(cvar_tail_risk_loss(risk, q=0.5), torch.tensor(0.85))


def test_critic_total_loss_returns_named_terms():
    losses = critic_total_loss(
        utility_score=torch.randn(2, 3),
        risk_logits_scene=torch.randn(2, 3, 6),
        risk_logits_horizon=torch.randn(2, 3, 8, 6),
        submetric_delta_pred=torch.randn(2, 3, 6),
        scene_labels=torch.zeros(2, 3, 6),
        horizon_labels=torch.zeros(2, 3, 8, 6),
        submetric_delta_target=torch.zeros(2, 3, 6),
        positive_index=torch.zeros(2, dtype=torch.long),
        negative_index=torch.ones(2, dtype=torch.long),
    )
    assert "total_loss" in losses
    assert losses["total_loss"].ndim == 0
