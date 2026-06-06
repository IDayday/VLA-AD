import torch

from navsim.agents.recogdrive.risk_vla.future_latent_predictor import (
    FutureLatentPredictor,
    future_latent_smooth_l1_loss,
)


def test_future_latent_predictor_shapes_and_loss():
    predictor = FutureLatentPredictor(planner_dim=16, hidden_dim=32, future_steps=3)
    out = predictor(
        vlm_tokens=torch.randn(2, 5, 16),
        status_feature=torch.randn(2, 8),
        history_trajectory=torch.randn(2, 4, 3),
        high_command_one_hot=torch.randn(2, 3),
        risk_embedding=torch.randn(2, 16),
        world_tokens=torch.randn(2, 6, 16),
    )
    assert out.future_latents.shape == (2, 3, 16)
    assert "future_latent_norm" in out.diagnostics
    loss = future_latent_smooth_l1_loss(out.future_latents, torch.zeros(2, 3, 16), mask=torch.ones(2, 3))
    loss.backward()
    assert any(param.grad is not None for param in predictor.parameters())
