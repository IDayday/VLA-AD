import torch

from navsim.agents.recogdrive.risk_vla.trajectory_risk_critic import TrajectoryRiskCritic


def test_trajectory_risk_critic_shapes_and_gradients():
    critic = TrajectoryRiskCritic(planner_dim=16, hidden_dim=32, num_risk_classes=6, horizon=8)
    out = critic(
        vlm_tokens=torch.randn(2, 4, 16),
        status_feature=torch.randn(2, 8),
        history_trajectory=torch.randn(2, 4, 3),
        high_command_one_hot=torch.randn(2, 3),
        candidate_trajectories=torch.randn(2, 3, 8, 3),
        risk_embedding=torch.randn(2, 16),
    )
    assert out.utility_score.shape == (2, 3)
    assert out.risk_logits_scene.shape == (2, 3, 6)
    assert out.risk_logits_horizon.shape == (2, 3, 8, 6)
    assert out.submetric_delta_pred.shape == (2, 3, 6)
    out.utility_score.mean().backward()
    assert any(param.grad is not None for param in critic.parameters())
