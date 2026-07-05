from __future__ import annotations

import torch

from navsim.agents.recogdrive.pareto_support.bpdas import compute_pdas_metrics


def test_bpdas_metrics_finite_and_nonconstant_shape() -> None:
    rewards = torch.tensor([[0.7, 0.8, 0.4, 0.9], [0.2, 0.25, 0.3, 0.1]], dtype=torch.float32)
    trajs = torch.zeros((2, 4, 8, 3), dtype=torch.float32)
    trajs[:, :, :, 0] = torch.linspace(0.2, 4.0, 8)
    trajs[0, :, -1, 1] = torch.tensor([-1.0, 0.0, 1.0, 0.5])
    components = {
        "pdms": rewards,
        "ego_progress": rewards,
        "driving_direction_compliance": torch.ones_like(rewards),
        "feas_cost": torch.zeros_like(rewards),
    }
    ref = {
        "reward": torch.tensor([0.7, 0.2]),
        "pdms": torch.tensor([0.7, 0.2]),
        "ego_progress": torch.tensor([0.7, 0.2]),
        "driving_direction_compliance": torch.ones(2),
    }
    metrics = compute_pdas_metrics(rewards, components, trajs, ref)
    assert metrics.weight.shape == (2,)
    assert torch.isfinite(metrics.weight).all()
