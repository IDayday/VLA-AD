from __future__ import annotations

import torch

from navsim.agents.recogdrive.pareto_support.vector_scorer import ParetoVectorScorer, scorer_loss


def test_vector_scorer_forward_and_loss() -> None:
    model = ParetoVectorScorer(scene_dim=16, hidden_dim=32)
    traj = torch.zeros((4, 8, 3), dtype=torch.float32)
    scene = torch.randn(4, 16)
    outputs = model(scene, None, None, traj, torch.zeros(4, dtype=torch.long))
    assert outputs["pdms"].shape == (4,)
    assert torch.isfinite(outputs["pdms"]).all()
    targets = {
        "pdms": torch.ones(4),
        "ep": torch.ones(4),
        "ttc": torch.ones(4),
        "comfort": torch.ones(4),
        "ddc": torch.ones(4),
        "feas_cost": torch.zeros(4),
        "utility": torch.ones(4),
        "nc": torch.ones(4),
        "dac": torch.ones(4),
        "tlc": torch.ones(4),
        "pareto_front": torch.ones(4),
    }
    loss, parts = scorer_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert parts
