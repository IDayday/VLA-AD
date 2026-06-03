import torch

from navsim.agents.recogdrive.risk_vla.dataclasses import StrategyWeights
from navsim.agents.recogdrive.risk_vla.risk_losses import (
    ensure_scene_level_labels,
    mvp_labels_to_extended,
    reduce_horizon_labels,
    risk_bce_loss,
    risk_focal_loss,
    strategy_activation_supervision_loss,
    strategy_entropy_loss,
)


def test_risk_losses_accept_scene_and_horizon_labels():
    logits = torch.zeros(2, 6)
    scene_labels = torch.tensor([[0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1]], dtype=torch.float32)
    horizon_labels = scene_labels.unsqueeze(1).expand(-1, 8, -1)

    assert risk_bce_loss(logits, scene_labels).ndim == 0
    assert risk_bce_loss(logits, horizon_labels).ndim == 0
    assert risk_focal_loss(logits, horizon_labels).ndim == 0
    assert ensure_scene_level_labels(horizon_labels).shape == (2, 6)
    assert reduce_horizon_labels(horizon_labels, mode="max").shape == (2, 6)


def test_mvp_label_conversion_works_for_scene_and_horizon():
    generic = torch.ones(2)
    drivable = torch.zeros(2)
    ttc = torch.full((2,), 0.5)
    comfort = torch.full((2,), 0.25)
    extended = mvp_labels_to_extended(generic, drivable, ttc, comfort)
    assert extended.shape == (2, 6)
    assert torch.allclose(extended[:, 2], ttc)
    assert torch.allclose(extended[:, 3], ttc)

    horizon = mvp_labels_to_extended(
        generic.unsqueeze(1).expand(-1, 8),
        drivable.unsqueeze(1).expand(-1, 8),
        ttc.unsqueeze(1).expand(-1, 8),
        comfort.unsqueeze(1).expand(-1, 8),
    )
    assert horizon.shape == (2, 8, 6)


def test_strategy_losses_accept_dataclass_weights():
    weights = StrategyWeights(
        base=torch.tensor([[0.5], [0.2]]),
        path_intent=torch.tensor([[0.2], [0.4]]),
        interaction=torch.tensor([[0.1], [0.2]]),
        progress=torch.tensor([[0.1], [0.1]]),
        comfort=torch.tensor([[0.1], [0.1]]),
        uncertainty=None,
        diagnostics={},
    )
    target = torch.full((2, 5), 0.2)
    assert strategy_entropy_loss(weights).ndim == 0
    assert strategy_activation_supervision_loss(weights, target).ndim == 0
