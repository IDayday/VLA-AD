from __future__ import annotations

import math

import torch

from navsim.agents.recogdrive.reference_relative_geometry import (
    compute_reference_relative_geometry_components,
    compute_reference_relative_geometry_loss,
)


def _straight(batch: int = 1) -> torch.Tensor:
    trajectory = torch.zeros(batch, 8, 3)
    trajectory[..., 0] = torch.arange(1, 9).float()
    return trajectory


def test_reference_relative_loss_zero_for_match_and_margin() -> None:
    target = _straight()
    assert compute_reference_relative_geometry_loss(target, target, 0.08, 0.05, 0.2) == 0

    within_margin = target.clone()
    within_margin[..., 2] = 0.04
    assert compute_reference_relative_geometry_loss(within_margin, target, 0.08, 0.05, 0.2) == 0


def test_tangent_and_curvature_excess_are_positive() -> None:
    target = _straight()
    tangent_bad = target.clone()
    tangent_bad[..., 2] = 0.4
    tangent_loss, _ = compute_reference_relative_geometry_components(tangent_bad, target, 0.08, 0.05, 0.2)
    assert tangent_loss.item() > 0.0

    curvature_bad = target.clone()
    curvature_bad[0, :, :2] = torch.tensor(
        [[1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6]],
        dtype=torch.float32,
    )
    curvature_bad[0, 2:, 2] = math.pi / 2
    _, curvature_loss = compute_reference_relative_geometry_components(curvature_bad, target, 0.08, 0.05, 0.2)
    assert curvature_loss.item() > 0.0


def test_stationary_segments_are_finite_and_target_is_detached() -> None:
    stationary_pred = torch.zeros(2, 8, 3, requires_grad=True)
    stationary_target = torch.zeros(2, 8, 3, requires_grad=True)
    loss = compute_reference_relative_geometry_loss(stationary_pred, stationary_target, 0.08, 0.05, 0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert stationary_pred.grad is not None
    assert torch.isfinite(stationary_pred.grad).all()
    assert stationary_target.grad is None
