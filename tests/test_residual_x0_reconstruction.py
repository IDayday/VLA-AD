from __future__ import annotations

import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.recogdrive_diffusion_planner import TrainingTarget


def test_residual_x0_reconstructs_full_selected_representation() -> None:
    anchor = torch.randn(3, 8, 3)
    selected = torch.randn(3, 8, 3)
    alpha = 0.6
    residual = selected - alpha * anchor
    target = TrainingTarget(
        raw_trajectory=torch.randn(3, 8, 3),
        selected_repr=selected,
        diffusion_target_repr=residual,
        residual_anchor_repr=anchor,
        residual_alpha=alpha,
    )

    reconstructed = target.reconstruct_full_x0(residual)

    assert torch.allclose(reconstructed, selected, atol=1e-6)
    assert torch.nn.functional.l1_loss(reconstructed, target.selected_repr) < 1e-6


def test_non_residual_target_is_identity() -> None:
    prediction = torch.randn(2, 8, 3)
    target = TrainingTarget(
        raw_trajectory=torch.randn(2, 8, 3),
        selected_repr=torch.randn(2, 8, 3),
        diffusion_target_repr=torch.randn(2, 8, 3),
        residual_anchor_repr=None,
        residual_alpha=0.0,
    )
    assert target.reconstruct_full_x0(prediction) is prediction
