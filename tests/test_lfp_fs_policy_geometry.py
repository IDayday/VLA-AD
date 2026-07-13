import math

import pytest
import torch

from navsim.agents.recogdrive.stage3_lfp_grpo import LFPGRPOConfig
from navsim.agents.recogdrive.stage3_policy_geometry import (
    apply_group_credit_gate,
    apply_transition_std_floor,
    compute_group_trajectory_spread,
    endpoint_std_from_normalized_transition_floor,
    normalized_transition_floor_from_endpoint_std,
)
from navsim.agents.recogdrive.stage3_optimization import stable_gradient_norm


def test_fs_floor_reconstructs_requested_physical_endpoint_std() -> None:
    scale = torch.linspace(0.05, 0.40, steps=24).reshape(8, 3)
    requested = torch.tensor([0.8, 0.3, 0.04])
    normalized = normalized_transition_floor_from_endpoint_std(scale, requested)
    assert normalized.shape == (8, 3)
    reconstructed = endpoint_std_from_normalized_transition_floor(normalized, scale)
    torch.testing.assert_close(reconstructed, requested, rtol=1e-6, atol=1e-7)


def test_transition_floor_combines_scalar_and_representation_bounds() -> None:
    std = torch.zeros(2, 1, 1)
    representation_floor = torch.full((8, 3), 0.02)
    representation_floor[:, 0] = 0.08
    bounded = apply_transition_std_floor(std, 0.04, representation_floor)
    torch.testing.assert_close(bounded[..., 0], torch.full((2, 8), 0.08))
    torch.testing.assert_close(bounded[..., 1:], torch.full((2, 8, 2), 0.04))


def test_group_spread_uses_raw_xy_and_wraps_endpoint_heading() -> None:
    trajectories = torch.zeros(1, 2, 8, 3)
    trajectories[:, 1, :, 0] = 1.0
    trajectories[:, 0, -1, 2] = math.pi - 0.01
    trajectories[:, 1, -1, 2] = -math.pi + 0.01
    spread = compute_group_trajectory_spread(trajectories)
    torch.testing.assert_close(spread.pairwise_ade_m, torch.tensor([1.0]))
    torch.testing.assert_close(spread.endpoint_std_x_m, torch.tensor([0.5]))
    assert spread.endpoint_std_heading_rad.item() < 0.02


def test_low_information_gate_zeroes_only_unresolved_groups() -> None:
    advantages = torch.tensor([[1.0, -1.0], [0.5, -0.1]])
    output = apply_group_credit_gate(
        advantages,
        group_pairwise_ade_m=torch.tensor([0.01, 0.20]),
        group_scalar_span=torch.tensor([0.10, 0.10]),
        min_pairwise_ade_m=0.05,
        min_scalar_span=0.01,
        advantage_deadband=0.2,
    )
    torch.testing.assert_close(output.advantages[0], torch.zeros(2))
    torch.testing.assert_close(output.advantages[1], torch.tensor([0.5, 0.0]))
    assert output.active_group.tolist() == [False, True]
    assert output.diagnostics["lfp_low_spread_group_ratio"].item() == 0.5
    assert output.diagnostics["lfp_advantage_deadband_zero_ratio"].item() == 0.25


def test_disabled_gate_is_exact_identity() -> None:
    advantages = torch.tensor([[0.1, -0.2, 0.0]])
    output = apply_group_credit_gate(
        advantages,
        group_pairwise_ade_m=None,
        group_scalar_span=torch.zeros(1),
    )
    torch.testing.assert_close(output.advantages, advantages)
    assert output.active_group.all()


def test_enabled_fs_floor_requires_positive_physical_targets() -> None:
    with pytest.raises(ValueError, match="fs_endpoint_std_x_m"):
        LFPGRPOConfig(fs_transition_std_enabled=True).validate()
    LFPGRPOConfig(
        fs_transition_std_enabled=True,
        fs_endpoint_std_x_m=0.8,
        fs_endpoint_std_y_m=0.3,
        fs_endpoint_std_heading_rad=0.04,
    ).validate()


def test_gradient_norm_diagnostic_does_not_overflow_on_finite_gradients() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.full_like(parameter, 1e20)
    norm = stable_gradient_norm([parameter])
    assert torch.isfinite(norm)
    torch.testing.assert_close(norm, torch.tensor(2.0**0.5 * 1e20), rtol=1e-6, atol=0.0)
