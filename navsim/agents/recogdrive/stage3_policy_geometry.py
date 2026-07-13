from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class GroupTrajectorySpread:
    pairwise_ade_m: torch.Tensor
    endpoint_std_x_m: torch.Tensor
    endpoint_std_y_m: torch.Tensor
    endpoint_std_heading_rad: torch.Tensor

    def scalar_diagnostics(self) -> Dict[str, torch.Tensor]:
        return {
            "lfp_group_pairwise_ade_m_mean": self.pairwise_ade_m.mean().detach(),
            "lfp_group_pairwise_ade_m_p50": torch.quantile(
                self.pairwise_ade_m.float(), 0.5
            ).to(self.pairwise_ade_m).detach(),
            "lfp_group_endpoint_std_x_m": self.endpoint_std_x_m.mean().detach(),
            "lfp_group_endpoint_std_y_m": self.endpoint_std_y_m.mean().detach(),
            "lfp_group_endpoint_std_heading_rad": (
                self.endpoint_std_heading_rad.mean().detach()
            ),
        }


@dataclass
class GroupCreditGateOutput:
    advantages: torch.Tensor
    active_group: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def normalized_transition_floor_from_endpoint_std(
    fs_scale: torch.Tensor,
    endpoint_std: torch.Tensor,
) -> torch.Tensor:
    """Maps a physical endpoint standard deviation to normalized delta space."""
    if fs_scale.ndim != 2 or fs_scale.shape[-1] != 3:
        raise ValueError(f"fs_scale must have shape [H, 3], got {tuple(fs_scale.shape)}.")
    if endpoint_std.shape != (3,):
        raise ValueError(f"endpoint_std must have shape [3], got {tuple(endpoint_std.shape)}.")
    if not torch.isfinite(fs_scale).all() or (fs_scale <= 0.0).any():
        raise ValueError("fs_scale must contain finite positive values.")
    if not torch.isfinite(endpoint_std).all() or (endpoint_std <= 0.0).any():
        raise ValueError("endpoint_std must contain finite positive values.")

    horizon = int(fs_scale.shape[0])
    raw_delta_std = endpoint_std / float(horizon) ** 0.5
    return raw_delta_std.unsqueeze(0) / fs_scale


def endpoint_std_from_normalized_transition_floor(
    normalized_floor: torch.Tensor,
    fs_scale: torch.Tensor,
) -> torch.Tensor:
    if normalized_floor.shape != fs_scale.shape:
        raise ValueError("normalized_floor and fs_scale must have identical [H, 3] shapes.")
    raw_delta_std = normalized_floor * fs_scale
    return raw_delta_std.square().sum(dim=0).sqrt()


def apply_transition_std_floor(
    std: torch.Tensor,
    scalar_floor: float,
    representation_floor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if scalar_floor <= 0.0:
        raise ValueError("scalar_floor must be positive.")
    bounded = std.clamp_min(float(scalar_floor))
    if representation_floor is None:
        return bounded
    if representation_floor.ndim != 2:
        raise ValueError("representation_floor must have shape [H, D].")
    floor = representation_floor.to(device=std.device, dtype=std.dtype)
    try:
        return torch.maximum(bounded, floor)
    except RuntimeError as error:
        raise ValueError(
            f"std shape {tuple(std.shape)} cannot broadcast with representation floor "
            f"{tuple(floor.shape)}."
        ) from error


def compute_group_trajectory_spread(trajectories: torch.Tensor) -> GroupTrajectorySpread:
    """Computes physical spread for raw trajectories shaped [B, G, H, 3]."""
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError(
            f"trajectories must have shape [B, G, H, 3], got {tuple(trajectories.shape)}."
        )
    _, group_size, _, _ = trajectories.shape
    if not torch.isfinite(trajectories).all():
        raise ValueError("trajectories must be finite.")

    xy = trajectories[..., :2]
    if group_size < 2:
        pairwise_ade = trajectories.new_zeros((trajectories.shape[0],))
    else:
        pairwise_distance = torch.linalg.vector_norm(
            xy[:, :, None] - xy[:, None, :],
            dim=-1,
        ).mean(dim=-1)
        pair_mask = torch.triu(
            torch.ones(
                (group_size, group_size),
                dtype=torch.bool,
                device=trajectories.device,
            ),
            diagonal=1,
        )
        pairwise_ade = pairwise_distance[:, pair_mask].mean(dim=1)

    endpoint = trajectories[:, :, -1]
    endpoint_std_xy = endpoint[..., :2].std(dim=1, unbiased=False)
    heading = endpoint[..., 2]
    heading_mean = torch.atan2(torch.sin(heading).mean(dim=1), torch.cos(heading).mean(dim=1))
    heading_error = torch.atan2(
        torch.sin(heading - heading_mean[:, None]),
        torch.cos(heading - heading_mean[:, None]),
    )
    heading_std = heading_error.square().mean(dim=1).sqrt()
    return GroupTrajectorySpread(
        pairwise_ade_m=pairwise_ade,
        endpoint_std_x_m=endpoint_std_xy[:, 0],
        endpoint_std_y_m=endpoint_std_xy[:, 1],
        endpoint_std_heading_rad=heading_std,
    )


def apply_group_credit_gate(
    advantages: torch.Tensor,
    group_pairwise_ade_m: Optional[torch.Tensor],
    group_scalar_span: torch.Tensor,
    *,
    min_pairwise_ade_m: float = 0.0,
    min_scalar_span: float = 0.0,
    advantage_deadband: float = 0.0,
) -> GroupCreditGateOutput:
    """Suppresses credit when a rollout group has no resolvable action/reward contrast."""
    if advantages.ndim != 2:
        raise ValueError("advantages must have shape [B, G].")
    if float(advantage_deadband) < 0.0:
        raise ValueError("advantage_deadband must be non-negative.")
    active_group, diagnostics = compute_group_credit_active_mask(
        group_pairwise_ade_m,
        group_scalar_span,
        min_pairwise_ade_m=min_pairwise_ade_m,
        min_scalar_span=min_scalar_span,
    )
    if active_group.shape[0] != advantages.shape[0]:
        raise ValueError("Group gate batch size must match advantages.")
    gated = torch.where(active_group[:, None], advantages, torch.zeros_like(advantages))
    pre_deadband = gated
    if float(advantage_deadband) > 0.0:
        gated = torch.where(
            gated.abs() >= float(advantage_deadband),
            gated,
            torch.zeros_like(gated),
        )

    diagnostics = dict(diagnostics)
    diagnostics["lfp_advantage_deadband_zero_ratio"] = (
        ((pre_deadband != 0.0) & (gated == 0.0)).float().mean().to(advantages).detach()
    )
    return GroupCreditGateOutput(
        advantages=gated.detach(),
        active_group=active_group.detach(),
        diagnostics=diagnostics,
    )


def compute_group_credit_active_mask(
    group_pairwise_ade_m: Optional[torch.Tensor],
    group_scalar_span: torch.Tensor,
    *,
    min_pairwise_ade_m: float = 0.0,
    min_scalar_span: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if group_scalar_span.ndim != 1:
        raise ValueError("group_scalar_span must have shape [B].")
    for name, value in (
        ("min_pairwise_ade_m", min_pairwise_ade_m),
        ("min_scalar_span", min_scalar_span),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    batch_size = group_scalar_span.shape[0]
    physical_ok = torch.ones(batch_size, dtype=torch.bool, device=group_scalar_span.device)
    if float(min_pairwise_ade_m) > 0.0:
        if group_pairwise_ade_m is None:
            raise ValueError("group_pairwise_ade_m is required when min_pairwise_ade_m > 0.")
        if group_pairwise_ade_m.shape != (batch_size,):
            raise ValueError("group_pairwise_ade_m must have shape [B].")
        physical_ok = group_pairwise_ade_m >= float(min_pairwise_ade_m)

    score_ok = group_scalar_span >= float(min_scalar_span)
    active_group = physical_ok & score_ok
    diagnostics = {
        "lfp_group_scalar_span_mean": group_scalar_span.mean().detach(),
        "lfp_low_spread_group_ratio": (~physical_ok).float().mean().to(group_scalar_span).detach(),
        "lfp_low_scalar_span_group_ratio": (~score_ok).float().mean().to(group_scalar_span).detach(),
        "lfp_credit_active_group_ratio": (
            active_group.float().mean().to(group_scalar_span).detach()
        ),
    }
    return active_group.detach(), diagnostics
