from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BitRuleFallbackConfig:
    early_x_delta_threshold: float = 1.0
    terminal_x_delta_threshold: float = 2.0


class BitSafetyRouter(nn.Module):
    """Small binary router that predicts whether to use the BiT trajectory."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def trajectory_rule_use_bit(
    base_traj: torch.Tensor,
    bit_traj: torch.Tensor,
    *,
    early_x_delta_threshold: float = 1.0,
    terminal_x_delta_threshold: float = 2.0,
    early_points: int = 3,
) -> torch.Tensor:
    """Conservative rule fallback: reject BiT if it is much more longitudinally aggressive."""

    if base_traj.ndim != 3 or bit_traj.ndim != 3:
        raise ValueError("base_traj and bit_traj must have shape [B, H, 3].")
    if base_traj.shape != bit_traj.shape:
        raise ValueError(f"Trajectory shape mismatch: {tuple(base_traj.shape)} vs {tuple(bit_traj.shape)}")
    points = min(max(int(early_points), 1), base_traj.shape[1])
    early_delta = (bit_traj[:, :points, 0] - base_traj[:, :points, 0]).mean(dim=1)
    terminal_delta = bit_traj[:, -1, 0] - base_traj[:, -1, 0]
    return (early_delta <= float(early_x_delta_threshold)) & (
        terminal_delta <= float(terminal_x_delta_threshold)
    )


def router_feature_vector(
    *,
    context_mean: Optional[torch.Tensor] = None,
    terminal_pred: Optional[torch.Tensor] = None,
    path_anchor_pred: Optional[torch.Tensor] = None,
    his_traj: Optional[torch.Tensor] = None,
    status_feature: Optional[torch.Tensor] = None,
    base_traj: Optional[torch.Tensor] = None,
    bit_traj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    parts = []
    reference = None
    for value in (context_mean, terminal_pred, path_anchor_pred, his_traj, status_feature, base_traj, bit_traj):
        if value is not None:
            reference = value
            break
    if reference is None:
        raise ValueError("At least one router feature tensor is required.")
    batch = reference.shape[0]
    for value in (context_mean, terminal_pred, path_anchor_pred, his_traj, status_feature, base_traj, bit_traj):
        if value is None:
            continue
        parts.append(value.reshape(batch, -1).to(device=reference.device, dtype=reference.dtype))
    return torch.cat(parts, dim=-1)
