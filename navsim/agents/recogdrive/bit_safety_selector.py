from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import torch
from torch import nn


SELECTOR_FEATURE_NAMES: List[str] = [
    "terminal_dx",
    "terminal_dy",
    "terminal_dheading",
    "early_x_delta_mean",
    "early_y_delta_mean",
    "early_step_length_delta",
    "mean_traj_l2",
    "max_traj_l2",
    "curvature_base",
    "curvature_bit",
    "curvature_delta",
    "endpoint_x_base",
    "endpoint_x_bit",
    "endpoint_y_base",
    "endpoint_y_bit",
    "early_x_delta_p0",
    "early_x_delta_p1",
    "early_x_delta_p2",
    "early_y_delta_p0",
    "early_y_delta_p1",
    "early_y_delta_p2",
    "early_heading_delta_mean",
    "early_heading_delta_max_abs",
    "terminal_dx_abs",
    "terminal_dy_abs",
    "terminal_dheading_abs",
    "longitudinal_delta_abs_mean",
    "lateral_delta_abs_mean",
    "max_forward_delta",
    "max_backward_delta",
    "max_lateral_abs_delta",
    "base_early_step_length_mean",
    "bit_early_step_length_mean",
    "endpoint_distance_base",
    "endpoint_distance_bit",
    "endpoint_distance_delta",
    "curvature_delta_abs",
]


class FeatureMLPSelector(nn.Module):
    """Lightweight selector over metric-free counterfactual trajectory features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ContextMLPSelector(nn.Module):
    """Optional selector variant combining context embeddings and handcrafted features."""

    def __init__(self, context_dim: int, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim + feature_dim),
            nn.Linear(context_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([context, features], dim=-1)).squeeze(-1)


def _as_tensor_traj(value: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
    tensor = tensor.float()
    if tensor.ndim != 2 or tensor.shape[-1] != 3:
        raise ValueError(f"Trajectory must have shape [H, 3], got {tuple(tensor.shape)}.")
    return tensor


def curvature_proxy(traj: Sequence[Sequence[float]] | torch.Tensor) -> float:
    tensor = _as_tensor_traj(traj)
    if tensor.shape[0] < 3:
        return 0.0
    deltas = tensor[1:] - tensor[:-1]
    headings = torch.atan2(deltas[:, 1], deltas[:, 0])
    return float(torch.mean(torch.abs(headings[1:] - headings[:-1])).item())


def trajectory_delta_features(
    base_traj: Sequence[Sequence[float]] | torch.Tensor,
    bit_traj: Sequence[Sequence[float]] | torch.Tensor,
    *,
    early_points: int = 3,
) -> Dict[str, float]:
    base = _as_tensor_traj(base_traj)
    bit = _as_tensor_traj(bit_traj)
    if base.shape != bit.shape:
        raise ValueError(f"Trajectory shape mismatch: {tuple(base.shape)} vs {tuple(bit.shape)}.")
    points = min(max(int(early_points), 1), base.shape[0])
    delta = bit - base
    base_steps = base[1:points] - base[: max(points - 1, 0)]
    bit_steps = bit[1:points] - bit[: max(points - 1, 0)]
    if base_steps.numel() == 0:
        early_step_length_delta = 0.0
        base_early_step_length_mean = 0.0
        bit_early_step_length_mean = 0.0
    else:
        base_step_lengths = torch.linalg.norm(base_steps[:, :2], dim=-1)
        bit_step_lengths = torch.linalg.norm(bit_steps[:, :2], dim=-1)
        early_step_length_delta = float((bit_step_lengths - base_step_lengths).mean().item())
        base_early_step_length_mean = float(base_step_lengths.mean().item())
        bit_early_step_length_mean = float(bit_step_lengths.mean().item())
    l2 = torch.linalg.norm(delta[:, :2], dim=-1)
    curv_base = curvature_proxy(base)
    curv_bit = curvature_proxy(bit)
    early_delta = delta[:points]
    early_heading_delta = early_delta[:, 2]
    endpoint_distance_base = float(torch.linalg.norm(base[-1, :2]).item())
    endpoint_distance_bit = float(torch.linalg.norm(bit[-1, :2]).item())
    features = {
        "terminal_dx": float(delta[-1, 0].item()),
        "terminal_dy": float(delta[-1, 1].item()),
        "terminal_dheading": float(delta[-1, 2].item()),
        "early_x_delta_mean": float(delta[:points, 0].mean().item()),
        "early_y_delta_mean": float(delta[:points, 1].mean().item()),
        "early_step_length_delta": early_step_length_delta,
        "mean_traj_l2": float(l2.mean().item()),
        "max_traj_l2": float(l2.max().item()),
        "curvature_base": curv_base,
        "curvature_bit": curv_bit,
        "curvature_delta": curv_bit - curv_base,
        "endpoint_x_base": float(base[-1, 0].item()),
        "endpoint_x_bit": float(bit[-1, 0].item()),
        "endpoint_y_base": float(base[-1, 1].item()),
        "endpoint_y_bit": float(bit[-1, 1].item()),
        "early_heading_delta_mean": float(early_heading_delta.mean().item()),
        "early_heading_delta_max_abs": float(torch.abs(early_heading_delta).max().item()),
        "terminal_dx_abs": float(torch.abs(delta[-1, 0]).item()),
        "terminal_dy_abs": float(torch.abs(delta[-1, 1]).item()),
        "terminal_dheading_abs": float(torch.abs(delta[-1, 2]).item()),
        "longitudinal_delta_abs_mean": float(torch.abs(delta[:, 0]).mean().item()),
        "lateral_delta_abs_mean": float(torch.abs(delta[:, 1]).mean().item()),
        "max_forward_delta": float(torch.max(delta[:, 0]).item()),
        "max_backward_delta": float(torch.min(delta[:, 0]).item()),
        "max_lateral_abs_delta": float(torch.abs(delta[:, 1]).max().item()),
        "base_early_step_length_mean": base_early_step_length_mean,
        "bit_early_step_length_mean": bit_early_step_length_mean,
        "endpoint_distance_base": endpoint_distance_base,
        "endpoint_distance_bit": endpoint_distance_bit,
        "endpoint_distance_delta": endpoint_distance_bit - endpoint_distance_base,
        "curvature_delta_abs": abs(curv_bit - curv_base),
    }
    for idx in range(3):
        safe_idx = min(idx, early_delta.shape[0] - 1)
        features[f"early_x_delta_p{idx}"] = float(early_delta[safe_idx, 0].item())
        features[f"early_y_delta_p{idx}"] = float(early_delta[safe_idx, 1].item())
    return features


def feature_tensor(
    feature_rows: Iterable[Mapping[str, float]],
    feature_names: Sequence[str] = SELECTOR_FEATURE_NAMES,
) -> torch.Tensor:
    values = [[float(row.get(name, 0.0)) for name in feature_names] for row in feature_rows]
    if not values:
        raise ValueError("No feature rows provided.")
    return torch.tensor(values, dtype=torch.float32)
