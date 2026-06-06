from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class TrajectoryRiskCriticOutput:
    utility_score: torch.Tensor
    risk_logits_scene: torch.Tensor
    risk_logits_horizon: torch.Tensor
    submetric_delta_pred: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _fit_last_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    return torch.cat([value, value.new_zeros(*value.shape[:-1], dim - value.shape[-1])], dim=-1)


def _flatten_history(history: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    history = history.to(device=reference.device, dtype=reference.dtype)
    if history.ndim == 3:
        history = history.reshape(history.shape[0], -1)
    return _fit_last_dim(history, 12)


class TrajectoryRiskCritic(nn.Module):
    """Scores candidate trajectories with scene/risk conditioned utility and safety heads."""

    def __init__(
        self,
        planner_dim: int,
        hidden_dim: int = 256,
        num_risk_classes: int = 6,
        horizon: int = 8,
        action_dim: int = 3,
        num_submetrics: int = 6,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_risk_classes = int(num_risk_classes)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.num_submetrics = int(num_submetrics)
        scene_dim = self.planner_dim + 8 + 12 + 3 + self.planner_dim
        traj_dim = self.horizon * self.action_dim + 8
        self.scene_encoder = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.traj_encoder = nn.Sequential(
            nn.LayerNorm(traj_dim),
            nn.Linear(traj_dim, hidden_dim),
            nn.GELU(),
        )
        self.joint = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.scene_risk_head = nn.Linear(hidden_dim, self.num_risk_classes)
        self.horizon_risk_head = nn.Linear(hidden_dim, self.horizon * self.num_risk_classes)
        self.submetric_delta_head = nn.Linear(hidden_dim, self.num_submetrics)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        candidate_trajectories: torch.Tensor,
        risk_embedding: Optional[torch.Tensor] = None,
        world_tokens: Optional[torch.Tensor] = None,
        strategy_tokens: Optional[torch.Tensor] = None,
    ) -> TrajectoryRiskCriticOutput:
        if vlm_tokens.ndim != 3:
            raise ValueError(f"vlm_tokens must be [B,N,D], got {tuple(vlm_tokens.shape)}")
        if candidate_trajectories.ndim != 4:
            raise ValueError(f"candidate_trajectories must be [B,K,H,3], got {tuple(candidate_trajectories.shape)}")
        batch, num_candidates = candidate_trajectories.shape[:2]
        dtype = vlm_tokens.dtype
        device = vlm_tokens.device
        pooled_tokens = vlm_tokens.mean(dim=1)
        if world_tokens is not None:
            pooled_tokens = pooled_tokens + _fit_last_dim(world_tokens.to(device=device, dtype=dtype).mean(dim=1), self.planner_dim)
        if strategy_tokens is not None:
            pooled_tokens = pooled_tokens + _fit_last_dim(strategy_tokens.to(device=device, dtype=dtype).mean(dim=1), self.planner_dim)
        status = _fit_last_dim(status_feature.to(device=device, dtype=dtype), 8)
        history = _flatten_history(history_trajectory, vlm_tokens)
        command = _fit_last_dim(high_command_one_hot.to(device=device, dtype=dtype), 3)
        risk = risk_embedding if risk_embedding is not None else vlm_tokens.new_zeros(batch, self.planner_dim)
        risk = _fit_last_dim(risk.to(device=device, dtype=dtype), self.planner_dim)
        scene_features = torch.cat([pooled_tokens, status, history, command, risk], dim=-1)
        param_dtype = next(self.parameters()).dtype
        scene_hidden = self.scene_encoder(scene_features.to(dtype=param_dtype))
        traj = candidate_trajectories.to(device=device, dtype=dtype)
        if traj.shape[2] != self.horizon:
            if traj.shape[2] > self.horizon:
                traj = traj[:, :, : self.horizon]
            else:
                pad = traj.new_zeros(batch, num_candidates, self.horizon - traj.shape[2], traj.shape[-1])
                traj = torch.cat([traj, pad], dim=2)
        traj = _fit_last_dim(traj, self.action_dim)
        deltas = traj[:, :, 1:, :2] - traj[:, :, :-1, :2]
        speed = torch.linalg.vector_norm(deltas.float(), dim=-1).to(dtype=dtype)
        traj_stats = torch.stack(
            [
                traj[..., 0].mean(dim=2),
                traj[..., 1].mean(dim=2),
                traj[..., 2].mean(dim=2),
                traj[..., 0].amax(dim=2),
                traj[..., 1].abs().amax(dim=2),
                speed.mean(dim=2),
                speed.amax(dim=2),
                traj[:, :, -1, 0],
            ],
            dim=-1,
        )
        traj_flat = torch.cat([traj.reshape(batch, num_candidates, -1), traj_stats], dim=-1)
        traj_hidden = self.traj_encoder(traj_flat.to(dtype=param_dtype))
        scene_expanded = scene_hidden[:, None, :].expand(-1, num_candidates, -1)
        joint = self.joint(torch.cat([scene_expanded, traj_hidden], dim=-1))
        utility = self.utility_head(joint).squeeze(-1).to(dtype=dtype)
        scene_logits = self.scene_risk_head(joint).to(dtype=dtype)
        horizon_logits = self.horizon_risk_head(joint).reshape(batch, num_candidates, self.horizon, self.num_risk_classes).to(dtype=dtype)
        submetric = self.submetric_delta_head(joint).to(dtype=dtype)
        diagnostics = {
            "critic_utility_mean": utility.mean(),
            "critic_scene_risk_prob_mean": torch.sigmoid(scene_logits.float()).mean().to(dtype=dtype),
            "critic_horizon_risk_prob_mean": torch.sigmoid(horizon_logits.float()).mean().to(dtype=dtype),
        }
        return TrajectoryRiskCriticOutput(utility, scene_logits, horizon_logits, submetric, diagnostics)
