from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .trajectory_feasibility import compute_curvature_proxy, compute_feasibility_metrics


class ParetoVectorScorer(nn.Module):
    def __init__(self, hidden_dim: int = 256, source_vocab_size: int = 32):
        super().__init__()
        self.traj_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.scene_encoder = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.status_encoder = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.command_encoder = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.source_embedding = nn.Embedding(source_vocab_size, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 11)

    def _trajectory_features(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(f"trajectory must have shape [B, H, 3], got {tuple(trajectory.shape)}.")
        delta = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        speed = torch.linalg.norm(delta[..., :2], dim=-1)
        accel = speed[:, 1:] - speed[:, :-1] if speed.shape[1] > 1 else speed.new_zeros(speed.shape[0], 0)
        curvature = compute_curvature_proxy(trajectory)
        feas = compute_feasibility_metrics(trajectory, {})
        pooled = [
            trajectory.flatten(1),
            delta.flatten(1),
            speed.mean(dim=1, keepdim=True),
            speed.std(dim=1, unbiased=False, keepdim=True),
            accel.mean(dim=1, keepdim=True) if accel.numel() else speed.new_zeros(speed.shape[0], 1),
            curvature.mean(dim=1, keepdim=True) if curvature.numel() else speed.new_zeros(speed.shape[0], 1),
            feas.feas_cost.reshape(-1, 1),
            feas.early_kink_rate.reshape(-1, 1),
            feas.tail_reverse_rate.reshape(-1, 1),
        ]
        return torch.cat(pooled, dim=1)

    def forward(
        self,
        scene_tokens: torch.Tensor,
        ego_status: torch.Tensor,
        route_command: torch.Tensor,
        trajectory: torch.Tensor,
        source_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if scene_tokens.ndim == 3:
            scene = scene_tokens.mean(dim=1)
        else:
            scene = scene_tokens.flatten(1)
        traj_feat = self.traj_encoder(self._trajectory_features(trajectory))
        scene_feat = self.scene_encoder(scene)
        status_feat = self.status_encoder(ego_status.flatten(1))
        command_feat = self.command_encoder(route_command.flatten(1))
        if source_ids is None:
            source_feat = torch.zeros_like(traj_feat)
        else:
            source_feat = self.source_embedding(source_ids.to(device=trajectory.device, dtype=torch.long).clamp_min(0))
        fused = self.fusion(torch.cat([traj_feat, scene_feat, status_feat, command_feat, source_feat], dim=1))
        raw = self.head(fused)
        pdms = torch.sigmoid(raw[:, 8])
        utility = raw[:, 10]
        return {
            "nc_logit": raw[:, 0],
            "dac_logit": raw[:, 1],
            "ttc": torch.sigmoid(raw[:, 2]),
            "_ep": torch.sigmoid(raw[:, 3]),
            "comfort": torch.sigmoid(raw[:, 4]),
            "ddc": torch.sigmoid(raw[:, 5]),
            "tlc_logit": raw[:, 6],
            "feas_cost": F.softplus(raw[:, 7]),
            "pdms": pdms,
            "pareto_front_logit": raw[:, 9],
            "utility": utility,
        }
