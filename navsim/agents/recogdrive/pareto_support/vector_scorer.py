from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


SCORER_OUTPUT_KEYS = (
    "nc_logit",
    "dac_logit",
    "ttc",
    "ep",
    "comfort",
    "ddc",
    "tlc_logit",
    "feas_cost",
    "pdms",
    "pareto_front_logit",
    "utility",
)


class ParetoVectorScorer(nn.Module):
    def __init__(self, scene_dim: int = 256, hidden_dim: int = 256, source_vocab: int = 16):
        super().__init__()
        self.scene_projection = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, hidden_dim), nn.GELU())
        self.source_embedding = nn.Embedding(source_vocab, 16)
        self.traj_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 16),
            nn.Linear(hidden_dim * 2 + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(SCORER_OUTPUT_KEYS)),
        )

    @staticmethod
    def trajectory_features(trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(f"trajectory must have shape [B, H, 3], got {tuple(trajectory.shape)}.")
        dx = torch.diff(trajectory[..., 0], prepend=torch.zeros_like(trajectory[..., :1, 0]), dim=1)
        dy = torch.diff(trajectory[..., 1], prepend=torch.zeros_like(trajectory[..., :1, 1]), dim=1)
        dtheta = torch.atan2(
            torch.sin(torch.diff(trajectory[..., 2], prepend=torch.zeros_like(trajectory[..., :1, 2]), dim=1)),
            torch.cos(torch.diff(trajectory[..., 2], prepend=torch.zeros_like(trajectory[..., :1, 2]), dim=1)),
        )
        speed = torch.sqrt(dx.square() + dy.square())
        accel = torch.diff(speed, prepend=torch.zeros_like(speed[:, :1]), dim=1)
        yaw_rate = dtheta
        jerk = torch.diff(accel, prepend=torch.zeros_like(accel[:, :1]), dim=1)
        curvature = dtheta.abs() / speed.clamp_min(1e-4)
        reverse = (dx < -0.02).to(trajectory.dtype)
        raw = torch.stack([trajectory[..., 0], trajectory[..., 1], trajectory[..., 2], dx, dy, dtheta, speed, accel, yaw_rate, jerk, curvature, reverse], dim=-1)
        return raw.mean(dim=1)

    def forward(
        self,
        scene_tokens: Optional[torch.Tensor],
        ego_status: Optional[torch.Tensor],
        route_command: Optional[torch.Tensor],
        trajectory: torch.Tensor,
        source_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch = trajectory.shape[0]
        traj_feat = self.traj_encoder(self.trajectory_features(trajectory))
        if scene_tokens is None:
            scene_feat = torch.zeros(batch, self.scene_projection[1].in_features, device=trajectory.device, dtype=trajectory.dtype)
        elif scene_tokens.ndim == 3:
            scene_feat = scene_tokens.mean(dim=1)
        else:
            scene_feat = scene_tokens
        if scene_feat.shape[-1] != self.scene_projection[1].in_features:
            if scene_feat.shape[-1] > self.scene_projection[1].in_features:
                scene_feat = scene_feat[..., : self.scene_projection[1].in_features]
            else:
                pad = torch.zeros(batch, self.scene_projection[1].in_features - scene_feat.shape[-1], device=scene_feat.device, dtype=scene_feat.dtype)
                scene_feat = torch.cat([scene_feat, pad], dim=-1)
        scene_encoded = self.scene_projection(scene_feat.to(dtype=trajectory.dtype))
        if source_ids is None:
            source_ids = torch.zeros(batch, device=trajectory.device, dtype=torch.long)
        source_feat = self.source_embedding(source_ids.clamp_min(0).clamp_max(self.source_embedding.num_embeddings - 1))
        logits = self.fusion(torch.cat([scene_encoded, traj_feat, source_feat.to(dtype=trajectory.dtype)], dim=-1))
        return {key: logits[:, idx] for idx, key in enumerate(SCORER_OUTPUT_KEYS)}


def scorer_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], lambda_formula: float = 0.1, lambda_front: float = 0.2) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    for key, pred in outputs.items():
        if key in {"nc_logit", "dac_logit", "tlc_logit", "pareto_front_logit"}:
            target_key = key.replace("_logit", "")
            if target_key in targets:
                losses[key] = F.binary_cross_entropy_with_logits(pred, targets[target_key].to(dtype=pred.dtype))
        elif key in targets:
            losses[key] = F.smooth_l1_loss(pred, targets[key].to(dtype=pred.dtype))
    if {"pdms", "nc_logit", "dac_logit", "ttc", "ep", "comfort"}.issubset(outputs):
        pdms_formula = torch.sigmoid(outputs["nc_logit"]) * torch.sigmoid(outputs["dac_logit"]) * (
            5.0 * outputs["ep"].clamp(0, 1) + 5.0 * outputs["ttc"].clamp(0, 1) + 2.0 * outputs["comfort"].clamp(0, 1)
        ) / 12.0
        losses["formula"] = F.l1_loss(outputs["pdms"].clamp(0, 1), pdms_formula)
    total = sum(losses.values()) if losses else torch.zeros((), device=next(iter(outputs.values())).device)
    if "formula" in losses:
        total = total + (float(lambda_formula) - 1.0) * losses["formula"]
    if "pareto_front_logit" in losses:
        total = total + (float(lambda_front) - 1.0) * losses["pareto_front_logit"]
    return total, losses
