from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FutureLatentPrediction:
    future_latents: torch.Tensor
    pooled_context: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _fit_last_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    return torch.cat([value, value.new_zeros(*value.shape[:-1], dim - value.shape[-1])], dim=-1)


class FutureLatentPredictor(nn.Module):
    """Predicts future planner latent tokens from current VLM/risk/world context."""

    def __init__(
        self,
        planner_dim: int,
        hidden_dim: int = 256,
        future_steps: int = 4,
        status_dim: int = 8,
        history_dim: int = 12,
        command_dim: int = 3,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.future_steps = int(future_steps)
        self.status_dim = int(status_dim)
        self.history_dim = int(history_dim)
        self.command_dim = int(command_dim)
        input_dim = self.planner_dim * 3 + self.status_dim + self.history_dim + self.command_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.future_head = nn.Linear(hidden_dim, self.future_steps * self.planner_dim)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        risk_embedding: Optional[torch.Tensor] = None,
        world_tokens: Optional[torch.Tensor] = None,
    ) -> FutureLatentPrediction:
        if vlm_tokens.ndim != 3:
            raise ValueError(f"vlm_tokens must be [B,N,D], got {tuple(vlm_tokens.shape)}")
        batch = vlm_tokens.shape[0]
        dtype = vlm_tokens.dtype
        device = vlm_tokens.device
        pooled = _fit_last_dim(vlm_tokens.mean(dim=1), self.planner_dim)
        risk = (
            vlm_tokens.new_zeros(batch, self.planner_dim)
            if risk_embedding is None
            else _fit_last_dim(risk_embedding.to(device=device, dtype=dtype), self.planner_dim)
        )
        if world_tokens is None:
            world = vlm_tokens.new_zeros(batch, self.planner_dim)
        else:
            world = _fit_last_dim(world_tokens.to(device=device, dtype=dtype).mean(dim=1), self.planner_dim)
        status = _fit_last_dim(status_feature.to(device=device, dtype=dtype), self.status_dim)
        history = history_trajectory.to(device=device, dtype=dtype)
        if history.ndim == 3:
            history = history.reshape(batch, -1)
        history = _fit_last_dim(history, self.history_dim)
        command = _fit_last_dim(high_command_one_hot.to(device=device, dtype=dtype), self.command_dim)
        features = torch.cat([pooled, risk, world, status, history, command], dim=-1)
        param_dtype = next(self.parameters()).dtype
        hidden = self.encoder(features.to(dtype=param_dtype))
        future = self.future_head(hidden).reshape(batch, self.future_steps, self.planner_dim).to(dtype=dtype)
        diagnostics = {
            "future_latent_norm": torch.linalg.vector_norm(future.float(), dim=-1).mean(dim=1).to(dtype=dtype),
            "future_latent_context_norm": torch.linalg.vector_norm(pooled.float(), dim=-1).to(dtype=dtype),
        }
        return FutureLatentPrediction(future_latents=future, pooled_context=pooled, diagnostics=diagnostics)


def future_latent_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if target.shape[1] != prediction.shape[1]:
        if target.shape[1] > prediction.shape[1]:
            target = target[:, : prediction.shape[1]]
        else:
            pad = target.new_zeros(target.shape[0], prediction.shape[1] - target.shape[1], target.shape[2])
            target = torch.cat([target, pad], dim=1)
    target = _fit_last_dim(target, prediction.shape[-1])
    loss = F.smooth_l1_loss(prediction.float(), target.detach().float(), reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.to(device=prediction.device, dtype=prediction.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.expand_as(loss).sum().clamp_min(1.0)
