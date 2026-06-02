from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _flatten_or_none(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value.reshape(value.shape[0], -1)


def _fit_feature_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    return torch.nn.functional.pad(value, (0, dim - value.shape[-1]))


class TerminalPathHead(nn.Module):
    """Predicts explicit terminal intent and sparse path anchors from planner context."""

    def __init__(
        self,
        planner_dim: int = 384,
        status_dim: int = 8,
        his_dim: int = 12,
        hidden_dim: int = 512,
        num_anchors: int = 4,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.status_dim = int(status_dim)
        self.his_dim = int(his_dim)
        self.num_anchors = int(num_anchors)
        in_dim = self.planner_dim + self.status_dim + self.his_dim
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.terminal = nn.Linear(hidden_dim, 3)
        self.path = nn.Linear(hidden_dim, self.num_anchors * 3)

    def forward(
        self,
        context_summary: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        his_traj: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_summary.ndim != 2:
            raise ValueError(f"context_summary must have shape [B, D], got {tuple(context_summary.shape)}.")
        batch_size = context_summary.shape[0]
        status = _flatten_or_none(status_feature)
        if status is None:
            status = context_summary.new_zeros(batch_size, self.status_dim)
        else:
            status = _fit_feature_dim(status.to(device=context_summary.device, dtype=context_summary.dtype), self.status_dim)
        history = _flatten_or_none(his_traj)
        if history is None:
            history = context_summary.new_zeros(batch_size, self.his_dim)
        else:
            history = _fit_feature_dim(history.to(device=context_summary.device, dtype=context_summary.dtype), self.his_dim)
        features = torch.cat([context_summary, status, history], dim=-1)
        hidden = self.shared(features)
        terminal_pred = self.terminal(hidden)
        path_anchor_pred = self.path(hidden).view(batch_size, self.num_anchors, 3)
        return terminal_pred, path_anchor_pred


class TargetPathTokenEncoder(nn.Module):
    """Encodes selected terminal/path conditions as planner-dimension tokens."""

    def __init__(
        self,
        planner_dim: int = 384,
        hidden_dim: int = 512,
        num_anchors: int = 4,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.num_anchors = int(num_anchors)
        self.terminal_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.planner_dim),
            nn.LayerNorm(self.planner_dim),
        )
        self.path_encoder = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.planner_dim),
            nn.LayerNorm(self.planner_dim),
        )
        self.anchor_embedding = nn.Embedding(self.num_anchors, self.planner_dim)
        nn.init.normal_(self.anchor_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        terminal_state: torch.Tensor,
        path_anchors: torch.Tensor,
        *,
        use_path_anchors: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if terminal_state.ndim != 2 or terminal_state.shape[-1] != 3:
            raise ValueError(f"terminal_state must have shape [B, 3], got {tuple(terminal_state.shape)}.")
        if path_anchors.ndim != 3 or path_anchors.shape[-1] != 3:
            raise ValueError(f"path_anchors must have shape [B, K, 3], got {tuple(path_anchors.shape)}.")
        if path_anchors.shape[1] != self.num_anchors:
            raise ValueError(f"path_anchors has {path_anchors.shape[1]} anchors, expected {self.num_anchors}.")
        terminal_token = self.terminal_encoder(terminal_state).unsqueeze(1)
        path_tokens = self.path_encoder(path_anchors)
        anchor_ids = torch.arange(self.num_anchors, device=path_anchors.device)
        path_tokens = path_tokens + self.anchor_embedding(anchor_ids).to(dtype=path_tokens.dtype).unsqueeze(0)
        if use_path_anchors:
            target_path_summary = torch.cat([terminal_token, path_tokens], dim=1).mean(dim=1)
        else:
            path_tokens = torch.zeros_like(path_tokens)
            target_path_summary = terminal_token.squeeze(1)
        return terminal_token, path_tokens, target_path_summary


class ReverseTrajectoryDecoder(nn.Module):
    """Auxiliary training-only decoder for bidirectional trajectory consistency."""

    def __init__(
        self,
        planner_dim: int = 384,
        hidden_dim: int = 384,
        num_anchors: int = 4,
        reverse_points: int = 7,
    ) -> None:
        super().__init__()
        self.reverse_points = int(reverse_points)
        self.condition_encoder = TargetPathTokenEncoder(
            planner_dim=planner_dim,
            hidden_dim=max(hidden_dim, planner_dim),
            num_anchors=num_anchors,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(planner_dim * 2),
            nn.Linear(planner_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.reverse_points * 3),
        )

    def forward(
        self,
        context_summary: torch.Tensor,
        terminal_state: torch.Tensor,
        path_anchors: torch.Tensor,
        *,
        use_path_anchors: bool = True,
    ) -> torch.Tensor:
        _, _, target_path_summary = self.condition_encoder(
            terminal_state,
            path_anchors,
            use_path_anchors=use_path_anchors,
        )
        hidden = torch.cat([context_summary, target_path_summary], dim=-1)
        return self.decoder(hidden).view(context_summary.shape[0], self.reverse_points, 3)


class BitRiskHead(nn.Module):
    """Predicts trajectory-level zero/DAC/NC/TTC risks from BiT intent and context."""

    def __init__(
        self,
        planner_dim: int = 384,
        status_dim: int = 8,
        his_dim: int = 12,
        hidden_dim: int = 512,
        num_anchors: int = 4,
        num_risks: int = 4,
    ) -> None:
        super().__init__()
        self.status_dim = int(status_dim)
        self.his_dim = int(his_dim)
        self.num_anchors = int(num_anchors)
        self.num_risks = int(num_risks)
        in_dim = int(planner_dim) + 3 + self.num_anchors * 3 + self.status_dim + self.his_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_risks),
        )

    def forward(
        self,
        context_summary: torch.Tensor,
        terminal_state: torch.Tensor,
        path_anchors: torch.Tensor,
        status_feature: Optional[torch.Tensor] = None,
        his_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_summary.ndim != 2:
            raise ValueError(f"context_summary must have shape [B, D], got {tuple(context_summary.shape)}.")
        batch_size = context_summary.shape[0]
        status = _flatten_or_none(status_feature)
        if status is None:
            status = context_summary.new_zeros(batch_size, self.status_dim)
        else:
            status = _fit_feature_dim(status.to(device=context_summary.device, dtype=context_summary.dtype), self.status_dim)
        history = _flatten_or_none(his_traj)
        if history is None:
            history = context_summary.new_zeros(batch_size, self.his_dim)
        else:
            history = _fit_feature_dim(history.to(device=context_summary.device, dtype=context_summary.dtype), self.his_dim)
        path = path_anchors.reshape(batch_size, -1).to(device=context_summary.device, dtype=context_summary.dtype)
        features = torch.cat([context_summary, terminal_state, path, status, history], dim=-1)
        return self.net(features)


class RiskTokenEncoder(nn.Module):
    """Encodes predicted risk probabilities as context tokens for the diffusion planner."""

    def __init__(self, planner_dim: int = 384, hidden_dim: int = 256, num_risks: int = 4) -> None:
        super().__init__()
        self.num_risks = int(num_risks)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.num_risks),
            nn.Linear(self.num_risks, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, planner_dim),
            nn.LayerNorm(planner_dim),
        )
        self.type_embedding = nn.Parameter(torch.empty(1, 1, planner_dim))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

    def forward(self, risk_logits: torch.Tensor) -> torch.Tensor:
        if risk_logits.ndim != 2 or risk_logits.shape[-1] != self.num_risks:
            raise ValueError(f"risk_logits must have shape [B, {self.num_risks}], got {tuple(risk_logits.shape)}.")
        risk_prob = torch.sigmoid(risk_logits.float()).to(dtype=risk_logits.dtype)
        return self.encoder(risk_prob).unsqueeze(1) + self.type_embedding.to(
            device=risk_logits.device,
            dtype=risk_logits.dtype,
        )
