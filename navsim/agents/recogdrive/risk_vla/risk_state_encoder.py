from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .dataclasses import DEFAULT_RISK_CLASS_ORDER, RiskState


class RiskStateEncoder(nn.Module):
    def __init__(
        self,
        planner_dim: int,
        hidden_dim: int = 512,
        num_risk_classes: int = 6,
        action_horizon: int = 8,
        use_bit_summary: bool = True,
        use_uncertainty_head: bool = True,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.num_risk_classes = int(num_risk_classes)
        self.action_horizon = int(action_horizon)
        self.use_bit_summary = bool(use_bit_summary)
        self.use_uncertainty_head = bool(use_uncertainty_head)
        bit_dim = 9 if self.use_bit_summary else 0
        input_dim = self.planner_dim + 8 + 12 + 3 + bit_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(hidden_dim, self.num_risk_classes)
        self.embedding_head = nn.Linear(hidden_dim, self.planner_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if self.use_uncertainty_head else None

    def _zeros(self, batch: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros(batch, dim)

    def _flatten_history(self, history_trajectory: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        history = history_trajectory.to(device=reference.device, dtype=reference.dtype)
        if history.ndim == 3:
            history = history.reshape(history.shape[0], -1)
        if history.shape[-1] < 12:
            pad = history.new_zeros(history.shape[0], 12 - history.shape[-1])
            history = torch.cat([history, pad], dim=-1)
        return history[:, :12]

    def _bit_summary(
        self,
        batch: int,
        reference: torch.Tensor,
        bit_terminal: Optional[torch.Tensor],
        bit_path: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.use_bit_summary:
            return reference.new_zeros(batch, 0)
        terminal = (
            bit_terminal.to(device=reference.device, dtype=reference.dtype).reshape(batch, -1)[:, :3]
            if bit_terminal is not None
            else self._zeros(batch, 3, reference)
        )
        if bit_path is None:
            path_mean = self._zeros(batch, 3, reference)
            path_terminal = self._zeros(batch, 3, reference)
        else:
            path = bit_path.to(device=reference.device, dtype=reference.dtype)
            path_mean = path.mean(dim=1)[:, :3]
            path_terminal = path[:, -1, :3]
        return torch.cat([terminal, path_mean, path_terminal], dim=-1)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        bit_terminal: Optional[torch.Tensor] = None,
        bit_path: Optional[torch.Tensor] = None,
    ) -> RiskState:
        if vlm_tokens.ndim != 3:
            raise ValueError(f"vlm_tokens must have shape [B, N, D], got {tuple(vlm_tokens.shape)}")
        reference = vlm_tokens
        batch = vlm_tokens.shape[0]
        vlm_pool = vlm_tokens.mean(dim=1)
        status = status_feature.to(device=reference.device, dtype=reference.dtype)
        if status.shape[-1] < 8:
            status = torch.cat([status, status.new_zeros(batch, 8 - status.shape[-1])], dim=-1)
        status = status[:, :8]
        history = self._flatten_history(history_trajectory, reference)
        command = high_command_one_hot.to(device=reference.device, dtype=reference.dtype)
        if command.shape[-1] < 3:
            command = torch.cat([command, command.new_zeros(batch, 3 - command.shape[-1])], dim=-1)
        command = command[:, :3]
        features = torch.cat([vlm_pool, status, history, command, self._bit_summary(batch, reference, bit_terminal, bit_path)], dim=-1)
        module_dtype = next(self.parameters()).dtype
        hidden = self.encoder(features.to(dtype=module_dtype))
        risk_logits = self.risk_head(hidden).to(dtype=reference.dtype)
        risk_probs = torch.sigmoid(risk_logits)
        risk_embedding = self.embedding_head(hidden).to(dtype=reference.dtype)
        uncertainty = self.uncertainty_head(hidden).to(dtype=reference.dtype) if self.uncertainty_head is not None else None
        diagnostics = {
            "risk_embedding_norm": torch.linalg.vector_norm(risk_embedding.float(), dim=-1).to(dtype=reference.dtype),
        }
        for idx, name in enumerate(DEFAULT_RISK_CLASS_ORDER[: self.num_risk_classes]):
            diagnostics[f"risk_prob_{name}"] = risk_probs[:, idx]
        if uncertainty is not None:
            diagnostics["uncertainty_mean"] = uncertainty.mean().reshape(())
        return RiskState(
            risk_logits=risk_logits,
            risk_probs=risk_probs,
            risk_embedding=risk_embedding,
            uncertainty=uncertainty,
            diagnostics=diagnostics,
        )
