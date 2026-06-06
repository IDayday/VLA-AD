from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class RiskWorldTokenEncoder(nn.Module):
    """Builds expert-style tokens for semantic, geometric, dynamic, progress, comfort, and tail risk."""

    TOKEN_NAMES = [
        "semantic_interaction",
        "geometric_drivable",
        "dynamic_ttc",
        "ego_intent_progress",
        "comfort",
        "tail_risk",
    ]

    def __init__(self, planner_dim: int, hidden_dim: int = 256, num_risk_classes: int = 6) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.num_risk_classes = int(num_risk_classes)
        input_dim = self.planner_dim + self.num_risk_classes + 8 + 12 + 3 + self.planner_dim
        self.token_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.planner_dim),
                )
                for _ in self.TOKEN_NAMES
            ]
        )

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        risk_probs: torch.Tensor,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        candidate_summary: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = vlm_tokens.shape[0]
        dtype = vlm_tokens.dtype
        pooled = vlm_tokens.mean(dim=1)
        risk = risk_probs.to(device=vlm_tokens.device, dtype=dtype)
        if risk.shape[-1] < self.num_risk_classes:
            risk = torch.cat([risk, risk.new_zeros(batch, self.num_risk_classes - risk.shape[-1])], dim=-1)
        risk = risk[:, : self.num_risk_classes]
        status = status_feature.to(device=vlm_tokens.device, dtype=dtype)
        status = status[:, :8] if status.shape[-1] >= 8 else torch.cat([status, status.new_zeros(batch, 8 - status.shape[-1])], dim=-1)
        history = history_trajectory.to(device=vlm_tokens.device, dtype=dtype)
        if history.ndim == 3:
            history = history.reshape(batch, -1)
        history = history[:, :12] if history.shape[-1] >= 12 else torch.cat([history, history.new_zeros(batch, 12 - history.shape[-1])], dim=-1)
        command = high_command_one_hot.to(device=vlm_tokens.device, dtype=dtype)
        command = command[:, :3] if command.shape[-1] >= 3 else torch.cat([command, command.new_zeros(batch, 3 - command.shape[-1])], dim=-1)
        if candidate_summary is None:
            candidate = vlm_tokens.new_zeros(batch, self.planner_dim)
        else:
            candidate = candidate_summary.to(device=vlm_tokens.device, dtype=dtype)
            candidate = candidate[:, : self.planner_dim] if candidate.shape[-1] >= self.planner_dim else torch.cat([candidate, candidate.new_zeros(batch, self.planner_dim - candidate.shape[-1])], dim=-1)
        features = torch.cat([pooled, risk, status, history, command, candidate], dim=-1)
        param_dtype = next(self.parameters()).dtype
        tokens = torch.stack([head(features.to(dtype=param_dtype)).to(dtype=dtype) for head in self.token_heads], dim=1)
        diagnostics = {f"risk_world_token_{name}_norm": torch.linalg.vector_norm(tokens[:, idx].float(), dim=-1) for idx, name in enumerate(self.TOKEN_NAMES)}
        return tokens, diagnostics
