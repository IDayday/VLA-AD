from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class PlanningTokenAdapterConfig:
    planner_dim: int = 384
    hidden_dim: int = 1024
    num_tokens: int = 16
    num_heads: int = 8
    condition_dropout: float = 0.10
    context_gate_init: float = 0.05


def _logit_from_probability(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class PlanningTokenAdapter(nn.Module):
    """Builds static planning tokens from scene context and current ego state."""

    def __init__(self, config: PlanningTokenAdapterConfig) -> None:
        super().__init__()
        self.config = config
        if config.planner_dim <= 0 or config.hidden_dim <= 0 or config.num_tokens <= 0:
            raise ValueError("planner_dim, hidden_dim, and num_tokens must be positive.")
        if config.num_heads <= 0 or config.planner_dim % config.num_heads != 0:
            raise ValueError("num_heads must be positive and divide planner_dim.")
        if not 0.0 <= float(config.condition_dropout) < 1.0:
            raise ValueError("condition_dropout must be in [0.0, 1.0).")
        if not 0.0 < float(config.context_gate_init) < 1.0:
            raise ValueError("context_gate_init must be in (0.0, 1.0).")

        dim = int(config.planner_dim)
        hidden = int(config.hidden_dim)
        self.queries = nn.Parameter(torch.randn(config.num_tokens, dim) * 0.02)
        self.vlm_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.state_mlp = nn.Sequential(
            nn.LayerNorm(23),
            nn.Linear(23, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            dim,
            int(config.num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, hidden * 2)
        self.ffn_out = nn.Linear(hidden, dim)
        self.context_gate_logit = nn.Parameter(
            torch.tensor(_logit_from_probability(config.context_gate_init), dtype=torch.float32)
        )
        self.register_buffer("_forward_count", torch.zeros((), dtype=torch.long), persistent=False)

    @property
    def forward_count(self) -> int:
        return int(self._forward_count.detach().cpu().item())

    def reset_forward_count(self) -> None:
        self._forward_count.zero_()

    @staticmethod
    def _flatten_history(history_trajectory: torch.Tensor, batch_size: int) -> torch.Tensor:
        if history_trajectory.shape[0] != batch_size:
            raise ValueError("history_trajectory batch dimension must match vlm_tokens.")
        history = history_trajectory.reshape(batch_size, -1)
        if history.shape[1] != 12:
            raise ValueError(
                "history_trajectory must have shape [B, 12] or [B, 4, 3], "
                f"got {tuple(history_trajectory.shape)}."
            )
        return history

    @staticmethod
    def _pairwise_cosine(tokens: torch.Tensor) -> torch.Tensor:
        count = int(tokens.shape[1])
        if count < 2:
            return tokens.new_zeros(())
        normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        cosine = normalized @ normalized.transpose(1, 2)
        mask = ~torch.eye(count, device=tokens.device, dtype=torch.bool).unsqueeze(0)
        return cosine.masked_select(mask.expand(tokens.shape[0], -1, -1)).mean().to(tokens)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        status_feature: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        history_trajectory: torch.Tensor,
        condition_dropout_enabled: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if vlm_tokens.ndim != 3 or vlm_tokens.shape[-1] != self.config.planner_dim:
            raise ValueError(
                f"vlm_tokens must have shape [B, L, {self.config.planner_dim}], "
                f"got {tuple(vlm_tokens.shape)}."
            )
        batch_size = int(vlm_tokens.shape[0])
        if status_feature.shape != (batch_size, 8):
            raise ValueError(f"status_feature must have shape [B, 8], got {tuple(status_feature.shape)}.")
        if high_command_one_hot.shape != (batch_size, 3):
            raise ValueError(
                f"high_command_one_hot must have shape [B, 3], got {tuple(high_command_one_hot.shape)}."
            )
        history = self._flatten_history(history_trajectory, batch_size)
        state_input = torch.cat(
            (
                status_feature.to(vlm_tokens),
                high_command_one_hot.to(vlm_tokens),
                history.to(vlm_tokens),
            ),
            dim=-1,
        )
        state_token = self.state_mlp(state_input).unsqueeze(1)
        memory = torch.cat((self.vlm_norm(vlm_tokens), state_token), dim=1)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1).to(vlm_tokens) + state_token
        attended, _ = self.cross_attn(
            self.query_norm(queries),
            memory,
            memory,
            need_weights=False,
        )
        tokens = queries + attended
        value, gate = self.ffn_in(self.ffn_norm(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.ffn_out(value * F.silu(gate))

        apply_condition_dropout = (
            self.training if condition_dropout_enabled is None else bool(condition_dropout_enabled)
        )
        if apply_condition_dropout and self.config.condition_dropout > 0.0:
            keep = (
                torch.rand((batch_size, 1, 1), device=tokens.device)
                >= float(self.config.condition_dropout)
            ).to(tokens)
            tokens = tokens * keep
        else:
            keep = tokens.new_ones((batch_size, 1, 1))

        with torch.no_grad():
            self._forward_count.add_(1)
        diagnostics = {
            "planning_token_norm": tokens.detach().float().norm(dim=-1).mean().to(tokens),
            "planning_token_pairwise_cosine": self._pairwise_cosine(tokens.detach()),
            "planning_condition_keep_ratio": keep.detach().float().mean().to(tokens),
            "planning_adapter_forward_count": tokens.new_tensor(float(self.forward_count)),
        }
        return tokens, diagnostics
