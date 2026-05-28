from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def init_logit_from_prob(p: float) -> float:
    """Returns a logit whose sigmoid is p."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    return math.log(p / (1.0 - p))


def branch_logits_from_probs(probs: Sequence[float]) -> torch.Tensor:
    """Initializes branch logits so softmax(logits) matches probs."""
    if len(probs) == 0:
        raise ValueError("At least one branch probability is required.")
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    if torch.any(probs_tensor <= 0):
        raise ValueError(f"Branch probabilities must be positive, got {list(probs)}.")
    probs_tensor = probs_tensor / probs_tensor.sum()
    return torch.log(probs_tensor)


def normalized_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float().detach(), dim=-1)
    return F.mse_loss(pred, target)


class TeacherTokenProjector(nn.Module):
    """Projects frozen teacher tokens into the ReCogDrive planner space."""

    def __init__(self, teacher_dim: int, expert_adapter_dim: int = 768, planner_dim: int = 384) -> None:
        super().__init__()
        self.teacher_dim = teacher_dim
        self.expert_adapter_dim = expert_adapter_dim
        self.planner_dim = planner_dim
        self.net = nn.Sequential(
            nn.LayerNorm(teacher_dim),
            nn.Linear(teacher_dim, expert_adapter_dim),
            nn.GELU(),
            nn.Linear(expert_adapter_dim, planner_dim),
            nn.LayerNorm(planner_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Teacher tokens must have shape [B, K, D], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"Teacher token dim {tokens.shape[-1]} does not match configured teacher_dim={self.teacher_dim}."
            )
        return self.net(tokens)


class ExpertAdapter768(nn.Module):
    """Learns fixed expert-query tokens from VLM embeddings via cross-attention."""

    def __init__(
        self,
        planner_dim: int = 384,
        expert_adapter_dim: int = 768,
        num_tokens: int = 12,
        num_heads: int = 12,
        ffn_multiplier: int = 4,
        zero_init_down: bool = True,
    ) -> None:
        super().__init__()
        if expert_adapter_dim % num_heads != 0:
            raise ValueError(
                f"expert_adapter_dim={expert_adapter_dim} must be divisible by num_heads={num_heads}."
            )
        self.planner_dim = planner_dim
        self.expert_adapter_dim = expert_adapter_dim
        self.num_tokens = num_tokens
        self.up = nn.Linear(planner_dim, expert_adapter_dim)
        self.queries = nn.Parameter(torch.empty(num_tokens, expert_adapter_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=expert_adapter_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(expert_adapter_dim),
            nn.Linear(expert_adapter_dim, ffn_multiplier * expert_adapter_dim),
            nn.GELU(),
            nn.Linear(ffn_multiplier * expert_adapter_dim, expert_adapter_dim),
        )
        self.down = nn.Linear(expert_adapter_dim, planner_dim)
        self._reset_parameters(zero_init_down=zero_init_down)

    def _reset_parameters(self, *, zero_init_down: bool) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        if zero_init_down:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)
        else:
            nn.init.normal_(self.down.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.down.bias)

    def forward(self, vl_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if vl_embeds.ndim != 3:
            raise ValueError(f"vl_embeds must have shape [B, Nv, D], got {tuple(vl_embeds.shape)}.")
        if vl_embeds.shape[-1] != self.planner_dim:
            raise ValueError(
                f"vl_embeds dim {vl_embeds.shape[-1]} does not match planner_dim={self.planner_dim}."
            )

        batch_size = vl_embeds.shape[0]
        kv = self.up(vl_embeds)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(queries, kv, kv, need_weights=False)
        z_768 = attended + self.ffn(attended)
        z_384 = self.down(z_768)
        return z_768, z_384


class AlignmentHead(nn.Module):
    """Maps internal expert adapter tokens back to a teacher feature space."""

    def __init__(self, expert_adapter_dim: int = 768, teacher_dim: int = 1024) -> None:
        super().__init__()
        self.teacher_dim = teacher_dim
        self.net = nn.Sequential(
            nn.LayerNorm(expert_adapter_dim),
            nn.Linear(expert_adapter_dim, teacher_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)
