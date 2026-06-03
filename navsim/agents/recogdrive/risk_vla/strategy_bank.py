from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .dataclasses import RiskState, StrategyOutput, StrategyWeights


def _flatten_history(history_trajectory: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    history = history_trajectory.to(device=reference.device, dtype=reference.dtype)
    if history.ndim == 3:
        history = history.reshape(history.shape[0], -1)
    if history.shape[-1] < 12:
        history = torch.cat([history, history.new_zeros(history.shape[0], 12 - history.shape[-1])], dim=-1)
    return history[:, :12]


def _fit_last_dim(value: torch.Tensor, dim: int) -> torch.Tensor:
    if value.shape[-1] == dim:
        return value
    if value.shape[-1] > dim:
        return value[..., :dim]
    return torch.cat([value, value.new_zeros(*value.shape[:-1], dim - value.shape[-1])], dim=-1)


class _TokenHead(nn.Module):
    def __init__(self, input_dim: int, planner_dim: int, tokens_per_strategy: int) -> None:
        super().__init__()
        self.tokens_per_strategy = int(tokens_per_strategy)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, planner_dim),
            nn.GELU(),
            nn.Linear(planner_dim, self.tokens_per_strategy * planner_dim),
        )
        self.planner_dim = int(planner_dim)

    def forward(self, features: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        out = self.net(features.to(dtype=next(self.parameters()).dtype))
        return out.reshape(features.shape[0], self.tokens_per_strategy, self.planner_dim).to(dtype=dtype)


class PathIntentStrategy(nn.Module):
    """BiT-style path/terminal intent as one RISK-VLA path-risk strategy.

    This module produces conditioning tokens and horizon residuals only. It does
    not define the core algorithm and does not overwrite final trajectories.
    """

    def __init__(self, planner_dim: int, action_horizon: int = 8, action_dim: int = 3, tokens_per_strategy: int = 2) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.path_step_proj = nn.Linear(action_dim + planner_dim, planner_dim)
        self.token_head = _TokenHead(action_dim * 3 + planner_dim, planner_dim, tokens_per_strategy)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        risk_state: RiskState,
        bit_terminal: Optional[torch.Tensor] = None,
        bit_path: Optional[torch.Tensor] = None,
    ) -> StrategyOutput:
        batch = vlm_tokens.shape[0]
        dtype = vlm_tokens.dtype
        if bit_path is None:
            path = vlm_tokens.new_zeros(batch, self.action_horizon, self.action_dim)
        else:
            path = bit_path.to(device=vlm_tokens.device, dtype=dtype)
            if path.shape[1] != self.action_horizon:
                if path.shape[1] > self.action_horizon:
                    path = path[:, : self.action_horizon]
                else:
                    pad = path.new_zeros(batch, self.action_horizon - path.shape[1], path.shape[-1])
                    path = torch.cat([path, pad], dim=1)
            path = _fit_last_dim(path, self.action_dim)
        if bit_terminal is None:
            terminal = vlm_tokens.new_zeros(batch, self.action_dim)
        else:
            terminal = _fit_last_dim(bit_terminal.to(device=vlm_tokens.device, dtype=dtype).reshape(batch, -1), self.action_dim)
        path_mean = path.mean(dim=1)
        path_terminal = path[:, -1]
        token_features = torch.cat([terminal, path_mean, path_terminal, risk_state.risk_embedding.to(dtype=dtype)], dim=-1)
        tokens = self.token_head(token_features, dtype)
        step_risk = risk_state.risk_embedding.to(device=vlm_tokens.device, dtype=dtype).unsqueeze(1).expand(-1, self.action_horizon, -1)
        residual = self.path_step_proj(torch.cat([path, step_risk], dim=-1).to(dtype=next(self.path_step_proj.parameters()).dtype)).to(dtype=dtype)
        diagnostics = {
            "path_intent_token_norm": torch.linalg.vector_norm(tokens.float(), dim=-1).mean(),
            "path_intent_residual_norm": torch.linalg.vector_norm(residual.float(), dim=-1).mean(),
        }
        return StrategyOutput(tokens, residual, path, {}, diagnostics)


class InteractionSafetyStrategy(nn.Module):
    def __init__(self, planner_dim: int, action_horizon: int = 8, tokens_per_strategy: int = 2) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.action_horizon = int(action_horizon)
        self.token_head = _TokenHead(planner_dim + 8 + 12 + planner_dim, planner_dim, tokens_per_strategy)
        self.residual_head = nn.Linear(planner_dim + 8 + 12, planner_dim)
        early = torch.linspace(1.0, 0.25, steps=self.action_horizon).view(1, self.action_horizon, 1)
        self.register_buffer("early_emphasis", early, persistent=False)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        risk_state: RiskState,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
    ) -> StrategyOutput:
        dtype = vlm_tokens.dtype
        batch = vlm_tokens.shape[0]
        status = _fit_last_dim(status_feature.to(device=vlm_tokens.device, dtype=dtype), 8)
        history = _flatten_history(history_trajectory, vlm_tokens)
        pool = vlm_tokens.mean(dim=1)
        features = torch.cat([pool, status, history, risk_state.risk_embedding.to(dtype=dtype)], dim=-1)
        tokens = self.token_head(features, dtype)
        base = torch.cat([pool, status, history], dim=-1)
        step = self.residual_head(base.to(dtype=next(self.residual_head.parameters()).dtype)).to(dtype=dtype)
        residual = step.unsqueeze(1).expand(batch, self.action_horizon, -1) * self.early_emphasis.to(device=vlm_tokens.device, dtype=dtype)
        diagnostics = {"interaction_residual_norm": torch.linalg.vector_norm(residual.float(), dim=-1).mean()}
        return StrategyOutput(tokens, residual, None, {}, diagnostics)


class ProgressStrategy(nn.Module):
    def __init__(self, planner_dim: int, action_horizon: int = 8, tokens_per_strategy: int = 2) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.action_horizon = int(action_horizon)
        self.token_head = _TokenHead(planner_dim + 8 + 3 + planner_dim, planner_dim, tokens_per_strategy)
        self.residual_head = nn.Linear(planner_dim + 8 + 3, planner_dim)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        risk_state: RiskState,
        status_feature: torch.Tensor,
        high_command_one_hot: torch.Tensor,
    ) -> StrategyOutput:
        dtype = vlm_tokens.dtype
        batch = vlm_tokens.shape[0]
        status = _fit_last_dim(status_feature.to(device=vlm_tokens.device, dtype=dtype), 8)
        command = _fit_last_dim(high_command_one_hot.to(device=vlm_tokens.device, dtype=dtype), 3)
        pool = vlm_tokens.mean(dim=1)
        features = torch.cat([pool, status, command, risk_state.risk_embedding.to(dtype=dtype)], dim=-1)
        tokens = self.token_head(features, dtype)
        base = torch.cat([pool, status, command], dim=-1)
        step = self.residual_head(base.to(dtype=next(self.residual_head.parameters()).dtype)).to(dtype=dtype)
        ramp = torch.linspace(0.5, 1.0, steps=self.action_horizon, device=vlm_tokens.device, dtype=dtype).view(1, self.action_horizon, 1)
        residual = step.unsqueeze(1).expand(batch, self.action_horizon, -1) * ramp
        diagnostics = {"progress_residual_norm": torch.linalg.vector_norm(residual.float(), dim=-1).mean()}
        return StrategyOutput(tokens, residual, None, {}, diagnostics)


class ComfortStrategy(nn.Module):
    def __init__(self, planner_dim: int, action_horizon: int = 8, tokens_per_strategy: int = 2) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.action_horizon = int(action_horizon)
        self.token_head = _TokenHead(planner_dim + 12 + planner_dim, planner_dim, tokens_per_strategy)
        self.residual_head = nn.Linear(planner_dim + 12, planner_dim)

    def forward(self, vlm_tokens: torch.Tensor, risk_state: RiskState, history_trajectory: torch.Tensor) -> StrategyOutput:
        dtype = vlm_tokens.dtype
        batch = vlm_tokens.shape[0]
        history = _flatten_history(history_trajectory, vlm_tokens)
        pool = vlm_tokens.mean(dim=1)
        features = torch.cat([pool, history, risk_state.risk_embedding.to(dtype=dtype)], dim=-1)
        tokens = self.token_head(features, dtype)
        base = torch.cat([pool, history], dim=-1)
        step = self.residual_head(base.to(dtype=next(self.residual_head.parameters()).dtype)).to(dtype=dtype)
        residual = step.unsqueeze(1).expand(batch, self.action_horizon, -1)
        diagnostics = {"comfort_residual_norm": torch.linalg.vector_norm(residual.float(), dim=-1).mean()}
        return StrategyOutput(tokens, residual, None, {}, diagnostics)


class RiskConditionedStrategyBank(nn.Module):
    def __init__(
        self,
        planner_dim: int,
        action_horizon: int = 8,
        action_dim: int = 3,
        tokens_per_strategy: int = 2,
    ) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.action_horizon = int(action_horizon)
        self.path_intent = PathIntentStrategy(planner_dim, action_horizon, action_dim, tokens_per_strategy)
        self.interaction = InteractionSafetyStrategy(planner_dim, action_horizon, tokens_per_strategy)
        self.progress = ProgressStrategy(planner_dim, action_horizon, tokens_per_strategy)
        self.comfort = ComfortStrategy(planner_dim, action_horizon, tokens_per_strategy)

    def _weight_tokens(self, tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return tokens * weight.to(device=tokens.device, dtype=tokens.dtype).view(tokens.shape[0], 1, 1)

    def _weight_residual(self, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return residual * weight.to(device=residual.device, dtype=residual.dtype).view(residual.shape[0], 1, 1)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        risk_state: RiskState,
        strategy_weights: StrategyWeights,
        status_feature: torch.Tensor,
        history_trajectory: torch.Tensor,
        high_command_one_hot: torch.Tensor,
        bit_terminal: Optional[torch.Tensor] = None,
        bit_path: Optional[torch.Tensor] = None,
    ) -> StrategyOutput:
        path = self.path_intent(vlm_tokens, risk_state, bit_terminal, bit_path)
        interaction = self.interaction(vlm_tokens, risk_state, status_feature, history_trajectory)
        progress = self.progress(vlm_tokens, risk_state, status_feature, high_command_one_hot)
        comfort = self.comfort(vlm_tokens, risk_state, history_trajectory)
        weighted_tokens = [
            self._weight_tokens(path.strategy_tokens, strategy_weights.path_intent),
            self._weight_tokens(interaction.strategy_tokens, strategy_weights.interaction),
            self._weight_tokens(progress.strategy_tokens, strategy_weights.progress),
            self._weight_tokens(comfort.strategy_tokens, strategy_weights.comfort),
        ]
        strategy_tokens = torch.cat(weighted_tokens, dim=1)
        horizon_residual = (
            self._weight_residual(path.horizon_residual, strategy_weights.path_intent)
            + self._weight_residual(interaction.horizon_residual, strategy_weights.interaction)
            + self._weight_residual(progress.horizon_residual, strategy_weights.progress)
            + self._weight_residual(comfort.horizon_residual, strategy_weights.comfort)
        )
        diagnostics: Dict[str, torch.Tensor] = {
            "mean_weight_base": strategy_weights.base.mean(),
            "mean_weight_path_intent": strategy_weights.path_intent.mean(),
            "mean_weight_interaction": strategy_weights.interaction.mean(),
            "mean_weight_progress": strategy_weights.progress.mean(),
            "mean_weight_comfort": strategy_weights.comfort.mean(),
            "strategy_tokens_norm": torch.linalg.vector_norm(strategy_tokens.float(), dim=-1).mean(),
            "horizon_residual_norm": torch.linalg.vector_norm(horizon_residual.float(), dim=-1).mean(),
        }
        for output in (path, interaction, progress, comfort):
            diagnostics.update(output.diagnostics)
        return StrategyOutput(
            strategy_tokens=strategy_tokens,
            horizon_residual=horizon_residual,
            action_prior=path.action_prior,
            losses={},
            diagnostics=diagnostics,
        )
