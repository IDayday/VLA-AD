from __future__ import annotations

import torch
from torch import nn

from .dataclasses import DEFAULT_RISK_CLASS_ORDER, RiskState, StrategyWeights


class RiskConditionedStrategyRouter(nn.Module):
    def __init__(
        self,
        planner_dim: int,
        mode: str = "independent",
        hidden_dim: int = 256,
        alpha_path: float = 2.0,
        alpha_interaction: float = 2.0,
        alpha_progress: float = 1.5,
        alpha_comfort: float = 1.0,
        beta_safety: float = 3.0,
        beta_progress_safety: float = 2.0,
        bias_path: float = 0.0,
        bias_interaction: float = 0.0,
        bias_progress: float = -0.5,
        bias_comfort: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"independent", "learned_softmax"}:
            raise ValueError("mode must be 'independent' or 'learned_softmax'")
        self.mode = mode
        self.alpha_path = float(alpha_path)
        self.alpha_interaction = float(alpha_interaction)
        self.alpha_progress = float(alpha_progress)
        self.alpha_comfort = float(alpha_comfort)
        self.beta_safety = float(beta_safety)
        self.beta_progress_safety = float(beta_progress_safety)
        self.bias_path = float(bias_path)
        self.bias_interaction = float(bias_interaction)
        self.bias_progress = float(bias_progress)
        self.bias_comfort = float(bias_comfort)
        self.learned_router = nn.Sequential(
            nn.LayerNorm(planner_dim),
            nn.Linear(planner_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def _prob(self, risk_state: RiskState, index: int) -> torch.Tensor:
        if risk_state.risk_probs.shape[-1] <= index:
            return risk_state.risk_probs.new_zeros(risk_state.risk_probs.shape[0], 1)
        return risk_state.risk_probs[:, index : index + 1]

    def _entropy(self, weights: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)

    def forward(self, risk_state: RiskState) -> StrategyWeights:
        p_low_score = self._prob(risk_state, 0)
        p_path = self._prob(risk_state, 1)
        p_interaction = self._prob(risk_state, 2)
        p_ttc = self._prob(risk_state, 3)
        p_progress = self._prob(risk_state, 4)
        p_comfort = self._prob(risk_state, 5)
        safety_risk = torch.maximum(p_interaction, p_ttc)

        if self.mode == "independent":
            w_path = torch.sigmoid(self.alpha_path * p_path - self.beta_safety * safety_risk + self.bias_path)
            w_interaction = torch.sigmoid(self.alpha_interaction * safety_risk + self.bias_interaction)
            w_progress = torch.sigmoid(self.alpha_progress * p_progress - self.beta_progress_safety * safety_risk + self.bias_progress)
            w_comfort = torch.sigmoid(self.alpha_comfort * p_comfort + self.bias_comfort)
            max_strategy = torch.maximum(torch.maximum(w_path, w_interaction), torch.maximum(w_progress, w_comfort))
            w_base = (1.0 - max_strategy).clamp(0.0, 1.0)
        else:
            param_dtype = next(self.learned_router.parameters()).dtype
            logits = self.learned_router(risk_state.risk_embedding.to(dtype=param_dtype)).to(dtype=risk_state.risk_probs.dtype)
            probs = torch.softmax(logits, dim=-1)
            w_base, w_path, w_interaction, w_progress, w_comfort = [probs[:, idx : idx + 1] for idx in range(5)]

        if risk_state.uncertainty is None:
            w_uncertainty = None
        else:
            uncertainty = risk_state.uncertainty
            if uncertainty.min().item() < 0.0 or uncertainty.max().item() > 1.0:
                uncertainty = torch.sigmoid(uncertainty)
            w_uncertainty = uncertainty.clamp(0.0, 1.0)

        stacked = torch.cat([w_base, w_path, w_interaction, w_progress, w_comfort], dim=-1)
        diagnostics = {
            "risk_prob_low_score": p_low_score.squeeze(-1),
            "risk_prob_path_dac": p_path.squeeze(-1),
            "risk_prob_interaction_nc": p_interaction.squeeze(-1),
            "risk_prob_ttc": p_ttc.squeeze(-1),
            "risk_prob_progress": p_progress.squeeze(-1),
            "risk_prob_comfort": p_comfort.squeeze(-1),
            "safety_risk": safety_risk.squeeze(-1),
            "strategy_weight_base": w_base.squeeze(-1),
            "strategy_weight_path_intent": w_path.squeeze(-1),
            "strategy_weight_interaction": w_interaction.squeeze(-1),
            "strategy_weight_progress": w_progress.squeeze(-1),
            "strategy_weight_comfort": w_comfort.squeeze(-1),
            "strategy_entropy": self._entropy(stacked),
        }
        return StrategyWeights(
            base=w_base,
            path_intent=w_path,
            interaction=w_interaction,
            progress=w_progress,
            comfort=w_comfort,
            uncertainty=w_uncertainty,
            diagnostics=diagnostics,
        )
