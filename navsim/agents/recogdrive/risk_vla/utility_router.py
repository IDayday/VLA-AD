from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .dataclasses import RiskState


@dataclass
class UtilityRouterOutput:
    strategy_weights: torch.Tensor
    candidate_weights: torch.Tensor
    selected_candidate_id: torch.Tensor
    expected_utility: torch.Tensor
    expected_nc_risk: torch.Tensor
    expected_ttc_risk: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class RiskVLAv2UtilityRouter(nn.Module):
    MODES = {
        "heuristic_independent",
        "learned_strategy_softmax",
        "constrained_utility_router",
        "oracle_utility_router",
        "scorer_select",
    }

    def __init__(self, planner_dim: int, num_strategies: int = 5, hidden_dim: int = 256, safety_penalty: float = 3.0) -> None:
        super().__init__()
        self.planner_dim = int(planner_dim)
        self.num_strategies = int(num_strategies)
        self.safety_penalty = float(safety_penalty)
        self.strategy_head = nn.Sequential(
            nn.LayerNorm(planner_dim),
            nn.Linear(planner_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_strategies),
        )

    def forward(
        self,
        risk_state: RiskState,
        candidate_utility: Optional[torch.Tensor] = None,
        candidate_nc_risk: Optional[torch.Tensor] = None,
        candidate_ttc_risk: Optional[torch.Tensor] = None,
        oracle_candidate_id: Optional[torch.Tensor] = None,
        mode: str = "heuristic_independent",
    ) -> UtilityRouterOutput:
        if mode not in self.MODES:
            raise ValueError(f"Unknown router mode {mode!r}")
        batch = risk_state.risk_embedding.shape[0]
        dtype = risk_state.risk_embedding.dtype
        device = risk_state.risk_embedding.device
        if mode == "heuristic_independent":
            probs = risk_state.risk_probs
            base = (1.0 - probs.amax(dim=1, keepdim=True)).clamp(0.0, 1.0)
            strategy = torch.cat([base, probs[:, 1:5]], dim=1)
            strategy = strategy / strategy.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            logits = self.strategy_head(risk_state.risk_embedding.to(dtype=next(self.parameters()).dtype)).to(dtype=dtype)
            strategy = torch.softmax(logits, dim=-1)
        if candidate_utility is None:
            candidate_utility = strategy.new_zeros(batch, 1)
        utility = candidate_utility.to(device=device, dtype=dtype)
        num_candidates = utility.shape[1]
        nc = candidate_nc_risk.to(device=device, dtype=dtype) if candidate_nc_risk is not None else utility.new_zeros(batch, num_candidates)
        ttc = candidate_ttc_risk.to(device=device, dtype=dtype) if candidate_ttc_risk is not None else utility.new_zeros(batch, num_candidates)
        if mode == "oracle_utility_router":
            if oracle_candidate_id is None:
                raise ValueError("oracle_utility_router requires oracle_candidate_id")
            selected = oracle_candidate_id.to(device=device).long().view(batch)
            weights = torch.zeros_like(utility).scatter_(1, selected.view(-1, 1), 1.0)
        else:
            adjusted = utility
            if mode in {"constrained_utility_router", "scorer_select"}:
                adjusted = adjusted - self.safety_penalty * torch.maximum(nc, ttc)
            weights = torch.softmax(adjusted, dim=1)
            selected = adjusted.argmax(dim=1)
            if mode == "scorer_select":
                weights = torch.zeros_like(utility).scatter_(1, selected.view(-1, 1), 1.0)
        expected_utility = (weights * utility).sum(dim=1)
        expected_nc = (weights * nc).sum(dim=1)
        expected_ttc = (weights * ttc).sum(dim=1)
        diagnostics = {
            "selected_candidate_id": selected.to(dtype=dtype),
            "expected_utility": expected_utility,
            "expected_nc_risk": expected_nc,
            "expected_ttc_risk": expected_ttc,
        }
        return UtilityRouterOutput(strategy, weights, selected, expected_utility, expected_nc, expected_ttc, diagnostics)
