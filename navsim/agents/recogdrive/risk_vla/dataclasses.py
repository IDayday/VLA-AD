from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


DEFAULT_RISK_CLASS_ORDER = [
    "low_score",
    "path_dac",
    "interaction_nc",
    "ttc",
    "progress",
    "comfort",
]


@dataclass
class RiskState:
    risk_logits: torch.Tensor
    risk_probs: torch.Tensor
    risk_embedding: torch.Tensor
    uncertainty: Optional[torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class StrategyWeights:
    base: torch.Tensor
    path_intent: torch.Tensor
    interaction: torch.Tensor
    progress: torch.Tensor
    comfort: torch.Tensor
    uncertainty: Optional[torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class StrategyOutput:
    strategy_tokens: torch.Tensor
    horizon_residual: torch.Tensor
    action_prior: Optional[torch.Tensor]
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, torch.Tensor]
