from __future__ import annotations

from typing import Dict, Optional

import torch


def anchor_grpo_reward(
    utility: torch.Tensor,
    zero_path_repair: torch.Tensor,
    ttc_repair: torch.Tensor,
    progress_recovery: torch.Tensor,
    comfort_stability: torch.Tensor,
    nc_regression: torch.Tensor,
    ttc_regression: torch.Tensor,
    dac_regression: torch.Tensor,
    tail_risk: torch.Tensor,
    positive_anchor_distance: torch.Tensor,
    negative_anchor_distance: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    negative_margin: float = 0.5,
) -> torch.Tensor:
    """Anchor-GRPO reward used only after supervised safety gates pass."""

    weights = weights or {}
    reward = utility.float() * float(weights.get("utility", 1.0))
    reward = reward + zero_path_repair.float() * float(weights.get("zero_path_repair", 0.4))
    reward = reward + ttc_repair.float() * float(weights.get("ttc_repair", 0.4))
    reward = reward + progress_recovery.float() * float(weights.get("progress_recovery", 0.2))
    reward = reward + comfort_stability.float() * float(weights.get("comfort_stability", 0.1))
    reward = reward - nc_regression.float() * float(weights.get("nc_regression", 2.5))
    reward = reward - ttc_regression.float() * float(weights.get("ttc_regression", 2.0))
    reward = reward - dac_regression.float() * float(weights.get("dac_regression", 1.0))
    reward = reward - tail_risk.float() * float(weights.get("cvar_tail_risk", 0.5))
    reward = reward - positive_anchor_distance.float() * float(weights.get("positive_anchor_deviation", 0.3))
    reward = reward + (negative_anchor_distance.float() - float(negative_margin)).clamp_min(0.0) * float(weights.get("negative_anchor_margin", 0.3))
    return reward


def supervised_gates_pass(gates: Dict[str, float], tolerances: Optional[Dict[str, float]] = None) -> bool:
    tolerances = tolerances or {}
    required = {
        "mean_pdms_delta_vs_a0": 0.0,
        "p10_delta_vs_a0": 0.0,
        "zero_delta_vs_a0": 0.0,
        "dac0_delta_vs_a0": 0.0,
        "nc0_delta_vs_a0": float(tolerances.get("nc0_delta_vs_a0", 1.0)),
        "ttc0_delta_vs_a0": 0.0,
        "critic_pairwise_accuracy": float(tolerances.get("critic_pairwise_accuracy", 0.5)),
        "validation_samples": float(tolerances.get("validation_samples", 10000)),
    }
    return (
        gates.get("mean_pdms_delta_vs_a0", -1.0) >= required["mean_pdms_delta_vs_a0"]
        and gates.get("p10_delta_vs_a0", -1.0) >= required["p10_delta_vs_a0"]
        and gates.get("zero_delta_vs_a0", 1.0) <= required["zero_delta_vs_a0"]
        and gates.get("dac0_delta_vs_a0", 1.0) <= required["dac0_delta_vs_a0"]
        and gates.get("nc0_delta_vs_a0", 2.0) <= required["nc0_delta_vs_a0"]
        and gates.get("ttc0_delta_vs_a0", 1.0) <= required["ttc0_delta_vs_a0"]
        and gates.get("critic_pairwise_accuracy", 0.0) > required["critic_pairwise_accuracy"]
        and gates.get("validation_samples", 0.0) >= required["validation_samples"]
    )
