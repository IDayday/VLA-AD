from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence

import torch


CandidateType = Literal["base", "bit"]


@dataclass
class CandidateBatch:
    """Paired Base/BiT candidate trajectories for selector/reranker training."""

    scene_token: Sequence[str]
    sample_token: Sequence[str]
    base_pred_traj: torch.Tensor
    bit_pred_traj: torch.Tensor
    base_metrics: Optional[Sequence[Dict[str, Any]]] = None
    bit_metrics: Optional[Sequence[Dict[str, Any]]] = None
    features: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_pred_traj.shape != self.bit_pred_traj.shape:
            raise ValueError(
                "base_pred_traj and bit_pred_traj must have the same shape, "
                f"got {tuple(self.base_pred_traj.shape)} vs {tuple(self.bit_pred_traj.shape)}."
            )
        if self.base_pred_traj.ndim != 3 or self.base_pred_traj.shape[-1] != 3:
            raise ValueError(f"Trajectories must have shape [B, H, 3], got {tuple(self.base_pred_traj.shape)}.")
        batch = self.base_pred_traj.shape[0]
        if len(self.scene_token) != batch or len(self.sample_token) != batch:
            raise ValueError("scene_token and sample_token lengths must match trajectory batch size.")

    def select(self, choice: torch.Tensor) -> torch.Tensor:
        """Return selected trajectory where choice=True uses BiT and choice=False uses Base."""

        if choice.ndim != 1 or choice.shape[0] != self.base_pred_traj.shape[0]:
            raise ValueError(f"choice must have shape [B], got {tuple(choice.shape)}.")
        mask = choice.to(device=self.base_pred_traj.device, dtype=torch.bool).view(-1, 1, 1)
        return torch.where(mask, self.bit_pred_traj, self.base_pred_traj)
