from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .dataclasses import StrategyWeights


def reduce_horizon_labels(labels: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if labels.ndim < 3:
        return labels
    if mode == "mean":
        return labels.float().mean(dim=1).to(dtype=labels.dtype)
    if mode == "max":
        return labels.max(dim=1).values
    raise ValueError("mode must be 'mean' or 'max'")


def ensure_scene_level_labels(labels: torch.Tensor) -> torch.Tensor:
    return reduce_horizon_labels(labels, mode="mean") if labels.ndim == 3 else labels


def mvp_labels_to_extended(
    generic: torch.Tensor,
    drivable: torch.Tensor,
    ttc: torch.Tensor,
    comfort: torch.Tensor,
) -> torch.Tensor:
    tensors = [generic, drivable, ttc, ttc, comfort, comfort]
    return torch.stack([tensor.to(dtype=torch.float32) for tensor in tensors], dim=-1)


def _apply_mask(loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return loss
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return loss * mask


def _reduce(loss: torch.Tensor, mask: Optional[torch.Tensor], reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(loss).sum().clamp_min(1.0)
    return loss.sum() / denom


def risk_bce_loss(
    risk_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    target = ensure_scene_level_labels(labels).to(device=risk_logits.device, dtype=risk_logits.dtype)
    if target.shape != risk_logits.shape:
        raise ValueError(f"Label shape {tuple(target.shape)} does not match logits {tuple(risk_logits.shape)}")
    loss = F.binary_cross_entropy_with_logits(risk_logits, target, pos_weight=pos_weight, reduction="none")
    loss = _apply_mask(loss, mask)
    return _reduce(loss, mask, reduction)


def risk_focal_loss(
    risk_logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float | torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    target = ensure_scene_level_labels(labels).to(device=risk_logits.device, dtype=risk_logits.dtype)
    if target.shape != risk_logits.shape:
        raise ValueError(f"Label shape {tuple(target.shape)} does not match logits {tuple(risk_logits.shape)}")
    bce = F.binary_cross_entropy_with_logits(risk_logits, target, reduction="none")
    prob = torch.sigmoid(risk_logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    focal = bce * (1.0 - pt).pow(float(gamma))
    if alpha is not None:
        alpha_tensor = torch.as_tensor(alpha, device=risk_logits.device, dtype=risk_logits.dtype)
        focal = focal * torch.where(target > 0.5, alpha_tensor, 1.0 - alpha_tensor)
    focal = _apply_mask(focal, mask)
    return _reduce(focal, mask, reduction)


def _weights_tensor(strategy_weights: StrategyWeights | torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(strategy_weights, torch.Tensor):
        return strategy_weights
    if isinstance(strategy_weights, StrategyWeights):
        return torch.cat(
            [
                strategy_weights.base,
                strategy_weights.path_intent,
                strategy_weights.interaction,
                strategy_weights.progress,
                strategy_weights.comfort,
            ],
            dim=-1,
        )
    return torch.cat(
        [
            strategy_weights["base"],
            strategy_weights["path_intent"],
            strategy_weights["interaction"],
            strategy_weights["progress"],
            strategy_weights["comfort"],
        ],
        dim=-1,
    )


def strategy_entropy_loss(
    strategy_weights: StrategyWeights | torch.Tensor | dict[str, torch.Tensor],
    target_entropy: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    weights = _weights_tensor(strategy_weights)
    probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    loss = -entropy if target_entropy is None else (entropy - float(target_entropy)).pow(2)
    return _reduce(loss, None, reduction)


def strategy_activation_supervision_loss(
    strategy_weights: StrategyWeights | torch.Tensor | dict[str, torch.Tensor],
    target_weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    weights = _weights_tensor(strategy_weights)
    target = target_weights.to(device=weights.device, dtype=weights.dtype)
    if target.shape != weights.shape:
        raise ValueError(f"Target strategy shape {tuple(target.shape)} does not match weights {tuple(weights.shape)}")
    loss = F.smooth_l1_loss(weights, target, reduction="none")
    loss = _apply_mask(loss, mask)
    return _reduce(loss, mask, reduction)
