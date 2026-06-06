from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.expand_as(loss).sum().clamp_min(1.0)


def risk_bce_or_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    focal_gamma: Optional[float] = None,
) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if focal_gamma is not None:
        prob = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, prob, 1.0 - prob)
        loss = loss * (1.0 - pt).pow(float(focal_gamma))
    return _masked_mean(loss, mask)


def submetric_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    target = target.to(device=pred.device, dtype=pred.dtype)
    return _masked_mean(F.smooth_l1_loss(pred, target, reduction="none"), mask)


def pairwise_preference_loss(utility_score: torch.Tensor, positive_index: torch.Tensor, negative_index: torch.Tensor) -> torch.Tensor:
    pos = utility_score.gather(1, positive_index.to(device=utility_score.device).long().view(-1, 1)).squeeze(1)
    neg = utility_score.gather(1, negative_index.to(device=utility_score.device).long().view(-1, 1)).squeeze(1)
    return -torch.log(torch.sigmoid(pos - neg).clamp_min(1e-8)).mean()


def constrained_safety_loss(
    utility_score: torch.Tensor,
    nc_ttc_risk_prob: torch.Tensor,
    threshold: float = 0.5,
    margin: float = 0.0,
) -> torch.Tensor:
    risk = nc_ttc_risk_prob.to(device=utility_score.device, dtype=utility_score.dtype)
    if risk.ndim == utility_score.ndim + 1:
        risk = risk.amax(dim=-1)
    unsafe = (risk - float(threshold)).clamp_min(0.0)
    return (unsafe * (utility_score + float(margin)).clamp_min(0.0)).mean()


def cvar_tail_risk_loss(safety_risk_score: torch.Tensor, q: float = 0.2) -> torch.Tensor:
    if not 0.0 < float(q) <= 1.0:
        raise ValueError("q must be in (0, 1]")
    flat = safety_risk_score.reshape(safety_risk_score.shape[0], -1)
    k = max(1, int(round(flat.shape[1] * float(q))))
    return flat.topk(k, dim=1).values.mean()


def critic_total_loss(
    utility_score: torch.Tensor,
    risk_logits_scene: torch.Tensor,
    risk_logits_horizon: torch.Tensor,
    submetric_delta_pred: torch.Tensor,
    scene_labels: Optional[torch.Tensor] = None,
    horizon_labels: Optional[torch.Tensor] = None,
    submetric_delta_target: Optional[torch.Tensor] = None,
    positive_index: Optional[torch.Tensor] = None,
    negative_index: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or {}
    zero = utility_score.new_zeros(())
    losses: Dict[str, torch.Tensor] = {
        "scene_risk_loss": zero,
        "horizon_risk_loss": zero,
        "submetric_delta_loss": zero,
        "pairwise_preference_loss": zero,
        "constrained_safety_loss": zero,
        "cvar_tail_risk_loss": zero,
    }
    if scene_labels is not None:
        losses["scene_risk_loss"] = risk_bce_or_focal_loss(risk_logits_scene, scene_labels)
    if horizon_labels is not None:
        losses["horizon_risk_loss"] = risk_bce_or_focal_loss(risk_logits_horizon, horizon_labels)
    if submetric_delta_target is not None:
        losses["submetric_delta_loss"] = submetric_delta_loss(submetric_delta_pred, submetric_delta_target)
    if positive_index is not None and negative_index is not None:
        losses["pairwise_preference_loss"] = pairwise_preference_loss(utility_score, positive_index, negative_index)
    safety_prob = torch.sigmoid(risk_logits_horizon[..., 2:4]).amax(dim=(-1, -2))
    losses["constrained_safety_loss"] = constrained_safety_loss(utility_score, safety_prob)
    losses["cvar_tail_risk_loss"] = cvar_tail_risk_loss(safety_prob)
    total = zero
    for key, value in losses.items():
        total = total + float(weights.get(key, 1.0)) * value
    losses["total_loss"] = total
    return losses
