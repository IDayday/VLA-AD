from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(device=loss.device, dtype=loss.dtype).view(loss.shape[0])
    if mask.sum().item() <= 0:
        return reference.new_zeros(())
    per_item = loss.reshape(loss.shape[0], -1).mean(dim=1)
    return (per_item * mask).sum().to(dtype=reference.dtype) / mask.sum().clamp_min(1.0).to(dtype=reference.dtype)


def diffusion_kd_to_positive_loss(pred_traj: torch.Tensor, positive_anchor_traj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_traj.float(), positive_anchor_traj.detach().float().to(device=pred_traj.device), reduction="none")
    return _masked_mean(loss, mask, pred_traj)


def anti_imitation_margin_loss(
    pred_traj: torch.Tensor,
    negative_anchor_traj: torch.Tensor,
    safety_mask: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm((pred_traj - negative_anchor_traj.detach().to(device=pred_traj.device)).float(), dim=-1).mean(dim=-1)
    loss = (float(margin) - distance).clamp_min(0.0)
    return _masked_mean(loss, safety_mask, pred_traj)


def path_anchor_preserve_loss(
    pred_traj: torch.Tensor,
    path_anchor_traj: torch.Tensor,
    path_repair_mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_traj[..., :2].float(), path_anchor_traj.detach().to(device=pred_traj.device)[..., :2].float(), reduction="none")
    return _masked_mean(loss, path_repair_mask, pred_traj)


def safe_alignment_total_loss(
    pred_traj: torch.Tensor,
    positive_anchor_traj: torch.Tensor,
    negative_anchor_traj: torch.Tensor,
    safety_mask: torch.Tensor,
    path_repair_mask: Optional[torch.Tensor] = None,
    path_anchor_traj: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or {}
    kd = diffusion_kd_to_positive_loss(pred_traj, positive_anchor_traj, safety_mask)
    anti = anti_imitation_margin_loss(pred_traj, negative_anchor_traj, safety_mask)
    if path_repair_mask is not None and path_anchor_traj is not None:
        path = path_anchor_preserve_loss(pred_traj, path_anchor_traj, path_repair_mask)
    else:
        path = pred_traj.new_zeros(())
    total = (
        float(weights.get("positive_kd", 1.0)) * kd
        + float(weights.get("negative_margin", 1.0)) * anti
        + float(weights.get("path_anchor", 1.0)) * path
    )
    return {
        "safealign_positive_kd_loss": kd,
        "safealign_negative_margin_loss": anti,
        "safealign_path_anchor_loss": path,
        "safealign_total_loss": total,
    }
