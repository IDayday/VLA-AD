from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def smooth_l1_or_zero(
    pred: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if pred is None or target is None:
        return reference.new_zeros(())
    return F.smooth_l1_loss(pred.float(), target.detach().float(), reduction="mean").to(dtype=reference.dtype)


def mean_l2_or_zero(
    pred: Optional[torch.Tensor],
    target: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if pred is None or target is None:
        return reference.new_zeros(())
    return torch.linalg.vector_norm((pred.float() - target.detach().float()).reshape(pred.shape[0], -1), dim=-1).mean().to(
        dtype=reference.dtype
    )


def build_path_targets(
    actions: torch.Tensor,
    anchor_indices: tuple[int, ...],
    *,
    expected_anchors: int,
) -> torch.Tensor:
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError(f"actions must have shape [B, H, 3], got {tuple(actions.shape)}.")
    if len(anchor_indices) != expected_anchors:
        raise ValueError(f"Expected {expected_anchors} anchor indices, got {len(anchor_indices)}.")
    max_idx = actions.shape[1] - 1
    safe_indices = [min(max(int(index), 0), max_idx) for index in anchor_indices]
    return actions[:, safe_indices, :]


def bit_loss_bundle(
    *,
    terminal_pred: Optional[torch.Tensor],
    path_anchor_pred: Optional[torch.Tensor],
    terminal_gt: Optional[torch.Tensor],
    path_gt: Optional[torch.Tensor],
    pred_x0_terminal_norm: Optional[torch.Tensor],
    terminal_condition_norm: Optional[torch.Tensor],
    reverse_points: Optional[torch.Tensor],
    reverse_gt: Optional[torch.Tensor],
    cycle_pred_norm: Optional[torch.Tensor] = None,
    cycle_target_norm: Optional[torch.Tensor] = None,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "bit_terminal_loss": smooth_l1_or_zero(terminal_pred, terminal_gt, reference),
        "bit_path_loss": smooth_l1_or_zero(path_anchor_pred, path_gt, reference),
        "bit_end_consistency_loss": smooth_l1_or_zero(pred_x0_terminal_norm, terminal_condition_norm, reference),
        "bit_reverse_loss": smooth_l1_or_zero(reverse_points, reverse_gt, reference),
        "bit_cycle_loss": smooth_l1_or_zero(cycle_pred_norm, cycle_target_norm, reference),
        "bit_terminal_pred_mean_l2": mean_l2_or_zero(terminal_pred, terminal_gt, reference),
        "bit_path_pred_mean_l2": mean_l2_or_zero(path_anchor_pred, path_gt, reference),
    }
