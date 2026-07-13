from __future__ import annotations

import torch
import torch.nn.functional as F


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _masked_huber(excess: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if excess.numel() == 0:
        return excess.new_zeros((excess.shape[0],))
    terms = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=1.0, reduction="none")
    mask_f = mask.to(device=terms.device, dtype=terms.dtype)
    return (terms * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def compute_reference_relative_geometry_components(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    tangent_margin_rad: float,
    curvature_margin: float,
    min_segment_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_traj.shape != target_traj.shape or pred_traj.ndim != 3 or pred_traj.shape[-1] != 3:
        raise ValueError(
            "pred_traj and target_traj must have identical [B, H, 3] shapes, "
            f"got {tuple(pred_traj.shape)} and {tuple(target_traj.shape)}."
        )
    if tangent_margin_rad < 0.0 or curvature_margin < 0.0 or min_segment_length < 0.0:
        raise ValueError("geometry margins and min_segment_length must be non-negative.")

    target = target_traj.detach().to(pred_traj)
    pred = pred_traj.float()
    target_f = target.float()
    batch = int(pred.shape[0])
    p0_xy = pred.new_zeros((batch, 1, 2))
    p0_heading = pred.new_zeros((batch, 1))
    pred_points = torch.cat((p0_xy, pred[..., :2]), dim=1)
    target_points = torch.cat((p0_xy, target_f[..., :2]), dim=1)
    pred_heading = torch.cat((p0_heading, pred[..., 2]), dim=1)
    target_heading = torch.cat((p0_heading, target_f[..., 2]), dim=1)

    pred_segments = pred_points[:, 1:] - pred_points[:, :-1]
    target_segments = target_points[:, 1:] - target_points[:, :-1]
    pred_length = torch.linalg.vector_norm(pred_segments, dim=-1)
    target_length = torch.linalg.vector_norm(target_segments, dim=-1)
    valid_length_floor = max(float(min_segment_length), 1e-6)
    pred_segment_valid = pred_length >= valid_length_floor
    target_segment_valid = target_length >= valid_length_floor
    fallback_segment = pred_segments.new_zeros(pred_segments.shape)
    fallback_segment[..., 0] = 1.0
    safe_pred_segments = torch.where(pred_segment_valid[..., None], pred_segments, fallback_segment)
    safe_target_segments = torch.where(target_segment_valid[..., None], target_segments, fallback_segment)
    pred_tangent = torch.atan2(safe_pred_segments[..., 1], safe_pred_segments[..., 0])
    target_tangent = torch.atan2(safe_target_segments[..., 1], safe_target_segments[..., 0])

    tangent_mask = pred_segment_valid & target_segment_valid
    pred_tangent_error = _wrap_angle(pred_tangent - pred_heading[:, 1:]).abs()
    target_tangent_error = _wrap_angle(target_tangent - target_heading[:, 1:]).abs().detach()
    tangent_excess = F.relu(pred_tangent_error - target_tangent_error - float(tangent_margin_rad))
    tangent_loss = _masked_huber(tangent_excess, tangent_mask)

    if pred.shape[1] < 2:
        curvature_loss = pred.new_zeros((batch,))
    else:
        pred_turn = _wrap_angle(pred_tangent[:, 1:] - pred_tangent[:, :-1])
        target_turn = _wrap_angle(target_tangent[:, 1:] - target_tangent[:, :-1]).detach()
        pred_scale = 0.5 * (pred_length[:, 1:] + pred_length[:, :-1])
        target_scale = 0.5 * (target_length[:, 1:] + target_length[:, :-1])
        curvature_mask = (
            tangent_mask[:, 1:]
            & tangent_mask[:, :-1]
            & (pred_scale >= float(min_segment_length))
            & (target_scale >= float(min_segment_length))
        )
        pred_curvature = pred_turn.abs() / pred_scale.clamp_min(max(float(min_segment_length), 1e-6))
        target_curvature = target_turn.abs() / target_scale.clamp_min(max(float(min_segment_length), 1e-6))
        curvature_excess = F.relu(pred_curvature - target_curvature.detach() - float(curvature_margin))
        curvature_loss = _masked_huber(curvature_excess, curvature_mask)

    tangent_loss = torch.nan_to_num(tangent_loss, nan=0.0, posinf=0.0, neginf=0.0).to(pred_traj)
    curvature_loss = torch.nan_to_num(curvature_loss, nan=0.0, posinf=0.0, neginf=0.0).to(pred_traj)
    return tangent_loss, curvature_loss


def compute_reference_relative_geometry_loss(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    tangent_margin_rad: float,
    curvature_margin: float,
    min_segment_length: float,
) -> torch.Tensor:
    tangent_loss, curvature_loss = compute_reference_relative_geometry_components(
        pred_traj,
        target_traj,
        tangent_margin_rad,
        curvature_margin,
        min_segment_length,
    )
    return (tangent_loss + curvature_loss).mean()
