from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_float_sequence(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _direction_from_heading(traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    heading = traj[..., 2]
    direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
    normal = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
    return direction, normal


def _ramp(traj: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    h = traj.shape[-2]
    values = torch.linspace(0.0, 1.0, h, device=traj.device, dtype=traj.dtype)
    if power != 1.0:
        values = values.pow(power)
    return values.view(*((1,) * (traj.ndim - 2)), h, 1)


def generate_endpoint_extension(anchor: torch.Tensor, deltas_m: Sequence[float]) -> torch.Tensor:
    deltas = _as_float_sequence(deltas_m)
    if not deltas:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    direction, _ = _direction_from_heading(anchor)
    ramp = _ramp(anchor)
    variants = []
    for delta in deltas:
        traj = anchor.clone()
        traj[..., :2] = traj[..., :2] + ramp * direction * float(delta)
        variants.append(traj)
    return torch.stack(variants, dim=1)


def generate_speed_scale(anchor: torch.Tensor, scales: Sequence[float]) -> torch.Tensor:
    scales = _as_float_sequence(scales)
    if not scales:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    variants = []
    for scale in scales:
        traj = anchor.clone()
        traj[..., :2] = traj[..., :2] * float(scale)
        variants.append(traj)
    return torch.stack(variants, dim=1)


def _linear_resample(anchor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    h = anchor.shape[1]
    pos = positions.clamp(0.0, float(h - 1))
    left = pos.floor().long()
    right = (left + 1).clamp(max=h - 1)
    frac = (pos - left.to(pos.dtype)).view(1, h, 1)
    gather_left = anchor[:, left, :]
    gather_right = anchor[:, right, :]
    return gather_left * (1.0 - frac) + gather_right * frac


def generate_time_gamma(anchor: torch.Tensor, gammas: Sequence[float]) -> torch.Tensor:
    gammas = _as_float_sequence(gammas)
    if not gammas:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    h = anchor.shape[1]
    t = torch.linspace(0.0, 1.0, h, device=anchor.device, dtype=anchor.dtype)
    variants = []
    for gamma in gammas:
        positions = t.pow(float(gamma)) * float(h - 1)
        variants.append(_linear_resample(anchor, positions))
    return torch.stack(variants, dim=1)


def generate_lateral_offsets(anchor: torch.Tensor, offsets_m: Sequence[float]) -> torch.Tensor:
    offsets = _as_float_sequence(offsets_m)
    if not offsets:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    _, normal = _direction_from_heading(anchor)
    ramp = 0.25 + 0.75 * _ramp(anchor, power=0.5)
    variants = []
    for offset in offsets:
        traj = anchor.clone()
        traj[..., :2] = traj[..., :2] + ramp * normal * float(offset)
        variants.append(traj)
    return torch.stack(variants, dim=1)


def generate_endpoint_lateral_offsets(anchor: torch.Tensor, offsets_m: Sequence[float]) -> torch.Tensor:
    offsets = _as_float_sequence(offsets_m)
    if not offsets:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    _, normal = _direction_from_heading(anchor)
    ramp = _ramp(anchor, power=2.0)
    variants = []
    for offset in offsets:
        traj = anchor.clone()
        traj[..., :2] = traj[..., :2] + ramp * normal * float(offset)
        variants.append(traj)
    return torch.stack(variants, dim=1)


def generate_slow_first(anchor: torch.Tensor, scales: Sequence[float]) -> torch.Tensor:
    scales = _as_float_sequence(scales)
    if not scales:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    h = anchor.shape[1]
    t = torch.linspace(0.0, 1.0, h, device=anchor.device, dtype=anchor.dtype)
    variants = []
    for scale in scales:
        schedule = torch.where(
            t <= 0.5,
            t * float(scale),
            (0.5 * float(scale)) + (t - 0.5) * (1.0 - 0.5 * float(scale)) / 0.5,
        )
        positions = schedule.clamp(0.0, 1.0) * float(h - 1)
        variants.append(_linear_resample(anchor, positions))
    return torch.stack(variants, dim=1)


def generate_delay(anchor: torch.Tensor, strengths: Sequence[float]) -> torch.Tensor:
    strengths = _as_float_sequence(strengths)
    if not strengths:
        return anchor.new_empty(anchor.shape[0], 0, *anchor.shape[1:])
    ramp = _ramp(anchor)
    variants = []
    for strength in strengths:
        recovery = (1.0 - float(strength) * (1.0 - ramp).pow(2.0)).clamp(min=0.0)
        traj = anchor.clone()
        traj[..., :2] = traj[..., :2] * recovery
        variants.append(traj)
    return torch.stack(variants, dim=1)


def smooth_trajectory(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[-2] < 3:
        return traj
    xy_flat = traj[..., :2].reshape(-1, traj.shape[-2], 2).transpose(1, 2)
    xy_padded = F.pad(xy_flat, (1, 1), mode="replicate")
    xy_smoothed = F.avg_pool1d(xy_padded, kernel_size=3, stride=1).transpose(1, 2)

    heading = traj[..., 2]
    sin_flat = torch.sin(heading).reshape(-1, traj.shape[-2]).unsqueeze(1)
    cos_flat = torch.cos(heading).reshape(-1, traj.shape[-2]).unsqueeze(1)
    sin_smooth = F.avg_pool1d(F.pad(sin_flat, (1, 1), mode="replicate"), kernel_size=3, stride=1)
    cos_smooth = F.avg_pool1d(F.pad(cos_flat, (1, 1), mode="replicate"), kernel_size=3, stride=1)
    heading_smoothed = torch.atan2(sin_smooth.squeeze(1), cos_smooth.squeeze(1))

    smoothed = traj.clone()
    smoothed[..., :2] = xy_smoothed.reshape(*traj.shape[:-1], 2)
    smoothed[..., 2] = heading_smoothed.reshape(*traj.shape[:-1])
    smoothed[..., 0, :] = traj[..., 0, :]
    return smoothed


def enforce_forward_monotonic_x(traj: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    x = out[..., 0]
    eps = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) * 1e-4
    mono = torch.cummax(x + eps, dim=-1).values - eps
    out[..., 0] = mono
    return out


def clamp_heading_steps(traj: torch.Tensor, max_heading_step_rad: float) -> torch.Tensor:
    if traj.shape[-2] < 2:
        return traj
    out = traj.clone()
    headings = [out[..., 0, 2]]
    max_step = float(max_heading_step_rad)
    for idx in range(1, out.shape[-2]):
        delta = (out[..., idx, 2] - headings[-1]).clamp(-max_step, max_step)
        headings.append(headings[-1] + delta)
    out[..., 2] = torch.stack(headings, dim=-1)
    return out


def clamp_final_heading_delta(
    variants: torch.Tensor,
    anchor: torch.Tensor,
    max_delta_rad: float,
) -> torch.Tensor:
    if variants.numel() == 0:
        return variants
    out = variants.clone()
    max_delta = float(max_delta_rad)
    final_delta = out[:, :, -1, 2] - anchor[:, None, -1, 2]
    out[:, :, -1, 2] = anchor[:, None, -1, 2] + final_delta.clamp(-max_delta, max_delta)
    return out


def clip_to_model_norm_range(traj: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    out[..., 0].clamp_(-1.57, 65.17)
    out[..., 1].clamp_(-19.68, 22.32)
    out[..., 2].clamp_(-1.67, 1.86)
    return out


def _repair_candidates(candidates: torch.Tensor, cfg: Any, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    if candidates.numel() == 0:
        return candidates
    out = smooth_trajectory(candidates)
    if anchor is not None and bool(getattr(cfg, "use_final_heading_guard", True)):
        out = clamp_final_heading_delta(
            out,
            anchor,
            float(getattr(cfg, "max_final_heading_delta_rad", 0.4)),
        )
    if bool(getattr(cfg, "enforce_forward_monotonic_x", True)):
        out = enforce_forward_monotonic_x(out)
    out = clamp_heading_steps(out, float(getattr(cfg, "max_heading_step_rad", 0.25)))
    if bool(getattr(cfg, "clip_candidates_to_norm_range", True)):
        out = clip_to_model_norm_range(out)
    return out


def _append_variants(
    groups: list[torch.Tensor],
    sources: list[str],
    distances: list[torch.Tensor],
    variants: torch.Tensor,
    source: str,
    anchor: torch.Tensor,
    cfg: Any,
) -> None:
    if variants.shape[1] == 0:
        return
    variants = _repair_candidates(variants, cfg, anchor)
    groups.append(variants)
    sources.extend([source] * variants.shape[1])
    dist = torch.linalg.norm(variants[..., :2] - anchor[:, None, :, :2], dim=-1).mean(dim=-1)
    distances.append(dist)


def build_structured_perturbations(
    anchors: torch.Tensor,
    anchor_sources: List[str],
    cfg: Any,
) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    if anchors.ndim != 4 or anchors.shape[-1] != 3:
        raise ValueError(f"anchors must have shape [B, A, H, 3], got {tuple(anchors.shape)}.")
    if len(anchor_sources) != anchors.shape[1]:
        raise ValueError(f"anchor_sources length {len(anchor_sources)} does not match A={anchors.shape[1]}.")

    groups: list[torch.Tensor] = []
    sources: list[str] = []
    distances: list[torch.Tensor] = []
    for idx, source_name in enumerate(anchor_sources):
        anchor = anchors[:, idx]
        use_progress = source_name == "gt" or source_name == "il"
        if use_progress:
            _append_variants(
                groups,
                sources,
                distances,
                generate_endpoint_extension(anchor, getattr(cfg, "progress_endpoint_deltas_m", ())),
                "progress_endpoint",
                anchor,
                cfg,
            )
            _append_variants(
                groups,
                sources,
                distances,
                generate_speed_scale(anchor, getattr(cfg, "progress_speed_scales", ())),
                "progress_speed",
                anchor,
                cfg,
            )
            _append_variants(
                groups,
                sources,
                distances,
                generate_time_gamma(anchor, getattr(cfg, "progress_time_gammas", ())),
                "progress_gamma",
                anchor,
                cfg,
            )
        _append_variants(
            groups,
            sources,
            distances,
            generate_lateral_offsets(anchor, getattr(cfg, "lateral_offsets_m", ())),
            "lateral_offset",
            anchor,
            cfg,
        )
        _append_variants(
            groups,
            sources,
            distances,
            generate_endpoint_lateral_offsets(anchor, getattr(cfg, "endpoint_lateral_offsets_m", ())),
            "endpoint_lateral",
            anchor,
            cfg,
        )
        _append_variants(
            groups,
            sources,
            distances,
            generate_slow_first(anchor, getattr(cfg, "timing_slow_first_scales", ())),
            "timing_slow_first",
            anchor,
            cfg,
        )
        _append_variants(
            groups,
            sources,
            distances,
            generate_delay(anchor, getattr(cfg, "timing_delay_strengths", ())),
            "timing_delay",
            anchor,
            cfg,
        )

    if not groups:
        b, _, h, d = anchors.shape
        return anchors.new_empty(b, 0, h, d), [], anchors.new_empty(b, 0)

    candidates = torch.cat(groups, dim=1)
    anchor_distance = torch.cat(distances, dim=1)
    return candidates, sources, anchor_distance
