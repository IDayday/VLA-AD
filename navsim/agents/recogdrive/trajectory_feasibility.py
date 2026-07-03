from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FeasibilityMetrics:
    curvature_violation: torch.Tensor
    reverse_violation: torch.Tensor
    tail_reverse_violation: torch.Tensor
    early_kink_violation: torch.Tensor
    jerk_violation: torch.Tensor
    heading_jump_violation: torch.Tensor
    feas_cost: torch.Tensor
    early_kink_rate: torch.Tensor
    tail_reverse_rate: torch.Tensor
    curvature_violation_rate: torch.Tensor


def _cfg_value(cfg_or_kwargs: Any, name: str, default: Any) -> Any:
    if cfg_or_kwargs is None:
        return default
    if isinstance(cfg_or_kwargs, dict):
        return cfg_or_kwargs.get(name, default)
    return getattr(cfg_or_kwargs, name, default)


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _flatten_traj(traj: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not isinstance(traj, torch.Tensor):
        raise TypeError(f"traj must be a torch.Tensor, got {type(traj).__name__}.")
    if traj.ndim < 3 or traj.shape[-1] != 3:
        raise ValueError(f"traj must have shape [B, H, 3] or [B, K, H, 3], got {tuple(traj.shape)}.")
    prefix = tuple(traj.shape[:-2])
    flat = traj.reshape(-1, traj.shape[-2], 3)
    return flat, prefix


def _restore(value: torch.Tensor, prefix: tuple[int, ...]) -> torch.Tensor:
    return value.reshape(*prefix, *value.shape[1:])


def compute_delta_xy_heading(traj: torch.Tensor) -> dict[str, torch.Tensor]:
    flat, prefix = _flatten_traj(traj)
    if flat.shape[1] < 2:
        empty = flat.new_zeros((flat.shape[0], 0))
        return {
            "delta_xy": _restore(flat.new_zeros((flat.shape[0], 0, 2)), prefix),
            "step_distance": _restore(empty, prefix),
            "heading": _restore(flat[..., 2], prefix),
            "heading_delta": _restore(empty, prefix),
            "forward_step": _restore(empty, prefix),
            "lateral_step": _restore(empty, prefix),
        }

    delta_xy = flat[:, 1:, :2] - flat[:, :-1, :2]
    prev_heading = flat[:, :-1, 2]
    direction = torch.stack((torch.cos(prev_heading), torch.sin(prev_heading)), dim=-1)
    normal = torch.stack((-torch.sin(prev_heading), torch.cos(prev_heading)), dim=-1)
    forward_step = (delta_xy * direction).sum(dim=-1)
    lateral_step = (delta_xy * normal).sum(dim=-1)
    step_distance = torch.linalg.norm(delta_xy, dim=-1)
    heading_delta = _wrap_angle(flat[:, 1:, 2] - flat[:, :-1, 2])
    return {
        "delta_xy": _restore(delta_xy, prefix),
        "step_distance": _restore(step_distance, prefix),
        "heading": _restore(flat[..., 2], prefix),
        "heading_delta": _restore(heading_delta, prefix),
        "forward_step": _restore(forward_step, prefix),
        "lateral_step": _restore(lateral_step, prefix),
    }


def compute_curvature_proxy(traj: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    parts = compute_delta_xy_heading(traj)
    return parts["heading_delta"].abs() / parts["step_distance"].clamp_min(float(eps))


def compute_reverse_violation(
    traj: torch.Tensor,
    tail_steps: int = 3,
    eps_m: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = compute_delta_xy_heading(traj)
    forward_step = parts["forward_step"]
    reverse_step = (-forward_step - float(eps_m)).clamp_min(0.0)
    if reverse_step.shape[-1] == 0:
        zeros = reverse_step.sum(dim=-1)
        return zeros, zeros
    reverse_violation = reverse_step.mean(dim=-1)
    tail_n = min(max(int(tail_steps), 1), reverse_step.shape[-1])
    tail_reverse_violation = reverse_step[..., -tail_n:].mean(dim=-1)
    return reverse_violation, tail_reverse_violation


def compute_early_kink_violation(
    traj: torch.Tensor,
    first_steps: int = 2,
    max_heading_delta_rad: float = 0.35,
) -> torch.Tensor:
    parts = compute_delta_xy_heading(traj)
    heading_delta = parts["heading_delta"].abs()
    if heading_delta.shape[-1] == 0:
        return heading_delta.sum(dim=-1)
    n = min(max(int(first_steps), 1), heading_delta.shape[-1])
    return (heading_delta[..., :n] - float(max_heading_delta_rad)).clamp_min(0.0).mean(dim=-1)


def compute_jerk_proxy(traj: torch.Tensor) -> torch.Tensor:
    flat, prefix = _flatten_traj(traj)
    if flat.shape[1] < 3:
        return _restore(flat.new_zeros((flat.shape[0],)), prefix)
    velocity = flat[:, 1:, :2] - flat[:, :-1, :2]
    accel = velocity[:, 1:, :] - velocity[:, :-1, :]
    jerk = torch.linalg.norm(accel, dim=-1)
    return _restore(jerk, prefix)


def _mean_last(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] == 0:
        return value.sum(dim=-1)
    return value.mean(dim=-1)


def compute_feasibility_metrics(traj: torch.Tensor, cfg_or_kwargs: Any) -> FeasibilityMetrics:
    max_curv = float(_cfg_value(cfg_or_kwargs, "max_curvature_proxy", 0.35))
    max_jerk = float(_cfg_value(cfg_or_kwargs, "max_jerk_proxy", 2.5))
    max_heading_jump = float(_cfg_value(cfg_or_kwargs, "max_heading_jump_rad", 0.70))
    tail_steps = int(_cfg_value(cfg_or_kwargs, "tail_reverse_steps", _cfg_value(cfg_or_kwargs, "tail_steps", 3)))
    eps_m = float(_cfg_value(cfg_or_kwargs, "reverse_eps_m", 0.05))
    first_steps = int(_cfg_value(cfg_or_kwargs, "early_kink_first_steps", 2))
    max_early_heading = float(_cfg_value(cfg_or_kwargs, "max_early_heading_delta_rad", 0.35))

    curvature = compute_curvature_proxy(traj)
    curvature_violation_steps = (curvature - max_curv).clamp_min(0.0)
    curvature_violation = _mean_last(curvature_violation_steps)
    curvature_violation_rate = _mean_last((curvature_violation_steps > 0.0).to(dtype=curvature.dtype))

    reverse_violation, tail_reverse_violation = compute_reverse_violation(traj, tail_steps=tail_steps, eps_m=eps_m)
    parts = compute_delta_xy_heading(traj)
    forward_step = parts["forward_step"]
    if forward_step.shape[-1] == 0:
        tail_reverse_rate = forward_step.sum(dim=-1)
    else:
        tail_n = min(max(tail_steps, 1), forward_step.shape[-1])
        tail_reverse_rate = ((-forward_step[..., -tail_n:] - eps_m) > 0.0).to(dtype=forward_step.dtype).mean(dim=-1)

    early_kink_violation = compute_early_kink_violation(
        traj,
        first_steps=first_steps,
        max_heading_delta_rad=max_early_heading,
    )
    heading_delta = parts["heading_delta"].abs()
    if heading_delta.shape[-1] == 0:
        early_kink_rate = heading_delta.sum(dim=-1)
    else:
        n = min(max(first_steps, 1), heading_delta.shape[-1])
        early_kink_rate = (heading_delta[..., :n] > max_early_heading).to(dtype=heading_delta.dtype).mean(dim=-1)

    jerk = compute_jerk_proxy(traj)
    jerk_violation = _mean_last((jerk - max_jerk).clamp_min(0.0)) if jerk.ndim > traj.ndim - 2 else jerk
    heading_jump_violation = _mean_last((heading_delta - max_heading_jump).clamp_min(0.0))

    w_curv = float(_cfg_value(cfg_or_kwargs, "w_curv", 1.0))
    w_reverse = float(_cfg_value(cfg_or_kwargs, "w_reverse", 1.0))
    w_tail_reverse = float(_cfg_value(cfg_or_kwargs, "w_tail_reverse", 2.0))
    w_kink = float(_cfg_value(cfg_or_kwargs, "w_kink", 2.0))
    w_jerk = float(_cfg_value(cfg_or_kwargs, "w_jerk", 0.2))
    w_heading = float(_cfg_value(cfg_or_kwargs, "w_heading", 1.0))

    feas_cost = (
        w_curv * curvature_violation
        + w_reverse * reverse_violation
        + w_tail_reverse * tail_reverse_violation
        + w_kink * early_kink_violation
        + w_jerk * jerk_violation
        + w_heading * heading_jump_violation
    )
    return FeasibilityMetrics(
        curvature_violation=curvature_violation,
        reverse_violation=reverse_violation,
        tail_reverse_violation=tail_reverse_violation,
        early_kink_violation=early_kink_violation,
        jerk_violation=jerk_violation,
        heading_jump_violation=heading_jump_violation,
        feas_cost=feas_cost,
        early_kink_rate=early_kink_rate,
        tail_reverse_rate=tail_reverse_rate,
        curvature_violation_rate=curvature_violation_rate,
    )


def cheap_filter_mask(traj: torch.Tensor, cfg_or_kwargs: Any) -> torch.Tensor:
    metrics = compute_feasibility_metrics(traj, cfg_or_kwargs)
    max_feas_cost = float(_cfg_value(cfg_or_kwargs, "cheap_filter_max_feas_cost", math.inf))
    max_tail_reverse_rate = float(_cfg_value(cfg_or_kwargs, "cheap_filter_max_tail_reverse_rate", 0.5))
    max_early_kink_rate = float(_cfg_value(cfg_or_kwargs, "cheap_filter_max_early_kink_rate", 0.5))
    max_curvature_violation_rate = float(_cfg_value(cfg_or_kwargs, "cheap_filter_max_curvature_violation_rate", 0.75))
    return (
        (metrics.feas_cost <= max_feas_cost)
        & (metrics.tail_reverse_rate <= max_tail_reverse_rate)
        & (metrics.early_kink_rate <= max_early_kink_rate)
        & (metrics.curvature_violation_rate <= max_curvature_violation_rate)
    )
