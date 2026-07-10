from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class FSNormStats:
    mean: torch.Tensor
    std: torch.Tensor
    median: Optional[torch.Tensor] = None
    mad: Optional[torch.Tensor] = None
    delta_min: Optional[torch.Tensor] = None
    delta_max: Optional[torch.Tensor] = None
    clip_lower: Optional[torch.Tensor] = None
    clip_upper: Optional[torch.Tensor] = None
    use_robust: bool = False
    clip: float = 5.0


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class FSNormTransform:
    def __init__(self, stats: FSNormStats, eps: float = 1e-6):
        self.stats = stats
        self.eps = float(eps)

    def _to(self, value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return value.to(device=ref.device, dtype=ref.dtype)

    def _center_scale(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stats.use_robust and self.stats.median is not None and self.stats.mad is not None:
            center = self._to(self.stats.median, ref)
            scale = 1.4826 * self._to(self.stats.mad, ref)
        else:
            center = self._to(self.stats.mean, ref)
            scale = self._to(self.stats.std, ref)
        return center, scale.clamp_min(self.eps)

    def absolute_to_delta(self, traj_abs: torch.Tensor) -> torch.Tensor:
        if traj_abs.ndim < 2 or traj_abs.shape[-1] != 3:
            raise ValueError(f"traj_abs must end with [H, 3], got {tuple(traj_abs.shape)}.")
        delta = traj_abs.clone()
        if traj_abs.shape[-2] > 1:
            delta[..., 1:, :2] = traj_abs[..., 1:, :2] - traj_abs[..., :-1, :2]
            delta[..., 1:, 2] = wrap_angle(traj_abs[..., 1:, 2] - traj_abs[..., :-1, 2])
        delta[..., 0, 2] = wrap_angle(delta[..., 0, 2])
        return delta

    def delta_to_absolute(self, traj_delta: torch.Tensor) -> torch.Tensor:
        if traj_delta.ndim < 2 or traj_delta.shape[-1] != 3:
            raise ValueError(f"traj_delta must end with [H, 3], got {tuple(traj_delta.shape)}.")
        traj_abs = traj_delta.clone()
        traj_abs[..., :2] = torch.cumsum(traj_delta[..., :2], dim=-2)
        traj_abs[..., 2] = wrap_angle(torch.cumsum(traj_delta[..., 2], dim=-1))
        return traj_abs

    def _clip_bounds(self, ref: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.stats.clip_lower is None or self.stats.clip_upper is None:
            return None
        lower = self._to(self.stats.clip_lower, ref)
        upper = self._to(self.stats.clip_upper, ref)
        return lower, upper

    def normalize_delta(
        self,
        traj_delta: torch.Tensor,
        clip: Optional[float] = None,
        use_stats_bounds: bool = False,
    ) -> torch.Tensor:
        center, scale = self._center_scale(traj_delta)
        normalized = (traj_delta - center) / scale
        if use_stats_bounds:
            bounds = self._clip_bounds(normalized)
            if bounds is not None:
                lower, upper = bounds
                return torch.minimum(torch.maximum(normalized, lower), upper)
        clip_value = float(self.stats.clip if clip is None else clip)
        if clip_value > 0.0:
            normalized = normalized.clamp(min=-clip_value, max=clip_value)
        return normalized

    def denormalize_delta(self, traj_norm: torch.Tensor) -> torch.Tensor:
        center, scale = self._center_scale(traj_norm)
        return traj_norm * scale + center

    def encode(
        self,
        traj_abs: torch.Tensor,
        clip: Optional[float] = None,
        use_stats_bounds: bool = False,
    ) -> torch.Tensor:
        return self.normalize_delta(self.absolute_to_delta(traj_abs), clip=clip, use_stats_bounds=use_stats_bounds)

    def decode(self, traj_norm: torch.Tensor) -> torch.Tensor:
        return self.delta_to_absolute(self.denormalize_delta(traj_norm))


def load_fs_norm_stats(path: str, *, map_location: str | torch.device = "cpu") -> FSNormStats:
    path_obj = Path(path)
    if path_obj.suffix == ".npz":
        data = np.load(path_obj, allow_pickle=True)
        device = torch.device(map_location) if isinstance(map_location, str) and map_location != "cpu" else None
        median_arr = data["median"] if "median" in data.files and data["median"].size else None
        mad_arr = data["mad"] if "mad" in data.files and data["mad"].size else None
        return FSNormStats(
            mean=torch.as_tensor(data["mean"], dtype=torch.float32, device=device),
            std=torch.as_tensor(data["std"], dtype=torch.float32, device=device),
            median=None if median_arr is None else torch.as_tensor(median_arr, dtype=torch.float32, device=device),
            mad=None if mad_arr is None else torch.as_tensor(mad_arr, dtype=torch.float32, device=device),
            delta_min=torch.as_tensor(data["delta_min"], dtype=torch.float32, device=device) if "delta_min" in data.files else None,
            delta_max=torch.as_tensor(data["delta_max"], dtype=torch.float32, device=device) if "delta_max" in data.files else None,
            clip_lower=torch.as_tensor(data["clip_lower"], dtype=torch.float32, device=device) if "clip_lower" in data.files else None,
            clip_upper=torch.as_tensor(data["clip_upper"], dtype=torch.float32, device=device) if "clip_upper" in data.files else None,
            use_robust=bool(int(data["use_robust"])) if "use_robust" in data.files else False,
            clip=float(data["clip"]) if "clip" in data.files else 5.0,
        )
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, FSNormStats):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"FS-Norm stats must be a dict or FSNormStats, got {type(payload).__name__}.")
    return FSNormStats(
        mean=payload["mean"],
        std=payload["std"],
        median=payload.get("median"),
        mad=payload.get("mad"),
        delta_min=payload.get("delta_min"),
        delta_max=payload.get("delta_max"),
        clip_lower=payload.get("clip_lower"),
        clip_upper=payload.get("clip_upper"),
        use_robust=bool(payload.get("use_robust", False)),
        clip=float(payload.get("clip", 5.0)),
    )


def save_fs_norm_stats(path: str, stats: FSNormStats) -> None:
    payload = {
        "mean": stats.mean.detach().cpu(),
        "std": stats.std.detach().cpu(),
        "median": None if stats.median is None else stats.median.detach().cpu(),
        "mad": None if stats.mad is None else stats.mad.detach().cpu(),
        "delta_min": None if stats.delta_min is None else stats.delta_min.detach().cpu(),
        "delta_max": None if stats.delta_max is None else stats.delta_max.detach().cpu(),
        "clip_lower": None if stats.clip_lower is None else stats.clip_lower.detach().cpu(),
        "clip_upper": None if stats.clip_upper is None else stats.clip_upper.detach().cpu(),
        "use_robust": bool(stats.use_robust),
        "clip": float(stats.clip),
    }
    torch.save(payload, path)
