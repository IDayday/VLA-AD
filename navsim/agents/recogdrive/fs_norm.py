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
    version: int = 1
    representation: str = "ego_step_delta"
    p0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scene_balanced: bool = False
    heading_center_zero: bool = False
    num_scenes: int = 0
    num_supports: int = 0
    archive_path: str = ""
    archive_fingerprint: str = ""


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
            if self.stats.version >= 2 and self.stats.heading_center_zero:
                center = center.clone()
                scale = scale.clone()
                center[..., 2] = 0.0
                scale[..., 2] = self._to(self.stats.mad, ref)[..., 2]
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
        *,
        apply_clip: bool = False,
        clip_value: Optional[float] = None,
        use_stats_bounds: bool = False,
    ) -> torch.Tensor:
        center, scale = self._center_scale(traj_delta)
        normalized = (traj_delta - center) / scale
        if not apply_clip:
            if clip_value is not None or use_stats_bounds:
                raise ValueError("clip_value/use_stats_bounds require apply_clip=True.")
            return normalized
        if use_stats_bounds:
            if clip_value is not None:
                raise ValueError("Choose either stats bounds or scalar clip_value, not both.")
            bounds = self._clip_bounds(normalized)
            if bounds is None:
                raise ValueError("FS-Norm stats bounds requested but clip_lower/clip_upper are unavailable.")
            lower, upper = bounds
            return torch.minimum(torch.maximum(normalized, lower), upper)
        if clip_value is None or float(clip_value) <= 0.0:
            raise ValueError("apply_clip=True requires a positive clip_value or use_stats_bounds=True.")
        return normalized.clamp(min=-float(clip_value), max=float(clip_value))

    def denormalize_delta(self, traj_norm: torch.Tensor) -> torch.Tensor:
        center, scale = self._center_scale(traj_norm)
        return traj_norm * scale + center

    def encode(
        self,
        traj_abs: torch.Tensor,
        *,
        apply_clip: bool = False,
        clip_value: Optional[float] = None,
        use_stats_bounds: bool = False,
    ) -> torch.Tensor:
        return self.normalize_delta(
            self.absolute_to_delta(traj_abs),
            apply_clip=apply_clip,
            clip_value=clip_value,
            use_stats_bounds=use_stats_bounds,
        )

    def decode(self, traj_norm: torch.Tensor) -> torch.Tensor:
        return self.delta_to_absolute(self.denormalize_delta(traj_norm))


def load_fs_norm_stats(path: str, *, map_location: str | torch.device = "cpu") -> FSNormStats:
    path_obj = Path(path)
    if path_obj.suffix == ".npz":
        data = np.load(path_obj, allow_pickle=True)
        device = torch.device(map_location) if isinstance(map_location, str) and map_location != "cpu" else None
        median_arr = data["median"] if "median" in data.files and data["median"].size else None
        mad_arr = data["mad"] if "mad" in data.files and data["mad"].size else None
        p0_arr = data["p0"] if "p0" in data.files else np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
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
            version=int(data["version"]) if "version" in data.files else 1,
            representation=str(data["representation"].item()) if "representation" in data.files else "ego_step_delta",
            p0=tuple(float(item) for item in np.asarray(p0_arr).reshape(-1)[:3]),
            scene_balanced=bool(int(data["scene_balanced"])) if "scene_balanced" in data.files else False,
            heading_center_zero=bool(int(data["heading_center_zero"])) if "heading_center_zero" in data.files else False,
            num_scenes=int(data["num_scenes"]) if "num_scenes" in data.files else 0,
            num_supports=int(data["num_supports"]) if "num_supports" in data.files else 0,
            archive_path=str(data["archive_path"].item()) if "archive_path" in data.files else "",
            archive_fingerprint=(
                str(data["archive_fingerprint"].item()) if "archive_fingerprint" in data.files else ""
            ),
        )
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
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
        version=int(payload.get("version", 1)),
        representation=str(payload.get("representation", "ego_step_delta")),
        p0=tuple(float(item) for item in payload.get("p0", (0.0, 0.0, 0.0))),
        scene_balanced=bool(payload.get("scene_balanced", False)),
        heading_center_zero=bool(payload.get("heading_center_zero", False)),
        num_scenes=int(payload.get("num_scenes", 0)),
        num_supports=int(payload.get("num_supports", 0)),
        archive_path=str(payload.get("archive_path", "")),
        archive_fingerprint=str(payload.get("archive_fingerprint", "")),
    )


def save_fs_norm_stats(path: str, stats: FSNormStats) -> None:
    path_obj = Path(path)
    if path_obj.suffix == ".npz":
        def array_or_empty(value: Optional[torch.Tensor]) -> np.ndarray:
            if value is None:
                return np.asarray([], dtype=np.float32)
            return value.detach().cpu().numpy().astype(np.float32, copy=False)

        np.savez(
            path_obj,
            mean=array_or_empty(stats.mean),
            std=array_or_empty(stats.std),
            median=array_or_empty(stats.median),
            mad=array_or_empty(stats.mad),
            delta_min=array_or_empty(stats.delta_min),
            delta_max=array_or_empty(stats.delta_max),
            clip_lower=array_or_empty(stats.clip_lower),
            clip_upper=array_or_empty(stats.clip_upper),
            use_robust=np.asarray(int(stats.use_robust), dtype=np.int64),
            clip=np.asarray(float(stats.clip), dtype=np.float32),
            version=np.asarray(int(stats.version), dtype=np.int64),
            representation=np.asarray(str(stats.representation)),
            p0=np.asarray(stats.p0, dtype=np.float32),
            scene_balanced=np.asarray(int(stats.scene_balanced), dtype=np.int64),
            heading_center_zero=np.asarray(int(stats.heading_center_zero), dtype=np.int64),
            num_scenes=np.asarray(int(stats.num_scenes), dtype=np.int64),
            num_supports=np.asarray(int(stats.num_supports), dtype=np.int64),
            archive_path=np.asarray(str(stats.archive_path)),
            archive_fingerprint=np.asarray(str(stats.archive_fingerprint)),
        )
        return
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
        "version": int(stats.version),
        "representation": str(stats.representation),
        "p0": tuple(float(item) for item in stats.p0),
        "scene_balanced": bool(stats.scene_balanced),
        "heading_center_zero": bool(stats.heading_center_zero),
        "num_scenes": int(stats.num_scenes),
        "num_supports": int(stats.num_supports),
        "archive_path": str(stats.archive_path),
        "archive_fingerprint": str(stats.archive_fingerprint),
    }
    torch.save(payload, path_obj)
