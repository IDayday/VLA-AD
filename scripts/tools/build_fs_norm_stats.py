#!/usr/bin/env python
from __future__ import annotations

import argparse
import lzma
import pickle
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.fs_norm import FSNormStats, FSNormTransform, save_fs_norm_stats


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value_l = value.lower()
    if value_l in {"1", "true", "yes", "y"}:
        return True
    if value_l in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def _load_record(path: Path) -> Any:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported archive/stat input file: {path}")


def _collect_trajs(payload: Any) -> list[torch.Tensor]:
    trajs: list[torch.Tensor] = []
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
        support_tags = payload.get("support_tags")
        if candidates is not None and support_tags is not None:
            arr = torch.as_tensor(candidates, dtype=torch.float32)
            tags = [str(tag) for tag in support_tags]
            if arr.ndim == 3 and arr.shape[-1] == 3 and len(tags) == int(arr.shape[0]):
                selected = [idx for idx, tag in enumerate(tags) if tag]
                if selected:
                    return [arr[idx] for idx in selected]

        for key in ("trajectory", "gt_trajectory", "il_trajectory"):
            if key in payload:
                arr = torch.as_tensor(payload[key], dtype=torch.float32)
                if arr.ndim == 2 and arr.shape[-1] == 3:
                    trajs.append(arr)
                elif arr.ndim == 3 and arr.shape[-1] == 3:
                    trajs.extend([row for row in arr])
        for key in ("candidates", "support_trajectories", "trajectories"):
            if key in payload:
                arr = torch.as_tensor(payload[key], dtype=torch.float32)
                if arr.ndim == 2 and arr.shape[-1] == 3:
                    trajs.append(arr)
                elif arr.ndim == 3 and arr.shape[-1] == 3:
                    trajs.extend([row for row in arr])
        return trajs
    if isinstance(payload, list):
        for item in payload:
            trajs.extend(_collect_trajs(item))
        return trajs
    arr = torch.as_tensor(payload, dtype=torch.float32)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return [arr]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return [row for row in arr]
    return []


def iter_archive_paths(root: Path):
    if root.is_file():
        yield root
        return
    for pattern in ("*.pkl.xz", "*.pt", "*.pkl"):
        yield from sorted(root.rglob(pattern))


def build_stats(
    support_archive_path: str,
    use_robust: bool,
    clip: float = 5.0,
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
) -> FSNormStats:
    paths = list(iter_archive_paths(Path(support_archive_path)))
    trajs: list[torch.Tensor] = []
    for path in paths:
        try:
            trajs.extend(_collect_trajs(_load_record(path)))
        except Exception as exc:
            raise RuntimeError(f"Failed loading trajectories from {path}: {exc}") from exc
    if not trajs:
        raise ValueError(f"No [H, 3] trajectories found under {support_archive_path}.")
    stacked = torch.stack(trajs, dim=0)
    delta = FSNormTransform(FSNormStats(mean=torch.zeros_like(stacked[0]), std=torch.ones_like(stacked[0]))).absolute_to_delta(stacked)
    mean = delta.mean(dim=0)
    std = delta.std(dim=0, unbiased=False).clamp_min(1e-6)
    delta_min = delta.amin(dim=0)
    delta_max = delta.amax(dim=0)
    median = None
    mad = None
    if use_robust:
        median = delta.median(dim=0).values
        mad = (delta - median).abs().median(dim=0).values.clamp_min(1e-6)
        center = median
        scale = (1.4826 * mad).clamp_min(1e-6)
    else:
        center = mean
        scale = std
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            f"Expected 0 <= lower_quantile < upper_quantile <= 1, got "
            f"{lower_quantile}, {upper_quantile}."
        )
    normalized = (delta - center.unsqueeze(0)) / scale.unsqueeze(0)
    clip_lower = torch.quantile(normalized, float(lower_quantile), dim=0)
    clip_upper = torch.quantile(normalized, float(upper_quantile), dim=0)
    return FSNormStats(
        mean=mean,
        std=std,
        median=median,
        mad=mad,
        delta_min=delta_min,
        delta_max=delta_max,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        use_robust=use_robust,
        clip=float(clip),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FS-Norm step-wise delta statistics from support archives.")
    parser.add_argument("--support_archive_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--use_robust", type=_str_to_bool, default=False)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--lower_quantile", type=float, default=0.001)
    parser.add_argument("--upper_quantile", type=float, default=0.999)
    args = parser.parse_args()

    stats = build_stats(
        args.support_archive_path,
        use_robust=args.use_robust,
        clip=args.clip,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
    )
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_fs_norm_stats(str(output), stats)
    print(
        f"saved FS-Norm stats to {output} "
        f"(H={stats.mean.shape[0]}, D={stats.mean.shape[1]}, robust={stats.use_robust}, "
        f"clip_quantiles=[{args.lower_quantile}, {args.upper_quantile}])"
    )


if __name__ == "__main__":
    main()
