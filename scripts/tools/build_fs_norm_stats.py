#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import lzma
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.fs_norm import FSNormStats, save_fs_norm_stats
from navsim.agents.recogdrive.pareto_support.fs_norm import fit_fs_norm_stats as fit_numpy_fs_norm_stats


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
    support_set = getattr(payload, "support_set", None)
    if support_set is not None:
        return [torch.as_tensor(item.trajectory, dtype=torch.float32) for item in support_set]
    if isinstance(payload, dict):
        support_set = payload.get("support_set")
        if support_set is not None:
            for item in support_set:
                trajectory = getattr(item, "trajectory", None)
                if trajectory is None and isinstance(item, dict):
                    trajectory = item.get("trajectory")
                if trajectory is not None:
                    trajs.append(torch.as_tensor(trajectory, dtype=torch.float32))
            return trajs
        candidates = payload.get("candidates")
        support_indices = payload.get("support_indices")
        if candidates is not None and support_indices is not None:
            arr = torch.as_tensor(candidates, dtype=torch.float32)
            indices = [int(index) for index in support_indices]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                if any(index < 0 or index >= int(arr.shape[0]) for index in indices):
                    raise ValueError("support_indices contain out-of-range candidate indices.")
                return [arr[index] for index in indices]

        if "support_trajectories" in payload:
            arr = torch.as_tensor(payload["support_trajectories"], dtype=torch.float32)
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
    scene_ids: list[str] = []
    fingerprint = hashlib.sha256()
    for path in paths:
        try:
            scene_trajs = _collect_trajs(_load_record(path))
        except Exception as exc:
            raise RuntimeError(f"Failed loading trajectories from {path}: {exc}") from exc
        if not scene_trajs:
            continue
        trajs.extend(scene_trajs)
        scene_ids.extend([str(path)] * len(scene_trajs))
        stat = path.stat()
        fingerprint.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
    if not trajs:
        raise ValueError(f"No [H, 3] trajectories found under {support_archive_path}.")
    stacked = torch.stack(trajs, dim=0).numpy()
    numpy_stats = fit_numpy_fs_norm_stats(
        stacked,
        robust=use_robust,
        clip=clip,
        scene_ids=np.asarray(scene_ids),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        archive_path=str(Path(support_archive_path).resolve()),
        archive_fingerprint=fingerprint.hexdigest(),
    )

    def tensor_or_none(value):
        return None if value is None else torch.as_tensor(value, dtype=torch.float32)

    return FSNormStats(
        mean=torch.as_tensor(numpy_stats.mean, dtype=torch.float32),
        std=torch.as_tensor(numpy_stats.std, dtype=torch.float32),
        median=tensor_or_none(numpy_stats.median),
        mad=tensor_or_none(numpy_stats.mad),
        delta_min=tensor_or_none(numpy_stats.delta_min),
        delta_max=tensor_or_none(numpy_stats.delta_max),
        clip_lower=tensor_or_none(numpy_stats.clip_lower),
        clip_upper=tensor_or_none(numpy_stats.clip_upper),
        use_robust=numpy_stats.use_robust,
        clip=numpy_stats.clip,
        version=numpy_stats.version,
        representation=numpy_stats.representation,
        p0=numpy_stats.p0,
        scene_balanced=numpy_stats.scene_balanced,
        heading_center_zero=numpy_stats.heading_center_zero,
        num_scenes=numpy_stats.num_scenes,
        num_supports=numpy_stats.num_supports,
        archive_path=numpy_stats.archive_path,
        archive_fingerprint=numpy_stats.archive_fingerprint,
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
        f"(version={stats.version}, scenes={stats.num_scenes}, supports={stats.num_supports}, "
        f"H={stats.mean.shape[0]}, D={stats.mean.shape[1]}, robust={stats.use_robust}, "
        f"clip_quantiles=[{args.lower_quantile}, {args.upper_quantile}])"
    )


if __name__ == "__main__":
    main()
