#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.support_aligned_diversity import (  # noqa: E402
    SupportAlignedDiversityConfig,
    compute_support_aligned_diversity,
    load_support_reference,
    support_archive_path,
)
from scripts.eval_recogdrive_expert_pdm import (  # noqa: E402
    build_planner,
    dtype_from_precision,
    load_checkpoint,
    load_eval_sample,
    load_yaml,
    make_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ReCogDrive policy distribution against a positive multi-trajectory support archive."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hidden-cache-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-manifest-only", action="store_true")
    parser.add_argument("--num-scenes", type=int, default=1024)
    parser.add_argument("--samples-per-scene", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260711)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--deterministic-sampler", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--mode-threshold", type=float, default=0.40)
    parser.add_argument("--density-bandwidth", type=float, default=0.40)
    parser.add_argument("--hard-match-threshold", type=float, default=0.50)
    return parser.parse_args()


def _stable_order_key(token: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest()


def prepare_manifest(args: argparse.Namespace) -> None:
    if not args.hidden_cache_root.is_dir():
        raise FileNotFoundError(args.hidden_cache_root)
    if not args.support_root.is_dir():
        raise FileNotFoundError(args.support_root)
    records: list[tuple[str, Path]] = []
    for feature_path in args.hidden_cache_root.glob("*/*/internvl_feature.gz"):
        token = feature_path.parent.name
        if support_archive_path(args.support_root, token).is_file():
            records.append((token, feature_path.parent))
    records.sort(key=lambda item: (_stable_order_key(item[0], args.seed), item[0]))
    if args.num_scenes > 0:
        records = records[: args.num_scenes]
    if not records:
        raise RuntimeError("No hidden-cache tokens have matching support archives.")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as stream:
        stream.write("token\tcache_path\n")
        for token, cache_path in records:
            stream.write(f"{token}\t{cache_path}\n")
    print(json.dumps({"manifest": str(args.manifest), "num_scenes": len(records)}, sort_keys=True))


def load_manifest(path: Path) -> list[tuple[str, Path]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[tuple[str, Path]] = []
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["token", "cache_path"]:
            raise ValueError(f"Unexpected manifest columns {reader.fieldnames}: {path}")
        for row in reader:
            records.append((str(row["token"]), Path(row["cache_path"])))
    if not records:
        raise RuntimeError(f"Manifest is empty: {path}")
    if len({token for token, _ in records}) != len(records):
        raise ValueError(f"Manifest contains duplicate tokens: {path}")
    return records


def _repeat_batch(
    vl_features: torch.Tensor,
    action_input: BatchFeature,
    batch_size: int,
) -> tuple[torch.Tensor, BatchFeature]:
    if vl_features.shape[0] != 1:
        raise ValueError(f"Expected a one-scene VLM batch, got {tuple(vl_features.shape)}.")

    def repeat_tensor(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            return value
        if value.shape[0] != 1:
            raise ValueError(f"Expected one-scene condition tensor, got {tuple(value.shape)}.")
        return value.expand((batch_size,) + tuple(value.shape[1:]))

    repeated = {
        key: repeat_tensor(value) if isinstance(value, torch.Tensor) else value
        for key, value in action_input.items()
    }
    return repeat_tensor(vl_features), BatchFeature(data=repeated)


def _scene_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


def sample_policy(
    planner,
    sample: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    samples_per_scene: int,
    sample_batch_size: int,
    deterministic: bool,
    seed: int,
) -> np.ndarray:
    vl_features, action_input = make_batch(sample, planner, device, dtype)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    predictions: list[np.ndarray] = []
    remaining = samples_per_scene
    while remaining > 0:
        current_batch = min(sample_batch_size, remaining)
        batch_vl, batch_input = _repeat_batch(vl_features, action_input, current_batch)
        with torch.inference_mode():
            output = planner.get_action(batch_vl, batch_input, deterministic=deterministic)
        trajectory = output["pred_traj"].detach().float().cpu().numpy()
        if trajectory.shape != (current_batch, 8, 3) or not np.isfinite(trajectory).all():
            raise RuntimeError(f"Invalid sampled trajectories with shape {trajectory.shape}.")
        predictions.append(trajectory)
        remaining -= current_batch
    return np.concatenate(predictions, axis=0)


def _gt_metrics(predictions: np.ndarray, target: Any) -> dict[str, float]:
    gt = np.asarray(target, dtype=np.float64)
    if gt.shape != (8, 3):
        return {}
    xy_distance = np.linalg.norm(predictions[..., :2] - gt[None, ..., :2], axis=-1)
    return {
        "mean_gt_ade_m": float(xy_distance.mean(axis=1).mean()),
        "best_gt_ade_m": float(xy_distance.mean(axis=1).min()),
        "mean_gt_fde_m": float(xy_distance[:, -1].mean()),
        "best_gt_fde_m": float(xy_distance[:, -1].min()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.prepare_manifest_only:
        prepare_manifest(args)
        return 0
    if args.checkpoint is None or args.config is None:
        raise ValueError("--checkpoint and --config are required for inference.")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if args.samples_per_scene <= 0 or args.sample_batch_size <= 0:
        raise ValueError("Sampling counts must be positive.")

    all_records = load_manifest(args.manifest)
    records = [record for index, record in enumerate(all_records) if index % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SupportAlignedDiversityConfig(
        density_bandwidth=args.density_bandwidth,
        mode_threshold=args.mode_threshold,
        hard_match_threshold=args.hard_match_threshold,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = build_planner(load_yaml(args.config)).to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    load_checkpoint(planner, args.checkpoint)
    planner.eval()

    rows: list[dict[str, Any]] = []
    saved_tokens: list[str] = []
    saved_predictions: list[np.ndarray] = []
    start_time = time.monotonic()
    for index, (token, cache_path) in enumerate(records):
        sample = load_eval_sample(cache_path)
        support = load_support_reference(args.support_root, token)
        predictions = sample_policy(
            planner,
            sample,
            device=device,
            dtype=dtype,
            samples_per_scene=args.samples_per_scene,
            sample_batch_size=args.sample_batch_size,
            deterministic=args.deterministic_sampler,
            seed=_scene_seed(args.seed, token),
        )
        metrics = compute_support_aligned_diversity(predictions, support.trajectories, config)
        row: dict[str, Any] = {
            "token": token,
            "archive_version": support.archive_version,
            "support_reward_mean": float(support.rewards.mean()),
            "support_reward_max": float(support.rewards.max()),
            "support_has_gt_anchor": float(any(tag == "gt_anchor" for tag in support.tags)),
            **metrics,
            **_gt_metrics(predictions, sample.get("trajectory")),
        }
        rows.append(row)
        if args.save_predictions:
            saved_tokens.append(token)
            saved_predictions.append(predictions.astype(np.float32))
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            elapsed = time.monotonic() - start_time
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "completed": index + 1,
                        "total": len(records),
                        "elapsed_seconds": round(elapsed, 2),
                        "snsad_running_mean": float(np.mean([item["snsad"] for item in rows])),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output_csv = args.output_dir / "scene_metrics.csv"
    _write_csv(output_csv, rows)
    if args.save_predictions:
        np.savez_compressed(
            args.output_dir / "predictions.npz",
            tokens=np.asarray(saved_tokens),
            trajectories=np.stack(saved_predictions),
        )
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "manifest": str(args.manifest),
        "support_root": str(args.support_root),
        "samples_per_scene": args.samples_per_scene,
        "sample_batch_size": args.sample_batch_size,
        "seed": args.seed,
        "deterministic_sampler": args.deterministic_sampler,
        "precision": args.precision,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_scenes": len(rows),
        "elapsed_seconds": time.monotonic() - start_time,
        "metric_config": config.__dict__,
        "scene_metrics_csv": str(output_csv),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
