#!/usr/bin/env python3
"""Validate NAVSIM inference JSON, image paths, logs, and metric-cache alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("/mnt/project/OneVL_training"))
    parser.add_argument("--navsim-log-path", type=Path, default=Path("/mnt/project/onevl_navsim_data/navsim_logs/test"))
    parser.add_argument("--metric-cache-path", type=Path, default=Path("/mnt/project/onevl_navsim_exp/metric_cache"))
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("/mnt/project/onevl_navsim_data/navsim_test_alignment.report.json"),
    )
    parser.add_argument("--limit-image-check", type=int, default=0)
    parser.add_argument("--allow-missing-cache-tokens", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open(encoding="utf-8") as src:
        data = json.load(src)
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported test-set shape: {path}")
    return data


def normalize_image_path(path: str) -> str:
    path = path.replace("file://", "")
    markers = [
        "navsim_v1.1_all/dataset/sensor_blobs/test/",
        "/sensor_blobs/test/",
        "/test_sensor_blobs/test/",
    ]
    for marker in markers:
        if marker in path:
            return path.split(marker, 1)[1]
    parts = Path(path).parts
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return path


def first_image(row: dict[str, Any]) -> str:
    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    messages = row.get("messages") or []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    return str(part.get("image") or "")
    return ""


def resolve_image(path: str, repo_root: Path) -> Path:
    image = Path(path.replace("file://", ""))
    if image.is_absolute():
        return image
    return repo_root / image


def build_image_to_token(navsim_log_path: Path) -> dict[str, str]:
    with initialize_config_dir(config_dir="/mnt/navsim/planning/script/config/metric_caching", version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=["train_test_split=navtest", "worker=sequential"])
    cfg.navsim_log_path = str(navsim_log_path)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        data_path=navsim_log_path,
        sensor_blobs_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    current_index = scene_filter.num_history_frames - 1
    image_to_token = {}
    for token, frames in scene_loader.scene_frames_dicts.items():
        cam_path = frames[current_index]["cams"]["CAM_F0"]["data_path"]
        image_to_token[normalize_image_path(cam_path)] = token
    return image_to_token


def main() -> None:
    args = parse_args()
    rows = load_rows(args.test_set_path)
    image_paths = [first_image(row) for row in rows]
    image_keys = [normalize_image_path(path) for path in image_paths if path]

    missing_image_field = len(rows) - len(image_keys)
    duplicate_image_keys = len(image_keys) - len(set(image_keys))
    image_check_paths = image_paths[: args.limit_image_check] if args.limit_image_check else image_paths
    missing_local_images = [
        path for path in image_check_paths if path and not resolve_image(path, args.repo_root).is_file()
    ]

    image_to_token = build_image_to_token(args.navsim_log_path)
    missing_log_images = sorted(set(image_keys) - set(image_to_token))
    matched_tokens = [image_to_token[key] for key in image_keys if key in image_to_token]
    duplicate_tokens = len(matched_tokens) - len(set(matched_tokens))

    metric_cache_tokens = set(MetricCacheLoader(args.metric_cache_path, require_existing=True).tokens)
    missing_cache_tokens = sorted(set(matched_tokens) - metric_cache_tokens)
    cache_tokens_not_in_test = sorted(metric_cache_tokens - set(matched_tokens))

    report: dict[str, Any] = {
        "test_set_path": str(args.test_set_path),
        "repo_root": str(args.repo_root),
        "navsim_log_path": str(args.navsim_log_path),
        "metric_cache_path": str(args.metric_cache_path),
        "rows": len(rows),
        "image_keys": len(image_keys),
        "unique_image_keys": len(set(image_keys)),
        "missing_image_field": missing_image_field,
        "duplicate_image_keys": duplicate_image_keys,
        "image_check_count": len(image_check_paths),
        "missing_local_images": len(missing_local_images),
        "missing_local_images_preview": missing_local_images[:10],
        "navsim_log_image_keys": len(image_to_token),
        "missing_log_images": len(missing_log_images),
        "missing_log_images_preview": missing_log_images[:10],
        "matched_tokens": len(matched_tokens),
        "unique_matched_tokens": len(set(matched_tokens)),
        "duplicate_tokens": duplicate_tokens,
        "metric_cache_tokens": len(metric_cache_tokens),
        "missing_cache_tokens": len(missing_cache_tokens),
        "missing_cache_tokens_preview": missing_cache_tokens[:10],
        "cache_tokens_not_in_test": len(cache_tokens_not_in_test),
        "cache_tokens_not_in_test_preview": cache_tokens_not_in_test[:10],
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = (
        missing_image_field
        or duplicate_image_keys
        or missing_local_images
        or missing_log_images
        or duplicate_tokens
        or missing_cache_tokens
        or (cache_tokens_not_in_test and not args.allow_missing_cache_tokens)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
