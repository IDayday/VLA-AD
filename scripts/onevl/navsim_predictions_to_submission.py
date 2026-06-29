#!/usr/bin/env python3
"""Convert OneVL NAVSIM prediction JSON into NAVSIM official submission pickle."""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig, Trajectory


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


def response_to_traj(response: Any) -> list[list[float]] | None:
    if isinstance(response, list):
        if response and isinstance(response[0], (list, tuple)):
            return [[float(v) for v in point[:3]] for point in response]
        if response and isinstance(response[0], str):
            response = response[0]
    if not isinstance(response, str):
        return None

    text = response.strip().replace("\n", " ")
    if text.startswith(">["):
        text = text[1:]
    for tag in [
        "<answer>",
        "</answer>",
        "<|im_end|>",
        "<|start-latent|>",
        "<|latent|>",
        "<|end-latent|>",
        "<|start-latent-vis|>",
        "<|latent-vis|>",
        "<|end-latent-vis|>",
    ]:
        text = text.replace(tag, "")
    text = text.strip()
    for candidate in (f"[{text}]", f"[[{text}]"):
        try:
            parsed = json.loads(candidate)
            return [[float(v) for v in point[:3]] for point in parsed]
        except Exception:
            pass
    try:
        parsed = ast.literal_eval(f"[{text}]")
        return [[float(v) for v in point[:3]] for point in parsed]
    except Exception:
        return None


def load_prediction_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported prediction JSON shape in {path}")
    return data


def extract_image(item: dict[str, Any]) -> str:
    messages = item.get("messages", [])
    if not messages:
        raise ValueError("Prediction item has no messages")
    content = messages[0].get("content")
    if isinstance(content, list):
        images = [part.get("image") for part in content if isinstance(part, dict) and part.get("image")]
        if not images:
            raise ValueError("Prediction item content has no image entries")
        return images[-1]
    images = item.get("images")
    if images:
        return images[-1]
    raise ValueError("Prediction item has no usable image path")


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
    image_to_token: dict[str, str] = {}
    current_index = scene_filter.num_history_frames - 1
    for token, frames in scene_loader.scene_frames_dicts.items():
        cam_path = frames[current_index]["cams"]["CAM_F0"]["data_path"]
        image_to_token[normalize_image_path(cam_path)] = token
    return image_to_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--navsim-log-path", default="/mnt/project/onevl_navsim_data/navsim_logs/test")
    parser.add_argument("--metric-cache-path", default="/mnt/project/onevl_navsim_exp/metric_cache")
    parser.add_argument("--allow-missing-cache-tokens", action="store_true")
    parser.add_argument(
        "--invalid-trajectory-policy",
        choices=["error", "zero", "gt"],
        default="error",
        help="How to handle unparseable model trajectories.",
    )
    args = parser.parse_args()

    image_to_token = build_image_to_token(Path(args.navsim_log_path))
    predictions: dict[str, Trajectory] = {}
    bad_items: list[tuple[int, str]] = []
    for index, item in enumerate(load_prediction_items(Path(args.input_json))):
        try:
            image_key = normalize_image_path(extract_image(item))
            token = image_to_token[image_key]
            traj = item.get("pre_traj")
            if traj is None:
                traj = response_to_traj(item.get("output_text"))
            else:
                traj = response_to_traj(traj)
            if traj is None:
                if args.invalid_trajectory_policy == "zero":
                    traj = [[0.0, 0.0, 0.0] for _ in range(8)]
                elif args.invalid_trajectory_policy == "gt":
                    traj = response_to_traj(item.get("GT") or item.get("gt_traj"))
                    if traj is None:
                        raise ValueError("could not parse trajectory or GT fallback")
                else:
                    raise ValueError("could not parse trajectory")
            poses = np.asarray(traj, dtype=np.float32)
            predictions[token] = Trajectory(poses)
        except Exception as exc:
            bad_items.append((index, str(exc)))

    if bad_items:
        preview = "; ".join(f"{idx}: {err}" for idx, err in bad_items[:10])
        raise SystemExit(f"Failed to convert {len(bad_items)} prediction items. First errors: {preview}")

    cache_tokens = set(MetricCacheLoader(Path(args.metric_cache_path), require_existing=True).tokens)
    missing_cache_tokens = sorted(cache_tokens - set(predictions))
    extra_tokens = sorted(set(predictions) - cache_tokens)
    if missing_cache_tokens and not args.allow_missing_cache_tokens:
        raise SystemExit(
            f"Predictions are missing {len(missing_cache_tokens)} metric-cache tokens. "
            f"First missing: {missing_cache_tokens[:10]}"
        )
    if extra_tokens:
        print(f"[WARN] Dropping {len(extra_tokens)} predictions not present in metric cache")
        for token in extra_tokens:
            predictions.pop(token, None)

    output_path = Path(args.output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"predictions": [predictions]}, f)
    print(f"Wrote {output_path} with {len(predictions)} predictions")


if __name__ == "__main__":
    main()
