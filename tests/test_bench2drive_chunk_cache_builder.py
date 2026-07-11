from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from scripts.build_bench2drive_recogdrive_chunk_cache import build_cache, clip_dirs, parse_args
from navsim.agents.recogdrive.expert_cache import load_sample


def _write_anno(path: Path, idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": 10.0 + idx * 0.1,
        "y": 20.0,
        "theta": 0.0,
        "speed": 1.0,
        "command_near": 4,
        "command_far": 4,
        "next_command": 4,
        "acceleration": [0.0, 0.0, 9.8],
        "sensors": {
            "LIDAR_TOP": {
                "world2lidar": [
                    [1.0, 0.0, 0.0, -idx * 0.1],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def test_build_bench2drive_chunk_cache(tmp_path: Path):
    data_root = tmp_path / "Bench2Drive-Base"
    clip = data_root / "Straight_Town01_Route1_Weather0"
    for idx in range(60):
        _write_anno(clip / "anno" / f"{idx:05d}.json.gz", idx)
    camera_names = (
        "rgb_front",
        "rgb_front_left",
        "rgb_front_right",
        "rgb_back_left",
        "rgb_back_right",
        "rgb_back",
    )
    for camera_name in camera_names:
        directory = clip / "camera" / camera_name
        directory.mkdir(parents=True)
        for idx in (15, 16):
            (directory / f"{idx:05d}.jpg").write_bytes(b"fake")

    args = parse_args(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "cache"),
            "--max-samples",
            "2",
            "--cache-hidden-dtype",
            "float32",
        ]
    )

    class FakeVLMFeatureBuilder:
        def compute_bench2drive_multiview_features(self, **kwargs):
            assert len(kwargs["image_paths"]) == 6
            return {
                "last_hidden_state": torch.arange(24, dtype=torch.float32).reshape(3, 8),
                "image_patch_counts": torch.tensor([3, 3, 3, 3, 3, 3]),
            }

    metadata = build_cache(args, vlm_feature_builder=FakeVLMFeatureBuilder())
    assert metadata["num_records"] == 2

    lines = (tmp_path / "cache" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    sample = load_sample(tmp_path / "cache" / record["path"])
    assert tuple(sample["history_trajectory"].shape) == (4, 3)
    assert tuple(sample["trajectory"].shape) == (6, 3)
    assert tuple(sample["high_command_one_hot"].shape) == (3,)
    assert tuple(sample["status_feature"].shape) == (8,)
    assert tuple(sample["last_hidden_state"].shape) == (3, 8)
    assert torch.isfinite(sample["last_hidden_state"].float()).all()


def test_clip_list_filters_before_shard_slice(tmp_path: Path):
    data_root = tmp_path / "Bench2Drive-Base"
    for name in ("clip_a", "clip_b", "clip_c"):
        (data_root / name / "anno").mkdir(parents=True)
    clip_list = tmp_path / "clips.txt"
    clip_list.write_text("# held-out split\nclip_c\nclip_a\n", encoding="utf-8")

    selected = clip_dirs(data_root, None, None, 1, None, clip_list=clip_list)
    assert [path.name for path in selected] == ["clip_c"]
