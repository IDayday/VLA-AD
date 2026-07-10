from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from scripts.build_bench2drive_recogdrive_chunk_cache import build_cache, parse_args
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
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def test_build_bench2drive_chunk_cache(tmp_path: Path):
    data_root = tmp_path / "Bench2Drive-Base"
    clip = data_root / "Straight_Town01_Route1_Weather0"
    for idx in range(60):
        _write_anno(clip / "anno" / f"{idx:05d}.json.gz", idx)
    (clip / "camera" / "rgb_front").mkdir(parents=True)
    (clip / "camera" / "rgb_front" / "00015.jpg").write_bytes(b"fake")

    args = parse_args(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "cache"),
            "--max-samples",
            "2",
            "--hidden-tokens",
            "3",
            "--vlm-feature-dim",
            "8",
            "--hidden-dtype",
            "float32",
            "--sample-stride",
            "1",
        ]
    )
    metadata = build_cache(args)
    assert metadata["num_records"] == 2

    lines = (tmp_path / "cache" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    sample = load_sample(tmp_path / "cache" / record["path"])
    assert tuple(sample["history_trajectory"].shape) == (4, 3)
    assert tuple(sample["trajectory"].shape) == (8, 3)
    assert tuple(sample["high_command_one_hot"].shape) == (3,)
    assert tuple(sample["status_feature"].shape) == (8,)
    assert tuple(sample["last_hidden_state"].shape) == (3, 8)
    assert torch.isfinite(sample["last_hidden_state"].float()).all()
