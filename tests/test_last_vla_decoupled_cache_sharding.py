from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.merge_last_vla_overlay_shards import merge_shards


def _write_shard(root: Path, shard_index: int, tokens: list[str], *, overlay_type: str = "jepa128") -> Path:
    shard = root / "shards" / f"shard_{shard_index:05d}"
    samples = shard / "samples"
    samples.mkdir(parents=True)
    with (shard / "index.jsonl").open("w", encoding="utf-8") as f:
        for token in tokens:
            sample_path = samples / f"{token}.pt"
            sample_path.write_bytes(f"payload:{token}".encode("utf-8"))
            f.write(json.dumps({"sample_token": token, "scene_token": f"scene_{token}", "path": f"samples/{token}.pt"}) + "\n")
    (shard / "metadata.json").write_text(
        json.dumps(
            {
                "overlay_type": overlay_type,
                "shard_index": shard_index,
                "num_shards": 2,
                "num_written": len(tokens),
                "num_skipped_by_shard": 0,
                "sample_token_minimal_list_or_hash": {"count": len(tokens)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return shard


def _args(root: Path, output: Path, *, expected: int = 2, overlay_type: str = "jepa128") -> argparse.Namespace:
    return argparse.Namespace(
        sharded_root=root,
        output_root=output,
        expected_num_shards=expected,
        overlay_type=overlay_type,
        copy_mode="hardlink",
        strict=True,
    )


def test_merge_overlay_shards_succeeds_with_non_overlapping_tokens(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_shard(raw, 0, ["a", "b"])
    _write_shard(raw, 1, ["c"])

    summary = merge_shards(_args(raw, tmp_path / "merged"))

    assert summary["found_num_shards"] == 2
    assert summary["total_records"] == 3
    assert summary["duplicate_count"] == 0
    index_lines = (tmp_path / "merged" / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(index_lines) == 3
    assert (tmp_path / "merged" / "samples" / "a.pt").is_file()
    metadata = json.loads((tmp_path / "merged" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["overlay_type"] == "jepa128"


def test_merge_overlay_shards_rejects_duplicate_tokens(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_shard(raw, 0, ["dup"])
    _write_shard(raw, 1, ["dup"])

    with pytest.raises(RuntimeError, match="duplicate_sample_tokens"):
        merge_shards(_args(raw, tmp_path / "merged"))


def test_merge_overlay_shards_rejects_missing_shard_in_strict_mode(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_shard(raw, 0, ["a"])

    with pytest.raises(RuntimeError, match="missing_shards"):
        merge_shards(_args(raw, tmp_path / "merged"))


def test_prepare_decoupled_cache_dryrun_uses_shard_subdirs(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_CACHE": "0",
            "BASE_CHUNK_ROOT": str(tmp_path / "base"),
            "OUTPUT_ROOT": str(tmp_path / "out"),
            "VGGT_MODEL_PATH": str(tmp_path / "vggt"),
            "VJEPA_MODEL_PATH": str(tmp_path / "vjepa"),
            "NUM_SHARDS": "2",
            "SHARD_INDEX": "0",
            "PYTHON_BIN": "python",
        }
    )

    subprocess.run(
        ["bash", "scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh"],
        env=env,
        check=True,
    )

    commands = (
        tmp_path / "out" / "decoupled_highcap_no_risk" / "commands.log"
    ).read_text(encoding="utf-8")
    assert "train_jepa128_overlay_raw/shards/shard_00000" in commands
    assert "train_geometry192_overlay_raw/shards/shard_00000" in commands
    assert "--split navtrain" in commands
    assert "--data-root /mnt/navsim" in commands
    assert "--output-shard-subdir" in commands
    assert "merge_last_vla_geometry_cache_into_chunks.py" not in commands
    assert "train_jepa128_overlay/index.jsonl" not in commands


def test_prepare_decoupled_cache_dryrun_allows_explicit_train_split_and_data_root(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_CACHE": "0",
            "BASE_CHUNK_ROOT": str(tmp_path / "base"),
            "OUTPUT_ROOT": str(tmp_path / "out"),
            "VGGT_MODEL_PATH": str(tmp_path / "vggt"),
            "VJEPA_MODEL_PATH": str(tmp_path / "vjepa"),
            "NAVSIM_DATA_ROOT": str(tmp_path / "navsim_data"),
            "TRAIN_SPLIT": "navtest",
            "LOADER_MAX_SCENES": "3",
            "NUM_SHARDS": "2",
            "SHARD_INDEX": "0",
            "PYTHON_BIN": "python",
        }
    )

    subprocess.run(
        ["bash", "scripts/last_vla_v2/decoupled_highcap_no_risk/prepare_decoupled_highcap_no_risk_data.sh"],
        env=env,
        check=True,
    )

    commands = (
        tmp_path / "out" / "decoupled_highcap_no_risk" / "commands.log"
    ).read_text(encoding="utf-8")
    assert "--split navtest" in commands
    assert f"--data-root {tmp_path / 'navsim_data'}" in commands
    assert "--loader-max-scenes 3" in commands
