from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from scripts.bench2drive.validate_recogdrive_b2d_stage2_cache import CacheValidationError, validate_cache


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_shard(root: Path, name: str, start: int, stop: int, vlm_path: Path) -> None:
    shard = root / name
    samples = shard / "samples"
    samples.mkdir(parents=True)
    sample = {
        "sample_token": f"sample_{start}",
        "last_hidden_state": torch.zeros(2, 1536),
        "history_trajectory": torch.zeros(4, 3),
        "trajectory": torch.zeros(8, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "status_feature": torch.zeros(8),
    }
    sample_path = samples / f"sample_{start}.pt"
    torch.save(sample, sample_path)
    row = {"path": str(sample_path.relative_to(shard)), "sample_token": sample["sample_token"]}
    (shard / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    metadata = {
        "is_dummy": False,
        "contains_vlm_hidden": True,
        "hidden_source": "recogdrive_vlm",
        "system_prompt_profile": "bench2drive",
        "recogdrive_vlm_path": str(vlm_path),
        "num_records": 1,
        "clip_start": start,
        "clip_stop": stop,
        "num_selected_clips": stop - start,
        "num_skipped_windows": 0,
        "clip_list": str(root / "all_clips.txt"),
    }
    (shard / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_validate_complete_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    vlm_path = tmp_path / "stage1"
    vlm_path.mkdir()
    _write_shard(cache, "shard_00", 0, 2, vlm_path)
    _write_shard(cache, "shard_01", 2, 4, vlm_path)

    summary = validate_cache(
        cache,
        expected_shards=2,
        expected_clips=4,
        expected_records=2,
        expected_vlm_path=vlm_path,
    )

    assert summary["ready"] is True
    assert summary["num_shards"] == 2
    assert summary["num_selected_clips"] == 4
    assert summary["num_records"] == 2
    assert len(summary["sample_checks"]) == 2


def test_validate_rejects_incomplete_shard(tmp_path: Path):
    cache = tmp_path / "cache"
    vlm_path = tmp_path / "stage1"
    vlm_path.mkdir()
    _write_shard(cache, "shard_00", 0, 2, vlm_path)
    (cache / "shard_00" / "metadata.json").unlink()

    with pytest.raises(CacheValidationError, match="missing"):
        validate_cache(
            cache,
            expected_shards=1,
            expected_clips=2,
            expected_records=1,
            expected_vlm_path=vlm_path,
        )


def test_validate_rejects_clip_gap(tmp_path: Path):
    cache = tmp_path / "cache"
    vlm_path = tmp_path / "stage1"
    vlm_path.mkdir()
    _write_shard(cache, "shard_00", 0, 2, vlm_path)
    _write_shard(cache, "shard_01", 3, 4, vlm_path)

    with pytest.raises(CacheValidationError, match="gap or overlap"):
        validate_cache(
            cache,
            expected_shards=2,
            expected_clips=4,
            expected_records=2,
            expected_vlm_path=vlm_path,
        )


def test_stage2_launcher_ignores_matching_log_files(tmp_path: Path):
    cache = tmp_path / "cache"
    shard = cache / "shard_00"
    shard.mkdir(parents=True)
    (cache / "shard_00.log").write_text("worker log\n", encoding="utf-8")
    source_vlm = tmp_path / "stage1"
    source_vlm.mkdir()
    metadata = {
        "is_dummy": False,
        "contains_vlm_hidden": True,
        "hidden_source": "recogdrive_vlm",
        "system_prompt_profile": "bench2drive",
        "recogdrive_vlm_path": str(source_vlm),
        "num_records": 1,
    }
    (shard / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (shard / "index.jsonl").write_text('{"path":"samples/example.pt"}\n', encoding="utf-8")

    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "VLA_AD_ROOT": str(REPO_ROOT),
            "CONDA_BIN": "/bin/true",
            "CHUNK_CACHE_ROOT": str(cache),
            "EXPECTED_VLM_PATH": str(source_vlm),
            "BASE_VLM_PATH": str(tmp_path / "base"),
            "OUTPUT_DIR": str(output),
            "RANDOM_INIT_POLICY": "1",
            "GPU_LIST": "0",
            "NPROC_PER_NODE": "1",
            "GLOBAL_EPOCHS": "1",
            "ALLOW_RETIRED_CUSTOM_B2D_PIPELINE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "cache_summary.json").read_text(encoding="utf-8"))
    assert summary["total_records"] == 1
    assert [row["name"] for row in summary["shards"]] == ["shard_00"]
