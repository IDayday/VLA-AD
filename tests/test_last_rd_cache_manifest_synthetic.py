from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.audit_last_rd_cache_manifest import audit_cache


def _write_sample(root: Path, name: str, high_command_shape: int = 3) -> None:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_token": name,
        "scene_token": f"scene_{name}",
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.zeros(high_command_shape),
        "last_hidden_state": torch.zeros(5, 1536),
        "status_feature": torch.zeros(8),
        "trajectory": torch.zeros(8, 3),
        "jepa_context_tokens": torch.zeros(12, 1024),
        "jepa_target_tokens": torch.zeros(12, 1024),
        "vggt_context_tokens": torch.zeros(12, 2048),
        "vggt_target_tokens": torch.zeros(12, 2048),
        "vggt_geometry_tokens": torch.zeros(12, 2048),
        "vggt_geometry_target_tokens": torch.zeros(12, 2048),
        "vggt_geometry_mode": "patch_fallback",
    }
    torch.save(payload, samples / f"{name}.pt")


def _write_index(root: Path, names: list[str]) -> None:
    with (root / "index.jsonl").open("w", encoding="utf-8") as f:
        for name in names:
            f.write(json.dumps({"sample_token": name, "path": f"samples/{name}.pt"}) + "\n")


def test_last_rd_cache_manifest_synthetic(tmp_path):
    _write_sample(tmp_path, "sample_a", high_command_shape=3)
    _write_index(tmp_path, ["sample_a"])
    manifest = audit_cache(tmp_path, future_jepa_loss_weight=0.3, risk_loss_weight=0.0)
    assert manifest["num_samples_scanned"] == 1
    assert manifest["required_base_key_coverage"]["high_command_one_hot"]["coverage"] == 1.0
    assert manifest["high_command_one_hot_shape_distribution"] == {"(3,)": 1}
    assert manifest["teacher_token_coverage"]["jepa_target_tokens"]["coverage"] == 1.0
    assert manifest["vggt_geometry_mode_distribution"]["patch_fallback"] == 1


def test_chunk_cache_high_command_shape_four_fails_fast(tmp_path):
    try:
        from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    _write_sample(tmp_path, "sample_bad", high_command_shape=4)
    _write_index(tmp_path, ["sample_bad"])
    dataset = ChunkCacheDataset(str(tmp_path), include_expert_features=False)
    with pytest.raises(ValueError, match="left/straight/right"):
        _ = dataset[0]
