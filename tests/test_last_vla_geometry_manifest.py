from __future__ import annotations

import json
from pathlib import Path

import torch

from navsim.agents.recogdrive.expert_cache import atomic_torch_save
from scripts.audit_last_vla_cache_manifest import audit_cache


def _write_sample(root: Path, token: str, mode_code: int) -> None:
    sample_path = root / "samples" / f"{token}.pt"
    payload = {
        "sample_token": token,
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "last_hidden_state": torch.zeros(2, 1536),
        "status_feature": torch.zeros(8),
        "trajectory": torch.zeros(8, 3),
        "jepa_context_tokens": torch.zeros(12, 1024),
        "jepa_target_tokens": torch.zeros(12, 1024),
        "vggt_context_tokens": torch.zeros(12, 2048),
        "vggt_target_tokens": torch.zeros(12, 2048),
        "vggt_geometry_tokens": torch.zeros(12, 512),
        "vggt_geometry_mode_code": torch.tensor(mode_code),
    }
    atomic_torch_save(payload, sample_path)
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_token": token, "path": f"samples/{token}.pt"}) + "\n")


def test_last_vla_manifest_reports_geometry_modes_and_shapes(tmp_path: Path):
    root = tmp_path / "cache"
    (root / "samples").mkdir(parents=True)
    _write_sample(root, "a", 2)
    _write_sample(root, "b", 1)

    report = audit_cache(root, chunk_name_pattern=None, max_samples=None)

    assert report["base_field_coverage"]["last_hidden_state"]["coverage"] == 1.0
    assert report["teacher_token_coverage"]["jepa_target_tokens"]["coverage"] == 1.0
    assert report["vggt_geometry_mode_distribution"]["full_geometry"] == 1
    assert report["vggt_geometry_mode_distribution"]["patch_fallback"] == 1
    assert report["geometry_token_shape_distribution"][str((12, 512))] == 2
