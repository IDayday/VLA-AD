from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from navsim.agents.recogdrive.expert_cache import atomic_torch_save, load_sample


def test_lora_hidden_cache_synthetic_preserves_teacher_keys(tmp_path: Path):
    base = tmp_path / "base"
    (base / "samples").mkdir(parents=True)
    sample = {
        "sample_token": "sample_000",
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "status_feature": torch.zeros(8),
        "trajectory": torch.zeros(8, 3),
        "last_hidden_state": torch.zeros(3, 1536),
        "jepa_context_tokens": torch.ones(12, 1024),
        "vggt_geometry_tokens": torch.ones(12, 512),
        "vggt_geometry_mode_code": torch.tensor(2),
    }
    atomic_torch_save(sample, base / "samples" / "sample_000.pt")
    (base / "index.jsonl").write_text(json.dumps({"sample_token": "sample_000", "path": "samples/sample_000.pt"}) + "\n", encoding="utf-8")
    lora = tmp_path / "lora.pt"
    vlm = tmp_path / "vlm"
    lora.write_bytes(b"dummy")
    vlm.write_text("dummy", encoding="utf-8")
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_recogdrive_hidden_cache_with_lora.py",
            "--base-chunk-root",
            str(base),
            "--output-chunk-root",
            str(out),
            "--vlm-path",
            str(vlm),
            "--vlm-lora-adapter",
            str(lora),
            "--synthetic-smoke",
        ],
        check=True,
    )
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    regenerated = load_sample(out / "samples" / "sample_000.pt")

    assert metadata["cache_hidden_state_regenerated"] is True
    assert metadata["hidden_cache_source"] == "vlm_lora_regenerated"
    assert torch.allclose(regenerated["jepa_context_tokens"], sample["jepa_context_tokens"])
    assert torch.allclose(regenerated["vggt_geometry_tokens"], sample["vggt_geometry_tokens"])
    assert regenerated["last_hidden_state"].shape == sample["last_hidden_state"].shape
