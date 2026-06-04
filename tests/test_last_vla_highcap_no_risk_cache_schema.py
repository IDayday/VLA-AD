from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.expert_cache import atomic_torch_save


def _write_sample(root: Path, token: str) -> None:
    payload = {
        "sample_token": token,
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([1.0, 0.0, 0.0]),
        "last_hidden_state": torch.zeros(5, 1536),
        "status_feature": torch.zeros(8),
        "trajectory": torch.zeros(8, 3),
        "jepa_context_tokens": torch.zeros(128, 1024),
        "jepa_target_tokens": torch.zeros(128, 1024),
        "vggt_geometry_tokens": torch.zeros(192, 512),
        "vggt_geometry_mode_code": torch.tensor(2),
    }
    atomic_torch_save(payload, root / "samples" / f"{token}.pt")


def test_highcap_no_risk_dataset_does_not_require_risk_or_vggt_context(tmp_path: Path):
    try:
        from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset, custom_collate_fn
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    (tmp_path / "samples").mkdir(parents=True)
    _write_sample(tmp_path, "sample_000")
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"sample_token": "sample_000", "path": "samples/sample_000.pt"}) + "\n",
        encoding="utf-8",
    )

    dataset = ChunkCacheDataset(
        str(tmp_path),
        include_expert_features=True,
        include_expert_targets=True,
        use_jepa=True,
        use_vggt=True,
        num_jepa_tokens=128,
        num_vggt_tokens=128,
        num_geometry_tokens=192,
        use_last_vla=True,
        last_vla_stage="progressive_sft_bottleneck",
        last_vla_require_full_geometry=True,
        last_vla_allow_patch_geometry_fallback=False,
        last_vla_geometry_teacher_dim=512,
        last_vla_geometry_loss_weight=0.2,
    )
    features, targets, token = dataset[0]
    assert token == "sample_000"
    assert "risk_labels" not in features
    assert "vggt_context_tokens" not in features
    assert features["jepa_context_tokens"].shape == (128, 1024)
    assert features["jepa_target_tokens"].shape == (128, 1024)
    assert features["vggt_geometry_tokens"].shape == (192, 512)
    assert features["vggt_geometry_mode_code"].item() == 2
    collated_features, collated_targets, tokens = custom_collate_fn([(features, targets, token)])
    assert collated_features["vggt_geometry_tokens"].shape == (1, 192, 512)
    assert collated_features["jepa_target_tokens"].shape == (1, 128, 1024)
    assert collated_targets["trajectory"].shape == (1, 8, 3)
    assert tokens == ["sample_000"]
