from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs

_install_dependency_stubs()

from navsim.agents.recogdrive.expert_cache import atomic_torch_save


def _write_sample(root: Path, token: str, *, log_name: str, trajectory_value: float) -> None:
    payload = {
        "sample_token": token,
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "last_hidden_state": torch.zeros(5, 1536),
        "status_feature": torch.zeros(8),
        "trajectory": torch.full((8, 3), trajectory_value),
        "two_expert_h_dyn": torch.zeros(3, 12, 1536),
        "two_expert_h_geo": torch.zeros(12, 1536),
    }
    atomic_torch_save(payload, root / "samples" / f"{token}.pt")
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"sample_token": token, "log_name": log_name, "path": f"samples/{token}.pt"}) + "\n"
        )


def test_chunk_cache_dataset_can_train_all_records_with_elite_target_fallback(tmp_path: Path):
    try:
        from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    (tmp_path / "cache" / "samples").mkdir(parents=True)
    _write_sample(tmp_path / "cache", "train-token", log_name="train-log", trajectory_value=0.0)
    _write_sample(tmp_path / "cache", "val-token", log_name="val-log", trajectory_value=2.0)

    elite_target = torch.full((8, 3), 7.0)
    index_path = tmp_path / "elite_targets.pt"
    torch.save(
        {
            "version": 1,
            "tokens": ["train-token"],
            "trajectories": elite_target.unsqueeze(0),
            "summary": {"selected_target_count": 1},
        },
        index_path,
    )

    dataset = ChunkCacheDataset(
        str(tmp_path / "cache"),
        log_names=None,
        split_name="train_all_cache",
        use_two_expert_slots=True,
        stage2_target_source="awac_elite_best_valid_above_gt_or_gt",
        stage2_elite_target_index_path=str(index_path),
    )

    assert len(dataset) == 2
    _, train_targets, train_token = dataset[0]
    _, val_targets, val_token = dataset[1]
    assert train_token == "train-token"
    assert val_token == "val-token"
    assert torch.allclose(train_targets["trajectory"], elite_target)
    assert torch.allclose(val_targets["trajectory"], torch.full((8, 3), 2.0))
