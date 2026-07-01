from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from navsim.agents.recogdrive.expert_cache import atomic_torch_save
from navsim.agents.recogdrive.pareto_support import DESCRIPTOR_NAMES


def _write_sample(root: Path, token: str, *, log_name: str, trajectory_value: float) -> None:
    payload = {
        "sample_token": token,
        "history_trajectory": torch.zeros(4, 3),
        "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
        "last_hidden_state": torch.zeros(5, 1536),
        "status_feature": torch.zeros(8),
        "trajectory": torch.full((8, 3), trajectory_value),
    }
    atomic_torch_save(payload, root / "samples" / f"{token}.pt")
    with (root / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_token": token, "log_name": log_name, "path": f"samples/{token}.pt"}) + "\n")


def _write_support_index(path: Path) -> torch.Tensor:
    support = torch.zeros(1, 3, 8, 3)
    support[0, 0] = torch.full((8, 3), 7.0)
    support_mask = torch.tensor([[True, False, False]])
    payload = {
        "version": 1,
        "tokens": ["train-token"],
        "token_to_row": {"train-token": 0},
        "support_trajectories": support,
        "support_mask": support_mask,
        "support_weights": torch.tensor([[1.0, 0.0, 0.0]]),
        "support_scores": torch.tensor([[0.9, 0.0, 0.0]]),
        "descriptor_mean": torch.zeros(len(DESCRIPTOR_NAMES)),
        "descriptor_std": torch.ones(len(DESCRIPTOR_NAMES)),
        "summary": {"unit_test": True},
    }
    torch.save(payload, path)
    return support[0, 0]


def test_cache_train_all_records_switch_clears_train_log_filter():
    try:
        from navsim.planning.script.run_training_recogdrive import (
            _cache_train_all_records_enabled,
            _cache_train_log_names,
        )
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    split_cfg = OmegaConf.create({"cache_train_all_records": False, "train_logs": ["train-a", "train-b"]})
    assert _cache_train_all_records_enabled(split_cfg) is False
    assert _cache_train_log_names(split_cfg) == ["train-a", "train-b"]

    full_cfg = OmegaConf.create({"cache_train_all_records": True, "train_logs": ["train-a", "train-b"]})
    assert _cache_train_all_records_enabled(full_cfg) is True
    assert _cache_train_log_names(full_cfg) is None

    legacy_full_cfg = OmegaConf.create({"train_all_cache_records": True, "train_logs": ["train-a"]})
    assert _cache_train_all_records_enabled(legacy_full_cfg) is True
    assert _cache_train_log_names(legacy_full_cfg) is None


def test_cache_only_dataset_full_scan_ignores_root_metadata_files(tmp_path: Path):
    try:
        from navsim.planning.training.dataset import CacheOnlyDataset
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    cache_root = tmp_path / "legacy-cache"
    (cache_root / "log-a" / "token-a").mkdir(parents=True)
    (cache_root / "log-b" / "token-b").mkdir(parents=True)
    (cache_root / ".manifest.json").write_text("{}", encoding="utf-8")

    dataset = CacheOnlyDataset(
        cache_path=str(cache_root),
        feature_builders=[],
        target_builders=[],
        log_names=None,
    )

    assert sorted(dataset.tokens) == ["token-a", "token-b"]


def test_chunk_cache_dataset_returns_pareto_support_without_replacing_gt(tmp_path: Path):
    try:
        from navsim.planning.script.run_training_recogdrive import ChunkCacheDataset, custom_collate_fn
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    cache_root = tmp_path / "cache"
    (cache_root / "samples").mkdir(parents=True)
    _write_sample(cache_root, "train-token", log_name="train-log", trajectory_value=2.0)
    support_target = _write_support_index(tmp_path / "support.pt")

    dataset = ChunkCacheDataset(
        str(cache_root),
        log_names=None,
        split_name="train",
        stage2_target_source="pareto_support",
        stage2_pareto_support_index_path=str(tmp_path / "support.pt"),
    )

    features, targets, token = dataset[0]
    assert token == "train-token"
    assert torch.allclose(targets["trajectory"], torch.full((8, 3), 2.0))
    assert torch.allclose(targets["support_trajectories"][0], support_target)
    assert targets["support_mask"].tolist() == [True, False, False]
    assert targets["support_weights"].tolist() == [1.0, 0.0, 0.0]
    assert targets["support_missing_mask"].item() is False

    batch_features, batch_targets, batch_tokens = custom_collate_fn([(features, targets, token)])
    assert batch_tokens == ["train-token"]
    assert batch_targets["trajectory"].shape == (1, 8, 3)
    assert batch_targets["support_trajectories"].shape == (1, 3, 8, 3)
    assert batch_targets["support_mask"].shape == (1, 3)
    assert batch_features["last_hidden_state"].shape == (1, 5, 1536)


def test_stage2_pareto_support_wrapper_injects_cache_only_targets(tmp_path: Path):
    try:
        from navsim.planning.script.run_training_recogdrive import Stage2ParetoSupportDataset, custom_collate_fn
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    class TinyCacheOnly(torch.utils.data.Dataset):
        tokens = ["train-token"]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            features = {
                "history_trajectory": torch.zeros(4, 3),
                "high_command_one_hot": torch.tensor([0.0, 1.0, 0.0]),
                "last_hidden_state": torch.zeros(5, 1536),
                "status_feature": torch.zeros(8),
            }
            targets = {"trajectory": torch.full((8, 3), 2.0)}
            return features, targets, self.tokens[idx]

    support_target = _write_support_index(tmp_path / "support.pt")
    dataset = Stage2ParetoSupportDataset(
        TinyCacheOnly(),
        support_index_path=str(tmp_path / "support.pt"),
    )

    features, targets, token = dataset[0]
    assert token == "train-token"
    assert torch.allclose(targets["trajectory"], torch.full((8, 3), 2.0))
    assert torch.allclose(targets["support_trajectories"][0], support_target)

    _, batch_targets, tokens = custom_collate_fn([(features, targets, token)])
    assert tokens == ["train-token"]
    assert batch_targets["support_trajectories"].shape == (1, 3, 8, 3)
