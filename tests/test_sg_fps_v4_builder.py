from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.training.build_recogdrive_stage3_awac_elite_buffer import (
    _candidate_root_fingerprint,
    _candidate_records_from_awac_row,
    _resolve_build_logs,
    _scene_seed,
)


def _awac_batch() -> dict:
    candidates = torch.arange(1 * 3 * 8 * 3, dtype=torch.float32).reshape(1, 3, 8, 3)
    components = {"pdms": torch.ones((1, 3), dtype=torch.float32)}
    return {
        "candidate_sources": ["gt", "progress_endpoint", "ddv2"],
        "candidate_trajs": candidates,
        "candidate_rewards": torch.tensor([[0.9, 0.95, 0.96]]),
        "candidate_selection_score": torch.tensor([[0.9, 0.94, 0.95]]),
        "candidate_components": components,
        "selected_real_mask": torch.tensor([[True]]),
        "selected_source_index": torch.tensor([[0]]),
        "selected_trajs": candidates[:, :1],
        "selected_rewards": torch.tensor([[0.9]]),
        "selected_selection_score": torch.tensor([[0.9]]),
        "selected_components": {"pdms": torch.ones((1, 1), dtype=torch.float32)},
    }


def test_v4_builder_uses_raw_candidates_before_awac_topk() -> None:
    awac = _awac_batch()

    legacy = _candidate_records_from_awac_row("scene", awac, 0)
    raw = _candidate_records_from_awac_row("scene", awac, 0, use_raw_candidates=True)

    assert [record.source for record in legacy] == ["gt"]
    assert [record.source for record in raw] == ["gt", "progress_endpoint", "ddv2"]


def test_v4_builder_train_val_domain_contains_both_log_sets() -> None:
    cfg = SimpleNamespace(train_logs=["train-a", "train-b"], val_logs=["val-a"])

    assert _resolve_build_logs(cfg, "train_val") == ["train-a", "train-b", "val-a"]
    assert _resolve_build_logs(cfg, "train") == ["train-a", "train-b"]
    assert _resolve_build_logs(cfg, "val") == ["val-a"]


def test_v4_builder_rejects_unknown_log_domain() -> None:
    cfg = SimpleNamespace(train_logs=[], val_logs=[])

    with pytest.raises(ValueError, match="BUILD_LOG_SPLIT"):
        _resolve_build_logs(cfg, "partial")


def test_v4_scene_seed_is_stable_and_token_specific() -> None:
    assert _scene_seed(0, "scene-a") == _scene_seed(0, "scene-a")
    assert _scene_seed(0, "scene-a") != _scene_seed(0, "scene-b")
    assert _scene_seed(0, "scene-a") != _scene_seed(1, "scene-a")


def test_external_candidate_root_fingerprint_tracks_payload_content(tmp_path) -> None:
    root = tmp_path / "candidates"
    root.mkdir()
    (root / "a.pkl").write_bytes(b"a")
    (root / "ignored.json").write_text("{}", encoding="utf-8")

    first, count = _candidate_root_fingerprint(str(root))
    second, second_count = _candidate_root_fingerprint(str(root))
    (root / "a.pkl").write_bytes(b"changed")
    changed, changed_count = _candidate_root_fingerprint(str(root))

    assert first == second
    assert count == second_count == changed_count == 1
    assert changed != first
