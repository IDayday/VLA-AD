from __future__ import annotations

import json

import torch

from scripts.last_vla_v2.two_expert_slot.final_two_expert_readiness_gate import validate_coverage
from scripts.last_vla_v2.two_expert_slot.preflight_two_expert_coverage import coverage_report


def _write_index(root, tokens, *, shard: bool = False):
    base = root / "shards" / "shard_00000" if shard else root
    sample_dir = base / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for token in tokens:
        sample_path = sample_dir / f"{token}.pt"
        torch.save({"sample_token": token}, sample_path)
        rows.append({"sample_token": token, "path": str(sample_path.relative_to(base))})
    (base / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_two_expert_coverage_full_ok(tmp_path):
    base = _write_index(tmp_path / "base", ["a", "b", "c"])
    jepa = _write_index(tmp_path / "jepa", ["a", "b", "c"])
    vggt = _write_index(tmp_path / "vggt", ["a", "b", "c"])

    report = coverage_report(
        base_chunk_root=base,
        jepa_cache_root=jepa,
        vggt_cache_root=vggt,
        min_train_coverage=0.99,
    )

    assert report["ok"] is True
    assert report["base_count"] == 3
    assert report["all_teacher_coverage"] == 1.0


def test_two_expert_coverage_partial_fails(tmp_path):
    base = _write_index(tmp_path / "base", ["a", "b", "c"])
    jepa = _write_index(tmp_path / "jepa", ["a", "b"])
    vggt = _write_index(tmp_path / "vggt", ["a", "b", "c"])

    report = coverage_report(
        base_chunk_root=base,
        jepa_cache_root=jepa,
        vggt_cache_root=vggt,
        min_train_coverage=0.99,
    )

    assert report["ok"] is False
    assert report["intersection_count"] == 2
    assert report["all_teacher_coverage"] == 2 / 3


def test_two_expert_coverage_relative_shard_paths_work(tmp_path):
    base = _write_index(tmp_path / "base", ["a", "b"], shard=True)
    jepa = _write_index(tmp_path / "jepa", ["a", "b"], shard=True)
    vggt = _write_index(tmp_path / "vggt", ["a", "b"], shard=True)

    report = coverage_report(
        base_chunk_root=base,
        jepa_cache_root=jepa,
        vggt_cache_root=vggt,
        min_train_coverage=1.0,
    )

    assert report["ok"] is True
    assert report["all_teacher_coverage"] == 1.0


def test_readiness_gate_rejects_low_or_missing_coverage():
    assert validate_coverage({"ok": False, "all_teacher_coverage": 0.5}, 0.99)
    assert validate_coverage({"ok": True, "all_teacher_coverage": 0.5}, 0.99)
    assert validate_coverage({"ok": True, "all_teacher_coverage": 0.99}, 0.99) == []
