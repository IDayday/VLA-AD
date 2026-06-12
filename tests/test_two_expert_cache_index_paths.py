from __future__ import annotations

import json

import torch

from navsim.agents.recogdrive.expert_cache import load_sample
from scripts.last_vla_v2.two_expert_slot.build_jepa_dynamic_teacher_cache import merge_shards as merge_jepa_shards
from scripts.last_vla_v2.two_expert_slot.preflight_two_expert_teacher_cache import preflight
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import (
    iter_indexed_records,
    load_path_index,
    resolve_index_record_path,
)


def _write_indexed_sample(root, token: str, payload: dict, *, absolute: bool = False) -> None:
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"{token}.pt"
    torch.save({"sample_token": token, **payload}, sample_path)
    path = str(sample_path if absolute else sample_path.relative_to(root))
    (root / "index.jsonl").write_text(json.dumps({"sample_token": token, "path": path}) + "\n", encoding="utf-8")


def test_resolve_index_record_path_relative_and_absolute(tmp_path):
    root = tmp_path / "cache"
    _write_indexed_sample(root, "a", {"x": torch.tensor(1)}, absolute=False)
    record = {"path": "samples/a.pt"}
    assert resolve_index_record_path(root, record) == root / "samples" / "a.pt"

    absolute = root / "samples" / "a.pt"
    assert resolve_index_record_path(root, {"path": str(absolute)}) == absolute
    assert load_path_index(root)["a"] == root / "samples" / "a.pt"


def test_preflight_loads_relative_and_absolute_index_paths(tmp_path):
    jepa = tmp_path / "jepa"
    vggt = tmp_path / "vggt"
    _write_indexed_sample(
        jepa,
        "a",
        {
            "jepa_dynamic_teacher_tokens": torch.randn(3, 12, 1024),
            "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": True},
        },
        absolute=False,
    )
    _write_indexed_sample(
        vggt,
        "a",
        {
            "vggt_feature23_tokens": torch.randn(12, 768),
            "vggt_feature23_metadata": {"strict_geometry_teacher": True, "feature_dim": 768},
        },
        absolute=True,
    )

    report = preflight(jepa, vggt, strict=True)

    assert report["ok"] is True
    assert report["teacher_status"] == "strict_production_teacher"
    assert report["vggt_feature_dims"] == [768]


def test_shard_merge_preserves_loadable_relative_paths(tmp_path):
    root = tmp_path / "jepa_merged"
    shard = root / "shards" / "shard_00000"
    _write_indexed_sample(
        shard,
        "a",
        {
            "jepa_dynamic_teacher_tokens": torch.randn(3, 12, 1024),
            "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": True},
        },
        absolute=False,
    )
    (shard / "metadata.json").write_text(
        json.dumps({"num_legacy_fallback": 0, "num_strict_teacher": 1}) + "\n",
        encoding="utf-8",
    )

    metadata = merge_jepa_shards(root, overwrite=True)
    records = list(iter_indexed_records(root))
    payload = load_sample(records[0][1])

    assert metadata["route_status"] == "production_teacher_ok"
    assert records[0][1].is_file()
    assert payload["sample_token"] == "a"
