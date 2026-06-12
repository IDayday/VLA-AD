from __future__ import annotations

import torch

from scripts.audit_two_expert_teacher_cache import audit_records
from scripts.last_vla_v2.two_expert_slot.preflight_two_expert_teacher_cache import preflight
from scripts.last_vla_v2.two_expert_slot.build_jepa_dynamic_teacher_cache import (
    pack_legacy_jepa_tokens,
    resolve_dynamic_teacher,
)
from scripts.last_vla_v2.two_expert_slot.build_vggt_feature23_cache import (
    pack_to_12_tokens,
    resolve_feature23_teacher,
)


def test_jepa_dynamic_teacher_shape_and_fallback_metadata():
    legacy = torch.randn(128, 1024)
    packed = pack_legacy_jepa_tokens(legacy)
    tokens, metadata = resolve_dynamic_teacher({"sample_token": "a", "jepa_target_tokens": legacy}, strict_teacher=False)

    assert packed.shape == (3, 12, 1024)
    assert tokens.shape == (3, 12, 1024)
    assert metadata["teacher_source"] == "legacy_jepa_target_downsampled"
    assert metadata["strict_dynamic_teacher"] is False


def test_vggt_feature23_teacher_shape_and_strict_metadata():
    feature23 = torch.randn(12, 64)
    tokens, metadata = resolve_feature23_teacher({"vggt_feature23_tokens": feature23}, strict_teacher=True)
    fallback, fallback_metadata = resolve_feature23_teacher({"vggt_geometry_tokens": torch.randn(192, 32)}, strict_teacher=False)

    assert tokens.shape == (12, 64)
    assert metadata["layer_index"] == 23
    assert metadata["strict_geometry_teacher"] is True
    assert pack_to_12_tokens(torch.randn(192, 32)).shape == (12, 32)
    assert fallback.shape == (12, 32)
    assert fallback_metadata["strict_geometry_teacher"] is False


def test_teacher_audit_strict_flags():
    report = audit_records(
        [
            {
                "sample_token": "a",
                "jepa_dynamic_teacher_tokens": torch.randn(3, 12, 1024),
                "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": False},
            }
        ],
        expected_kind="jepa",
        require_strict=True,
    )

    assert report["total_records"] == 1
    assert report["ok"] is False
    assert "strict_teacher_required_but_fallback_record" in report["errors"][0]["errors"]


def test_teacher_preflight_rejects_fallback_in_strict_mode(tmp_path):
    jepa_root = tmp_path / "jepa"
    vggt_root = tmp_path / "vggt"
    for root in (jepa_root, vggt_root):
        (root / "samples").mkdir(parents=True)
    torch.save(
        {
            "sample_token": "a",
            "jepa_dynamic_teacher_tokens": torch.randn(3, 12, 1024),
            "jepa_dynamic_teacher_metadata": {"strict_dynamic_teacher": False},
        },
        jepa_root / "samples" / "a.pt",
    )
    torch.save(
        {
            "sample_token": "a",
            "vggt_feature23_tokens": torch.randn(12, 768),
            "vggt_feature23_metadata": {"strict_geometry_teacher": True, "feature_dim": 768},
        },
        vggt_root / "samples" / "a.pt",
    )
    (jepa_root / "index.jsonl").write_text('{"sample_token":"a","path":"samples/a.pt"}\n', encoding="utf-8")
    (vggt_root / "index.jsonl").write_text('{"sample_token":"a","path":"samples/a.pt"}\n', encoding="utf-8")

    report = preflight(jepa_root, vggt_root, strict=True)

    assert report["ok"] is False
    assert report["vggt_feature_dims"] == [768]
    assert report["num_jepa_legacy_fallback"] == 1
