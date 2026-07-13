from __future__ import annotations

import json

import torch

from scripts.audit_two_expert_hidden_cache import audit_cache


def test_two_expert_hidden_cache_audit_synthetic_fixture(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    payload = {
        "sample_token": "sample_a",
        "last_hidden_state": torch.randn(7, 1536),
        "two_expert_h_dyn": torch.randn(3, 12, 1536),
        "two_expert_h_geo": torch.randn(12, 1536),
        "two_expert_metadata": {"schema": "two_expert_slot_hidden_cache_v1"},
        "history_trajectory": torch.randn(4, 3),
        "high_command_one_hot": torch.tensor([1.0, 0.0, 0.0]),
        "status_feature": torch.randn(8),
        "trajectory": torch.randn(8, 3),
    }
    torch.save(payload, samples / "sample_a.pt")
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"sample_token": "sample_a", "path": "samples/sample_a.pt"}) + "\n",
        encoding="utf-8",
    )

    report = audit_cache(tmp_path, split="navtest")
    assert report["ok"] is True
    assert report["total_records"] == 1
    assert report["teacher_coverage"]["jepa_dynamic_teacher_tokens"] == 0.0
