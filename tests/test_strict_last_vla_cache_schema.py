from __future__ import annotations

import torch

from scripts.audit_strict_last_vla_latent_cache import audit_records


def test_strict_last_vla_cache_schema_synthetic_fixture():
    records = [
        {
            "sample_token": "token_1",
            "last_hidden_state": torch.randn(7, 1536),
            "last_vla_h_dyn": torch.randn(64, 1536),
            "last_vla_h_geo": torch.randn(64, 1536),
            "last_vla_h_plan": torch.randn(32, 1536),
            "last_vla_slot_metadata": {"dyn_count": 64, "geo_count": 64, "plan_count": 32},
            "jepa_target_tokens": torch.randn(128, 1024),
            "vggt_geometry_tokens": torch.randn(192, 512),
        }
    ]
    report = audit_records(records, split="train", require_train_teachers=True)
    assert report["ok"]
    assert report["total_records"] == 1
