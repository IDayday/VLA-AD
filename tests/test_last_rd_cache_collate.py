from __future__ import annotations

import pytest
import torch


def test_last_rd_synthetic_optional_expert_stack():
    try:
        from navsim.agents.recogdrive.recogdrive_features import stack_optional_expert_features
    except Exception as exc:
        pytest.skip(f"training dependencies unavailable: {exc}")

    features_list = [
        {
            "jepa_context_tokens": torch.randn(12, 1024),
            "vggt_context_tokens": torch.randn(12, 2048),
            "vggt_geometry_tokens": torch.randn(12, 2048),
            "jepa_target_tokens": torch.randn(12, 1024),
            "vggt_target_tokens": torch.randn(12, 2048),
            "risk_labels": torch.zeros(8, 4),
        },
        {
            "jepa_context_tokens": torch.randn(12, 1024),
            "vggt_context_tokens": torch.randn(12, 2048),
            "vggt_geometry_tokens": torch.randn(12, 2048),
            "jepa_target_tokens": torch.randn(12, 1024),
            "vggt_target_tokens": torch.randn(12, 2048),
            "risk_labels": torch.zeros(8, 4),
        },
    ]
    collated = {}
    stack_optional_expert_features(collated, features_list)
    assert collated["jepa_context_tokens"].shape == (2, 12, 1024)
    assert collated["vggt_geometry_tokens"].shape == (2, 12, 2048)
    assert collated["risk_labels"].shape == (2, 8, 4)

    broken = [dict(features_list[0]), dict(features_list[1])]
    broken[1].pop("vggt_geometry_tokens")
    with pytest.raises(KeyError):
        stack_optional_expert_features({}, broken)
