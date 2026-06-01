from __future__ import annotations

import torch

from scripts.audit_last_rd_adapter_checkpoint import audit_checkpoint


def test_last_rd_adapter_audit_synthetic(tmp_path):
    adapter_path = tmp_path / "adapter.pt"
    torch.save({"state_dict": {"action_head.last_rd.dynamic_queries": torch.zeros(2, 3)}}, adapter_path)

    report = audit_checkpoint(adapter_path)
    assert report["pass"]
    assert report["num_last_rd_tensors"] == 1
    assert report["num_legacy_a4_tensors"] == 0

    mixed_path = tmp_path / "mixed_adapter.pt"
    torch.save({
        "state_dict": {
            "action_head.last_rd.dynamic_queries": torch.zeros(2, 3),
            "action_head.jepa_projector.weight": torch.zeros(3, 3),
        }
    }, mixed_path)

    mixed_report = audit_checkpoint(mixed_path)
    assert not mixed_report["pass"]
    assert mixed_report["num_legacy_a4_tensors"] == 1
    assert "legacy_a4_tensors_present" in mixed_report["warnings"]

    allowed_report = audit_checkpoint(mixed_path, allow_legacy=True)
    assert allowed_report["pass"]
    assert allowed_report["num_legacy_a4_tensors"] == 1
