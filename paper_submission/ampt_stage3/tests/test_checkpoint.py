from __future__ import annotations

import torch
import pytest

from ampt_stage3.checkpoint import inspect_checkpoint, sha256_file


def test_checkpoint_hash_and_action_head_contract(tmp_path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    torch.save({"state_dict": {"agent.action_head.model.weight": torch.ones(1)}}, checkpoint)
    digest = sha256_file(checkpoint)
    report = inspect_checkpoint(checkpoint, expected_sha256=digest)
    assert report.action_head_keys == 1
    assert report.artifact_id == f"weight-{digest[:12]}"
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_checkpoint(checkpoint, expected_sha256="0" * 64)
