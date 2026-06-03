from __future__ import annotations

import torch

from scripts.risk_vla.resolve_r0_diagnostic_artifacts import resolve_artifacts, write_markdown


def test_resolve_r0_artifacts_handles_missing_checkpoint(tmp_path):
    report = resolve_artifacts(tmp_path)
    assert report["has_checkpoint"] is False
    assert "synthetic-smoke" in report["recommendation"]


def test_resolve_r0_artifacts_finds_best_checkpoint(tmp_path):
    r0 = tmp_path / "r0"
    (r0 / "train").mkdir(parents=True)
    torch.save({"state_dict": {}}, r0 / "train" / "last.ckpt")
    torch.save({"state_dict": {}}, r0 / "train" / "best.ckpt")
    (r0 / "train.log").write_text("ok\n", encoding="utf-8")

    report = resolve_artifacts(r0)
    assert report["has_checkpoint"] is True
    assert report["best_checkpoint"].endswith("best.ckpt")
    output = tmp_path / "artifacts.md"
    write_markdown(report, output)
    assert "Checkpoint Candidates" in output.read_text(encoding="utf-8")
