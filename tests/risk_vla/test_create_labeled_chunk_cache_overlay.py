from __future__ import annotations

import json

import torch

from scripts.risk_vla.create_labeled_chunk_cache_overlay import create_overlay, load_sample


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_overlay_dry_run_does_not_create_overlay_or_mutate_source(tmp_path):
    source = tmp_path / "source"
    (source / "samples").mkdir(parents=True)
    sample_path = source / "samples" / "a.pt"
    original = {"sample_token": "a", "value": torch.tensor([1.0])}
    torch.save(original, sample_path)
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [{"sample_token": "a", "split": "navval", "risk_labels": [1, 0, 0, 0, 0, 0]}])
    output = tmp_path / "overlay"

    rows = create_overlay(source, output, labels, mode="copy", dry_run=True, write=False)

    assert len(rows) == 1
    assert not output.exists()
    reloaded = load_sample(sample_path)
    assert "risk_labels" not in reloaded
    assert torch.equal(reloaded["value"], original["value"])


def test_overlay_write_copy_injects_risk_tensor_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    (source / "samples").mkdir(parents=True)
    source_sample = source / "samples" / "a.pt"
    torch.save({"sample_token": "a", "value": torch.tensor([2.0])}, source_sample)
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(labels, [{"sample_token": "a", "split": "navtrain", "risk_labels": [1, 0, 0, 0, 1, 0]}])
    output = tmp_path / "overlay"

    rows = create_overlay(source, output, labels, mode="copy", dry_run=False, write=True)

    assert rows[0]["operation"] == "copied_injected"
    copied = load_sample(output / "samples" / "a.pt")
    assert "risk_labels" in copied
    assert tuple(copied["risk_labels"].shape) == (6,)
    assert "risk_labels" not in load_sample(source_sample)
