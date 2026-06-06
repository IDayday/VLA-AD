import csv
import json
from pathlib import Path

import pytest

from scripts.risk_vla.build_finegrained_risk_labels import build_finegrained_labels, write_outputs
from scripts.risk_vla.merge_finegrained_labels_to_chunk_cache import merge_labels


def _csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_finegrained_labels_add_horizon_and_safety_labels(tmp_path):
    table = tmp_path / "pdm.csv"
    _csv(table, [{"sample_token": "a", "score": 0.0, "dac": 0, "nc": 0, "ttc": 1, "progress": 0.4, "comfort": 1}])
    rows, summary = build_finegrained_labels(table, split="train", purpose="training", horizon=6)
    assert summary["num_rows"] == 1
    row = rows[0]
    assert row["risk_labels_scene"] == [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert len(row["risk_labels_horizon"]) == 6
    assert len(row["safety_risk_horizon"]) == 6

    out = tmp_path / "labels"
    write_outputs(rows, summary, out)
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "index.jsonl").write_text(json.dumps({"sample_token": "a", "x": 1}) + "\n", encoding="utf-8")
    merge_summary = merge_labels(chunk, out / "finegrained_risk_labels.jsonl", tmp_path / "overlay")
    assert merge_summary["matched_labels"] == 1


def test_finegrained_labels_reject_navtest_training(tmp_path):
    table = tmp_path / "pdm.csv"
    _csv(table, [{"sample_token": "a", "score": 1, "dac": 1, "nc": 1, "ttc": 1, "progress": 1, "comfort": 1}])
    with pytest.raises(RuntimeError):
        build_finegrained_labels(table, split="navtest", purpose="training")
