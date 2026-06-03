import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def torch_load(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def test_merge_risk_labels_dry_run_and_write(tmp_path):
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir()
    sample_a = chunk_dir / "a.pt"
    sample_b = chunk_dir / "b.pt"
    torch.save({"sample_token": "a", "existing": torch.tensor([1.0])}, sample_a)
    torch.save({"sample_token": "b", "existing": torch.tensor([2.0])}, sample_b)
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {
                "sample_token": "a",
                "generic_risk_labels": [1.0] * 4,
                "risk_labels": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4,
            },
            {
                "sample_token": "b",
                "generic_risk_labels": [0.5] * 4,
                "risk_labels": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/risk_vla/merge_risk_labels_to_chunk_cache.py",
            "--chunk-cache-dir",
            str(chunk_dir),
            "--labels-jsonl",
            str(labels),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )
    assert "generic_risk_labels" not in torch_load(sample_a)
    assert "risk_labels" not in torch_load(sample_b)

    subprocess.run(
        [
            sys.executable,
            "scripts/risk_vla/merge_risk_labels_to_chunk_cache.py",
            "--chunk-cache-dir",
            str(chunk_dir),
            "--labels-jsonl",
            str(labels),
            "--write",
        ],
        cwd=ROOT,
        check=True,
    )
    merged_a = torch_load(sample_a)
    merged_b = torch_load(sample_b)
    assert torch.is_tensor(merged_a["generic_risk_labels"])
    assert torch.is_tensor(merged_a["risk_labels"])
    assert tuple(merged_a["risk_labels"].shape) == (4, 6)
    assert torch.is_tensor(merged_b["generic_risk_labels"])
    assert "existing" in merged_b
