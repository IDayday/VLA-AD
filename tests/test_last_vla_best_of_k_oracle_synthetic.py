from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.eval_last_vla_best_of_k_oracle import oracle_metrics, synthetic_rows


def test_synthetic_oracle_metrics_schema():
    rows = synthetic_rows(num_samples=4, k=8, seed=123)
    metrics = oracle_metrics(rows, k=8, score_mode="proxy")
    assert metrics["num_samples"] == 4
    assert metrics["K"] == 8
    assert metrics["score_mode"] == "proxy"
    assert metrics["oracle_best_of_K_PDMS"] is None
    assert metrics["oracle_best_of_K_score"] >= metrics["stochastic_mean_score"]
    assert "oracle_delta_vs_A0_baseline" in metrics


def test_oracle_script_synthetic_smoke_writes_json(tmp_path: Path):
    output_dir = tmp_path / "oracle"
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_last_vla_best_of_k_oracle.py",
            "--synthetic-smoke",
            "--num-candidates",
            "4",
            "--max-samples",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = json.loads((output_dir / "rows.json").read_text(encoding="utf-8"))
    assert metrics["num_samples"] == 3
    assert metrics["K"] == 4
    assert metrics["score_mode"] == "proxy"
    assert metrics["oracle_best_of_K_PDMS"] is None
    assert len(rows) == 3
