from __future__ import annotations

import json

import pandas as pd

from scripts.risk_vla.aggregate_risk_vla_diagnostics import aggregate, write_markdown


def test_aggregate_risk_vla_diagnostics_tiny_csv_jsonl(tmp_path):
    pred_path = tmp_path / "pred.csv"
    labels_path = tmp_path / "labels.jsonl"
    out_md = tmp_path / "summary.md"
    pd.DataFrame(
        {
            "token": ["a", "b", "c"],
            "risk_vla_prob_low_score": [0.9, 0.2, 0.1],
            "risk_vla_prob_path_dac": [0.1, 0.8, 0.3],
            "risk_vla_prob_interaction_nc": [0.2, 0.6, 0.4],
            "risk_vla_prob_ttc": [0.7, 0.2, 0.1],
            "risk_vla_prob_progress": [0.1, 0.2, 0.8],
            "risk_vla_prob_comfort": [0.3, 0.4, 0.9],
        }
    ).to_csv(pred_path, index=False)
    rows = [
        {"token": "a", "risk_labels": [1, 0, 0, 1, 0, 0]},
        {"token": "b", "risk_labels": [0, 1, 1, 0, 0, 0]},
        {"token": "c", "risk_labels": [0, 0, 0, 0, 1, 1]},
    ]
    labels_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary, warnings_list = aggregate(pred_path, labels_path)
    assert set(summary["risk_class"]) == {"low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"}
    assert "brier_score" in summary.columns
    write_markdown(summary, warnings_list, out_md)
    assert "RISK-VLA Diagnostic Aggregation" in out_md.read_text(encoding="utf-8")
