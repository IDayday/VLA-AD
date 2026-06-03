from __future__ import annotations

import pandas as pd

from scripts.risk_vla.round1.check_round1_go_no_go import (
    DECISION_GO_R1_ONLY,
    DECISION_GO_R1_R2,
    DECISION_NO_GO_FIX_EXPORT,
    decide_go_no_go,
    write_report,
)


def _metrics(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_go_no_go_missing_metrics_is_export_fix(tmp_path):
    result = decide_go_no_go(tmp_path / "missing.csv")
    assert result["decision"] == DECISION_NO_GO_FIX_EXPORT


def test_go_no_go_passes_r1_r2_for_strong_metrics(tmp_path):
    path = tmp_path / "metrics.csv"
    _metrics(
        path,
        [
            {"risk_class": "path_dac", "num_rows": 200, "positive_rate": 0.2, "enrichment_at_100": 1.6},
            {"risk_class": "low_score", "num_rows": 200, "positive_rate": 0.2, "enrichment_at_100": 1.3},
            {"risk_class": "interaction_nc", "num_rows": 200, "positive_rate": 0.1, "enrichment_at_100": 1.25},
        ],
    )
    result = decide_go_no_go(path)
    assert result["decision"] == DECISION_GO_R1_R2
    report = tmp_path / "go.md"
    write_report(result, report)
    assert "GO/NO-GO" in report.read_text(encoding="utf-8")


def test_go_no_go_weak_metrics_allow_r1_only(tmp_path):
    path = tmp_path / "metrics.csv"
    _metrics(
        path,
        [
            {"risk_class": "path_dac", "num_rows": 200, "positive_rate": 0.2, "enrichment_at_100": 1.1},
            {"risk_class": "low_score", "num_rows": 200, "positive_rate": 0.2, "enrichment_at_100": 1.3},
            {"risk_class": "ttc", "num_rows": 200, "positive_rate": 0.1, "enrichment_at_100": 1.0},
        ],
    )
    assert decide_go_no_go(path)["decision"] == DECISION_GO_R1_ONLY
