from __future__ import annotations

import pandas as pd

from scripts.risk_vla.build_round1_first_result_tables import build_tables


def test_first_result_tables_keep_sections_separate(tmp_path):
    transition = tmp_path / "transition.csv"
    pd.DataFrame(
        [
            {
                "method_name": "B3_direct_bit",
                "num_matched_tokens": 4,
                "path_repair_rate": 0.5,
                "interaction_regression_rate": 0.25,
            }
        ]
    ).to_csv(transition, index=False)
    risk = tmp_path / "risk.csv"
    pd.DataFrame([{"risk_class": "path_dac", "num_rows": 200, "enrichment_at_100": 1.6}]).to_csv(risk, index=False)
    activation = tmp_path / "activation.csv"
    pd.DataFrame([{"risk_subset": "path_dac", "mean_path_intent_positive": 0.7}]).to_csv(activation, index=False)
    go = tmp_path / "go.md"
    go.write_text("Decision: `GO_R1_R2`\n", encoding="utf-8")
    out = tmp_path / "tables"

    build_tables(
        matched_analysis_csv=transition,
        risk_metrics_csv=risk,
        strategy_activation_csv=activation,
        go_no_go_md=go,
        output_dir=out,
    )

    summary = (out / "round1_first_result_summary.md").read_text(encoding="utf-8")
    assert "A0/B3 Analysis PDM Transitions" in summary
    assert "R0 Risk Prediction Diagnostic" in summary
    assert "Strategy Activation Diagnostics" in summary
    assert "not a final PDM improvement claim" in summary
    assert (out / "round1_transition_table.csv").is_file()
