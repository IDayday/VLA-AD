from __future__ import annotations

import pandas as pd

from scripts.risk_vla.build_round1_report import build_report


def test_round1_report_builder_outputs_required_sections(tmp_path):
    matched = tmp_path / "matched.csv"
    diagnostics = tmp_path / "diagnostics.csv"
    transitions = tmp_path / "transitions.csv"
    activation = tmp_path / "activation.csv"
    pd.DataFrame({"token": ["a"], "label_low_score": [1]}).to_csv(matched, index=False)
    pd.DataFrame({"risk_class": ["low_score"], "auc": [0.5]}).to_csv(diagnostics, index=False)
    pd.DataFrame({"method": ["R2"], "num_matched_tokens": [1]}).to_csv(transitions, index=False)
    pd.DataFrame({"risk_subset": ["low_score"], "mean_path_intent_positive": [0.2]}).to_csv(activation, index=False)
    output = tmp_path / "report.md"

    build_report(
        matched_pdm_csv=matched,
        risk_diagnostics_csv=diagnostics,
        transition_csv=transitions,
        strategy_activation_csv=activation,
        output_md=output,
    )

    text = output.read_text(encoding="utf-8")
    assert "Experiment Registry Summary" in text
    assert "Risk Label Distribution" in text
    assert "Strategy Activation Diagnostics" in text
    assert "low-score risk scenarios" in text
    assert "BiT is treated as a path/terminal intent strategy" in text
