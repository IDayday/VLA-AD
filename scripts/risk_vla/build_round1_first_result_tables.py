#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{float(value):.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_optional_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{_markdown_table(df)}\n", encoding="utf-8")


def transition_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    transition_cols = [
        "method_name",
        "num_matched_tokens",
        "path_repair_rate",
        "path_regression_rate",
        "interaction_regression_rate",
        "safe_path_repair_rate",
        "progress_recovery_rate",
        "comfort_recovery_rate",
        "num_path_repair",
        "num_path_regression",
        "num_interaction_regression",
        "num_safe_path_repair",
    ]
    available = [column for column in transition_cols if column in df.columns]
    return df[available] if available else df


def build_tables(
    *,
    matched_analysis_csv: Optional[Path],
    risk_metrics_csv: Optional[Path],
    strategy_activation_csv: Optional[Path],
    go_no_go_md: Optional[Path],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_df = _read_optional_csv(risk_metrics_csv)
    activation_df = _read_optional_csv(strategy_activation_csv)
    transition_df = transition_table(_read_optional_csv(matched_analysis_csv))

    _write_table(risk_df, output_dir / "round1_risk_prediction_table.csv", output_dir / "round1_risk_prediction_table.md", "R0 Risk Prediction Diagnostic")
    _write_table(
        activation_df,
        output_dir / "round1_strategy_activation_table.csv",
        output_dir / "round1_strategy_activation_table.md",
        "Strategy Activation Diagnostic",
    )
    if matched_analysis_csv is not None:
        _write_table(transition_df, output_dir / "round1_transition_table.csv", output_dir / "round1_transition_table.md", "A0/B3 Analysis Transition Table")

    go_text = go_no_go_md.read_text(encoding="utf-8") if go_no_go_md is not None and go_no_go_md.is_file() else "_Not provided._"
    summary = [
        "# RISK-VLA Round 1 First-Result Summary",
        "",
        "This summary separates A0/B3 analysis PDM transitions, R0 risk prediction diagnostics, strategy activation diagnostics, and GO/NO-GO status. It is not a final PDM improvement claim.",
        "",
        "BiT is a path/terminal intent strategy inside RISK-VLA; the core framework is risk state -> strategy routing/modulation -> diffusion planning.",
        "",
        "## A0/B3 Analysis PDM Transitions",
        "",
        _markdown_table(transition_df) if not transition_df.empty else "_Not provided._",
        "",
        "## R0 Risk Prediction Diagnostic",
        "",
        _markdown_table(risk_df) if not risk_df.empty else "_Not provided._",
        "",
        "## Strategy Activation Diagnostics",
        "",
        _markdown_table(activation_df) if not activation_df.empty else "_Not provided._",
        "",
        "## GO/NO-GO Decision",
        "",
        go_text,
        "",
    ]
    (output_dir / "round1_first_result_summary.md").write_text("\n".join(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build first-result tables for RISK-VLA Round 1.")
    parser.add_argument("--matched-analysis-csv", type=Path, default=None)
    parser.add_argument("--risk-metrics-csv", type=Path, default=None)
    parser.add_argument("--strategy-activation-csv", type=Path, default=None)
    parser.add_argument("--go-no-go-md", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build_tables(
        matched_analysis_csv=args.matched_analysis_csv,
        risk_metrics_csv=args.risk_metrics_csv,
        strategy_activation_csv=args.strategy_activation_csv,
        go_no_go_md=args.go_no_go_md,
        output_dir=args.output_dir,
    )
    print(f"Wrote first-result tables under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
