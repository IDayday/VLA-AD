#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DECISION_GO_R1_ONLY = "GO_R1_ONLY"
DECISION_GO_R1_R2 = "GO_R1_R2"
DECISION_NO_GO_FIX_LABELS = "NO_GO_FIX_LABELS"
DECISION_NO_GO_FIX_EXPORT = "NO_GO_FIX_EXPORT"


def metric_row(df: pd.DataFrame, risk_class: str) -> Optional[pd.Series]:
    matches = df[df["risk_class"] == risk_class]
    return None if matches.empty else matches.iloc[0]


def decide_go_no_go(
    metrics_csv: Path,
    *,
    min_rows: int = 100,
    path_gate: float = 1.5,
    low_score_gate: float = 1.2,
    safety_gate: float = 1.2,
) -> Dict[str, object]:
    if not metrics_csv.is_file():
        return {"decision": DECISION_NO_GO_FIX_EXPORT, "reasons": [f"Risk metrics file missing: {metrics_csv}"]}
    df = pd.read_csv(metrics_csv)
    if df.empty or "risk_class" not in df.columns:
        return {"decision": DECISION_NO_GO_FIX_EXPORT, "reasons": ["Risk metrics CSV is empty or missing risk_class."]}
    max_rows = int(pd.to_numeric(df.get("num_rows", pd.Series([0])), errors="coerce").fillna(0).max())
    if max_rows < int(min_rows):
        return {
            "decision": DECISION_NO_GO_FIX_EXPORT,
            "reasons": [f"Only {max_rows} evaluated rows; require at least {min_rows}."],
        }

    required = ["path_dac", "low_score"]
    missing = [name for name in required if metric_row(df, name) is None]
    if metric_row(df, "interaction_nc") is None and metric_row(df, "ttc") is None:
        missing.append("interaction_nc_or_ttc")
    if missing:
        return {"decision": DECISION_NO_GO_FIX_LABELS, "reasons": [f"Missing metric rows: {', '.join(missing)}."]}

    def enrich(name: str) -> float:
        row = metric_row(df, name)
        return float(row.get("enrichment_at_100", 0.0)) if row is not None else 0.0

    positive_rates = {
        str(row["risk_class"]): float(row.get("positive_rate", 0.0))
        for _, row in df.iterrows()
        if "risk_class" in row
    }
    if any(positive_rates.get(name, 0.0) <= 0.0 for name in required):
        return {"decision": DECISION_NO_GO_FIX_LABELS, "reasons": ["Required risk classes have zero positive rate."]}

    path_ok = enrich("path_dac") >= float(path_gate)
    low_ok = enrich("low_score") >= float(low_score_gate)
    safety_enrichment = max(enrich("interaction_nc"), enrich("ttc"))
    safety_ok = safety_enrichment >= float(safety_gate)
    reasons = [
        f"path_dac enrichment@100={enrich('path_dac'):.3f} gate={path_gate:.3f}",
        f"low_score enrichment@100={enrich('low_score'):.3f} gate={low_score_gate:.3f}",
        f"safety enrichment@100={safety_enrichment:.3f} gate={safety_gate:.3f}",
    ]
    decision = DECISION_GO_R1_R2 if path_ok and low_ok and safety_ok else DECISION_GO_R1_ONLY
    if decision == DECISION_GO_R1_ONLY:
        reasons.append("Predicted-router evidence is weak; R1 oracle-router may proceed as analysis-only.")
    return {"decision": decision, "reasons": reasons, "num_rows": max_rows}


def write_report(result: Dict[str, object], output_md: Path) -> None:
    lines = ["# RISK-VLA Round 1 GO/NO-GO", "", f"Decision: `{result['decision']}`", "", "## Reasons", ""]
    lines.extend(f"- {reason}" for reason in result.get("reasons", []))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "R1 oracle-router is analysis-only. R2 predicted-router should proceed only when R0 prediction evidence is strong enough.",
            "",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide whether Round 1 should proceed from R0 to R1/R2.")
    parser.add_argument("--risk-metrics-csv", type=Path, required=True)
    parser.add_argument("--strategy-activation-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    result = decide_go_no_go(args.risk_metrics_csv)
    write_report(result, args.output_md)
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
