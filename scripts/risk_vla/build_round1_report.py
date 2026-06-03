#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


DEFAULT_REGISTRY = Path("configs/risk_vla/round1_experiment_registry.yaml")


def _read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.is_file():
        return None
    return pd.read_csv(path)


def _table(df: Optional[pd.DataFrame], max_rows: int = 20) -> str:
    if df is None:
        return "_Not available._"
    if df.empty:
        return "_Empty input._"
    limited = df.head(max_rows)
    columns = list(limited.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in limited.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def _registry_summary(path: Path) -> str:
    if not path.is_file():
        return "_Registry not found._"
    registry = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiments = registry.get("experiments") or {}
    lines = ["| experiment | role | description |", "| --- | --- | --- |"]
    for name, spec in experiments.items():
        lines.append(f"| {name} | {spec.get('role', '')} | {spec.get('description', '')} |")
    return "\n".join(lines)


def build_report(
    *,
    matched_pdm_csv: Optional[Path],
    risk_diagnostics_csv: Optional[Path],
    transition_csv: Optional[Path],
    strategy_activation_csv: Optional[Path],
    output_md: Path,
    registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    matched = _read_csv(matched_pdm_csv)
    diagnostics = _read_csv(risk_diagnostics_csv)
    transitions = _read_csv(transition_csv)
    activation = _read_csv(strategy_activation_csv)

    lines = [
        "# RISK-VLA Round 1 Report",
        "",
        "This report covers low-score risk scenarios / critical-risk subsets. BiT is treated as a path/terminal intent strategy inside the RISK-VLA strategy bank, not as the core method. RISK-VLA is the risk-conditioned multi-strategy framework: risk state -> strategy routing/modulation -> diffusion planning.",
        "",
        "Oracle-router results are analysis-only. Navtest/test labels must not be used for training.",
        "",
        "## 1. Experiment Registry Summary",
        "",
        _registry_summary(registry_path),
        "",
        "## 2. Risk Label Distribution",
        "",
        _table(matched),
        "",
        "## 3. Risk Prediction Diagnostic Metrics",
        "",
        _table(diagnostics),
        "",
        "## 4. Strategy Activation Diagnostics",
        "",
        _table(activation),
        "",
        "## 5. Base vs BiT vs RISK-VLA Transition Metrics",
        "",
        _table(transitions),
        "",
        "## 6. Open Issues / Next Experiment Recommendations",
        "",
        "- Keep risk diagnostic, oracle-router, and predicted-router results separate.",
        "- If risk prediction is weak on train/val labels, improve label quality before scaling pilots.",
        "- If oracle routing helps but predicted routing does not, prioritize risk-state supervision and calibration.",
        "- Do not claim final PDM gains from Stage 4 smoke or small pilot outputs.",
        "",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RISK-VLA round1 Markdown report.")
    parser.add_argument("--matched-pdm-csv", type=Path, default=None)
    parser.add_argument("--risk-diagnostics-csv", type=Path, default=None)
    parser.add_argument("--transition-csv", type=Path, default=None)
    parser.add_argument("--strategy-activation-csv", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    output_md = args.output_md
    if args.output_dir is not None and not output_md.is_absolute():
        output_md = args.output_dir / output_md
    build_report(
        matched_pdm_csv=args.matched_pdm_csv,
        risk_diagnostics_csv=args.risk_diagnostics_csv,
        transition_csv=args.transition_csv,
        strategy_activation_csv=args.strategy_activation_csv,
        output_md=output_md,
        registry_path=args.registry,
    )
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
