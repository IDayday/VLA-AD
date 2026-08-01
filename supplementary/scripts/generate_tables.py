#!/usr/bin/env python3
"""Generate appendix-ready CSV/LaTeX tables from ``stats.json`` only.

The displayed CSV and LaTeX cells are created from the same in-memory frames,
which prevents silent manual drift. Invalid canonical data is rejected before
any table is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    configure_logging,
    dataframe_to_latex,
    load_json,
    supplementary_root,
)


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Render generated statistical tables to matching CSV and LaTeX files."
    )
    parser.add_argument("--stats-json", type=Path, default=supp / "derived/stats.json")
    parser.add_argument(
        "--validation-status",
        type=Path,
        default=supp / "derived/data_validation_status.json",
    )
    parser.add_argument(
        "--table-config", type=Path, default=supp / "configs/table_generation.json"
    )
    parser.add_argument("--output-dir", type=Path, default=supp / "tables/generated")
    parser.add_argument(
        "--provenance", type=Path, default=supp / "derived/table_generation_provenance.json"
    )
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def display(value: Any, digits: int) -> str:
    if value is None or value is pd.NA:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def integer(value: Any) -> str:
    if value is None or value is pd.NA:
        return "--"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "--"


def mean_std(row: dict[str, Any], digits: int) -> str:
    mean = display(row.get("mean"), digits)
    std = display(row.get("std_across_runs"), digits)
    return mean if std == "--" else f"{mean} ± {std}"


def interval(row: dict[str, Any], low: str, high: str, digits: int) -> str:
    lo, hi = display(row.get(low), digits), display(row.get(high), digits)
    return "--" if "--" in (lo, hi) else f"[{lo}, {hi}]"


def method_table(records: list[dict[str, Any]], digits: int) -> pd.DataFrame:
    rows = []
    for row in records:
        if row.get("metric") != "aggregate_score":
            continue
        rows.append(
            {
                "Benchmark": row.get("benchmark", "--"),
                "Split": row.get("split", "--"),
                "Method": row.get("method", "--"),
                "Stage / variant": " / ".join(
                    str(value)
                    for value in (row.get("stage"), row.get("variant"))
                    if value is not None and str(value) not in {"", "nan", "<NA>"}
                )
                or "--",
                "Mean ± run std": mean_std(row, digits),
                "Scene-bootstrap CI": interval(
                    row, "scene_bootstrap_ci_low", "scene_bootstrap_ci_high", digits
                ),
                "Q10": display(row.get("q10"), digits),
                "CVaR-10": display(row.get("cvar10"), digits),
                "Runs": integer(row.get("n_runs")),
                "Scenes": integer(row.get("n_scenes")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Benchmark",
            "Split",
            "Method",
            "Stage / variant",
            "Mean ± run std",
            "Scene-bootstrap CI",
            "Q10",
            "CVaR-10",
            "Runs",
            "Scenes",
        ],
    )


def comparison_table(records: list[dict[str, Any]], digits: int) -> pd.DataFrame:
    rows = []
    for row in records:
        rows.append(
            {
                "Comparison": row.get("comparison_id", "--"),
                "Metric": row.get("metric", "--"),
                "A": row.get("label_a", "--"),
                "B": row.get("label_b", "--"),
                "Mean A": display(row.get("mean_a"), digits),
                "Mean B": display(row.get("mean_b"), digits),
                "Delta (B-A)": display(row.get("absolute_difference_b_minus_a"), digits),
                "Paired CI": interval(row, "ci_low_b_minus_a", "ci_high_b_minus_a", digits),
                "Effect dz": display(row.get("paired_effect_size_dz"), digits),
                "Runs": integer(row.get("n_runs")),
                "Scenes": integer(row.get("n_scenes")),
                "Evidence": row.get("run_evidence", row.get("status", "--")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Comparison",
            "Metric",
            "A",
            "B",
            "Mean A",
            "Mean B",
            "Delta (B-A)",
            "Paired CI",
            "Effect dz",
            "Runs",
            "Scenes",
            "Evidence",
        ],
    )


def recovery_table(records: list[dict[str, Any]], digits: int) -> pd.DataFrame:
    rows = []
    for row in records:
        rows.append(
            {
                "Benchmark": row.get("benchmark", "--"),
                "Method": row.get("method", "--"),
                "Recovered": integer(row.get("recovered_initial_failures")),
                "Persistent": integer(row.get("persistent_failures")),
                "Retained successes": integer(row.get("retained_successes")),
                "New failures": integer(row.get("new_failures")),
                "Net recovery": integer(row.get("net_recovery")),
                "Recovery rate": display(row.get("recovery_rate"), digits),
                "Wilson CI": interval(row, "recovery_ci_low", "recovery_ci_high", digits),
                "Runs": integer(row.get("n_runs")),
                "Scenes": integer(row.get("n_scenes")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Benchmark",
            "Method",
            "Recovered",
            "Persistent",
            "Retained successes",
            "New failures",
            "Net recovery",
            "Recovery rate",
            "Wilson CI",
            "Runs",
            "Scenes",
        ],
    )


def credit_table(records: list[dict[str, Any]], digits: int) -> pd.DataFrame:
    rows = []
    for row in records:
        rows.append(
            {
                "Benchmark": row.get("benchmark", "--"),
                "Method": row.get("method", "--"),
                "Positive-credit rollouts": integer(row.get("positive_credit_rollouts")),
                "Protected violations": integer(row.get("protected_condition_violations")),
                "Violation rate": display(row.get("positive_credit_violation_rate"), digits),
                "Wilson CI": interval(row, "violation_ci_low", "violation_ci_high", digits),
                "Dominated-positive rate": display(
                    row.get("dominated_positive_credit_rate"), digits
                ),
                "Runs": integer(row.get("n_runs")),
                "Scenes": integer(row.get("n_scenes")),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Benchmark",
            "Method",
            "Positive-credit rollouts",
            "Protected violations",
            "Violation rate",
            "Wilson CI",
            "Dominated-positive rate",
            "Runs",
            "Scenes",
        ],
    )


def write_table(
    frame: pd.DataFrame,
    *,
    stem: str,
    output_dir: Path,
    caption: str,
    label: str,
) -> list[Path]:
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    latex = dataframe_to_latex(frame, caption=caption, label=label)
    atomic_write_text(tex_path, latex)
    return [csv_path, tex_path]


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    validation = load_json(args.validation_status)
    if validation.get("status") == "invalid":
        raise RuntimeError("refusing to generate tables from invalid canonical data")
    stats = load_json(args.stats_json)
    config = load_json(args.table_config)
    digits = int(config.get("digits", 2))
    if digits < 0 or digits > 8:
        raise ValueError("table digits must be between zero and eight")
    inference = config.get(
        "inference_protocol",
        "single-trajectory inference without test-time scoring or reranking",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    definitions: list[tuple[str, pd.DataFrame, str, str]] = [
        (
            "statistical_method_summary",
            method_table(stats.get("method_summaries", []), digits),
            "Canonical scene-level aggregate results under "
            f"{inference}. Mean and run standard deviation report the observed run count; "
            "the interval resamples scenes and does not substitute for multi-run uncertainty. "
            "Aggregate score, Q10, and CVaR-10 are higher-is-better.",
            "tab:supp_method_statistics",
        ),
        (
            "paired_comparisons",
            comparison_table(stats.get("paired_comparisons", []), digits),
            "Paired scene-level comparisons under "
            f"{inference}. Each row reports observed runs and paired scenes. Positive differences "
            "favor method B for higher-is-better metrics; confidence intervals use the method stated in stats.json.",
            "tab:supp_paired_comparisons",
        ),
        (
            "failure_transitions",
            recovery_table(stats.get("recovery_summaries", []), digits),
            "Failure-state transitions on the benchmark shown under "
            f"{inference}. Recovery and net recovery are higher-is-better; new and persistent "
            "failures are lower-is-better. Rates use Wilson intervals and rows state runs/scenes.",
            "tab:supp_failure_transitions",
        ),
        (
            "positive_credit_diagnostics",
            credit_table(stats.get("positive_credit_summaries", []), digits),
            "Positive-credit diagnostics on the benchmark shown. Protected-condition and "
            "dominated-positive rates are lower-is-better. Wilson intervals, run counts, and "
            "scene counts are reported where the underlying rollout records are available.",
            "tab:supp_positive_credit",
        ),
    ]
    outputs: list[Path] = []
    for stem, frame, caption, label in definitions:
        outputs.extend(
            write_table(
                frame,
                stem=stem,
                output_dir=args.output_dir,
                caption=caption,
                label=label,
            )
        )
        logging.info("generated %s (%d rows)", stem, len(frame))

    provenance = build_provenance(
        seed=args.seed,
        inputs=[args.stats_json, args.validation_status, args.table_config],
        outputs=outputs,
        status="complete" if stats.get("status") == "complete" else "pending",
        notes=[
            "CSV and LaTeX cells were rendered from the same dataframes.",
            "No manually entered experimental values are used by this command.",
        ],
    )
    atomic_write_json(args.provenance, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
