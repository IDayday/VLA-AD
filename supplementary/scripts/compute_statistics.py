#!/usr/bin/env python3
"""Compute reproducible descriptive, paired, recovery, and credit statistics.

Comparison definitions come from a checked-in JSON file.  The script does not
guess which methods should be compared, and it marks fewer than three observed
training/evaluation seeds as insufficient rather than inventing variance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from statistics import NormalDist
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    METRIC_COLUMNS,
    atomic_write_json,
    build_provenance,
    configure_logging,
    read_table,
    relative_display,
    repository_root,
    sha256_file,
    supplementary_root,
)


GROUP_COLUMNS = ["benchmark", "split", "method", "stage", "variant", "round"]


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Compute scene-aware supplementary statistics with bootstrap/Wilson intervals."
    )
    parser.add_argument(
        "--input", type=Path, default=supp / "derived/canonical_scene_metrics.parquet"
    )
    parser.add_argument(
        "--validation-status",
        type=Path,
        default=supp / "derived/data_validation_status.json",
        help="Validation status JSON; invalid data aborts statistics.",
    )
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=supp / "configs/statistical_comparisons.json",
    )
    parser.add_argument("--output-json", type=Path, default=supp / "derived/stats.json")
    parser.add_argument("--output-dir", type=Path, default=supp / "derived/statistics")
    parser.add_argument(
        "--provenance", type=Path, default=supp / "derived/statistics_provenance.json"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} root must be a JSON object")
    return payload


def selector_mask(frame: pd.DataFrame, selector: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, expected in selector.items():
        if column not in frame:
            raise ValueError(f"selector references absent column {column!r}")
        if isinstance(expected, list):
            mask &= frame[column].isin(expected)
        elif expected is None:
            mask &= frame[column].isna()
        else:
            mask &= frame[column].astype(str).eq(str(expected))
    return mask


def percentile_interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, samples: int, confidence: float
) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return percentile_interval(draws, confidence)


def paired_bootstrap(
    paired: pd.DataFrame,
    *,
    seed_column: str | None,
    rng: np.random.Generator,
    samples: int,
    confidence: float,
) -> tuple[float, float, str]:
    """Paired bootstrap, hierarchical over seeds when multiple seeds exist."""

    diff = paired["difference"].to_numpy(dtype=float)
    if not len(diff):
        raise ValueError("paired bootstrap received no observations")
    if seed_column and seed_column in paired and paired[seed_column].notna().all():
        seeds = paired[seed_column].unique()
    else:
        seeds = np.array([])
    estimates = np.empty(samples, dtype=float)
    if len(seeds) > 1:
        grouped = {
            seed: paired.loc[paired[seed_column].eq(seed), "difference"].to_numpy(dtype=float)
            for seed in seeds
        }
        for index in range(samples):
            selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            chunks = []
            for selected in selected_seeds:
                values = grouped[selected]
                chunks.append(rng.choice(values, size=len(values), replace=True))
            estimates[index] = np.concatenate(chunks).mean()
        kind = "hierarchical_seed_scene_paired_bootstrap"
    else:
        for index in range(samples):
            estimates[index] = rng.choice(diff, size=len(diff), replace=True).mean()
        kind = "scene_paired_bootstrap_single_seed" if len(seeds) == 1 else "scene_paired_bootstrap_seed_unknown"
    low, high = percentile_interval(estimates, confidence)
    return low, high, kind


def wilson_interval(successes: int, total: int, confidence: float) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def analysis_units(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    columns = ["scene_id"]
    if "seed" in group and group["seed"].notna().any():
        columns.append("seed")
    unit = (
        group.loc[group[metric].notna(), columns + [metric]]
        .groupby(columns, dropna=False, as_index=False)[metric]
        .mean()
    )
    return unit


def method_summaries(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
    minimum_runs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        metadata = dict(zip(GROUP_COLUMNS, group_key))
        n_runs = int(group["seed"].dropna().nunique())
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            units = analysis_units(group, metric)
            values = pd.to_numeric(units[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            if "seed" in units and units["seed"].notna().any():
                run_means = units.groupby("seed", dropna=True)[metric].mean().to_numpy(float)
            else:
                run_means = np.array([], dtype=float)
            ci_low, ci_high = bootstrap_mean(values, rng, bootstrap_samples, confidence)
            cutoff = float(np.quantile(values, 0.1))
            tail = values[values <= cutoff]
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std_across_runs": float(run_means.std(ddof=1)) if len(run_means) >= 2 else None,
                    "scene_bootstrap_ci_low": ci_low,
                    "scene_bootstrap_ci_high": ci_high,
                    "q10": cutoff,
                    "cvar10": float(tail.mean()) if len(tail) else None,
                    "n_records": int(group[metric].notna().sum()),
                    "n_scenes": int(units["scene_id"].nunique()),
                    "n_runs": n_runs,
                    "run_evidence": "sufficient" if n_runs >= minimum_runs else "insufficient",
                    "ci_scope": "scene-level; does not replace multi-run uncertainty",
                }
            )
    return rows


def prepare_pair(
    frame: pd.DataFrame,
    metric: str,
    selector_a: dict[str, Any],
    selector_b: dict[str, Any],
    pair_on: list[str],
) -> tuple[pd.DataFrame, int, int]:
    if metric not in frame:
        raise ValueError(f"comparison metric {metric!r} is absent")
    left = frame.loc[selector_mask(frame, selector_a), pair_on + [metric]].dropna(subset=[metric])
    right = frame.loc[selector_mask(frame, selector_b), pair_on + [metric]].dropna(subset=[metric])
    if left.duplicated(pair_on).any():
        raise ValueError(f"selector_a has duplicate pair keys {pair_on}; narrow the selector")
    if right.duplicated(pair_on).any():
        raise ValueError(f"selector_b has duplicate pair keys {pair_on}; narrow the selector")
    left_count, right_count = len(left), len(right)
    paired = left.merge(right, on=pair_on, how="inner", suffixes=("_a", "_b"))
    paired["difference"] = paired[f"{metric}_b"] - paired[f"{metric}_a"]
    return paired, left_count - len(paired), right_count - len(paired)


def comparison_summaries(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
    minimum_runs: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, spec in enumerate(specs):
        comparison_id = str(spec.get("id", f"comparison_{index}"))
        metric = str(spec.get("metric", "aggregate_score"))
        selector_a = spec.get("selector_a")
        selector_b = spec.get("selector_b")
        if not isinstance(selector_a, dict) or not isinstance(selector_b, dict):
            raise ValueError(f"{comparison_id}: selector_a and selector_b must be JSON objects")
        pair_on = list(spec.get("pair_on", ["scene_id", "seed"]))
        pair_on = [column for column in pair_on if column in frame]
        if "scene_id" not in pair_on:
            raise ValueError(f"{comparison_id}: pair_on must include scene_id")
        if "seed" in pair_on and frame["seed"].isna().all():
            pair_on.remove("seed")
            notes.append(f"{comparison_id}: seed is unavailable; pairing used scene_id only.")
        paired, unmatched_a, unmatched_b = prepare_pair(
            frame, metric, selector_a, selector_b, pair_on
        )
        if paired.empty:
            notes.append(f"{comparison_id}: selectors yielded no paired records.")
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "status": "pending_no_pairs",
                    "n_pairs": 0,
                    "n_scenes": 0,
                    "n_runs": 0,
                }
            )
            continue
        seed_column = "seed" if "seed" in pair_on else None
        n_runs = int(paired["seed"].nunique()) if seed_column else 0
        low, high, ci_kind = paired_bootstrap(
            paired,
            seed_column=seed_column,
            rng=rng,
            samples=bootstrap_samples,
            confidence=confidence,
        )
        a = paired[f"{metric}_a"].to_numpy(float)
        b = paired[f"{metric}_b"].to_numpy(float)
        diff = b - a
        raw_difference = float(diff.mean())
        higher_is_better = bool(spec.get("higher_is_better", True))
        improvement = raw_difference if higher_is_better else -raw_difference
        baseline = float(a.mean())
        effect = float(diff.mean() / diff.std(ddof=1)) if len(diff) >= 2 and diff.std(ddof=1) > 0 else None
        rows.append(
            {
                "comparison_id": comparison_id,
                "label_a": spec.get("label_a", selector_a.get("method", "A")),
                "label_b": spec.get("label_b", selector_b.get("method", "B")),
                "metric": metric,
                "mean_a": baseline,
                "mean_b": float(b.mean()),
                "absolute_difference_b_minus_a": raw_difference,
                "relative_difference_percent": 100.0 * raw_difference / abs(baseline) if baseline != 0 else None,
                "improvement_signed": improvement,
                "paired_effect_size_dz": effect,
                "ci_low_b_minus_a": low,
                "ci_high_b_minus_a": high,
                "confidence": confidence,
                "ci_method": ci_kind,
                "n_pairs": int(len(paired)),
                "n_scenes": int(paired["scene_id"].nunique()),
                "n_runs": n_runs,
                "run_evidence": "sufficient" if n_runs >= minimum_runs else "insufficient",
                "unmatched_a": int(unmatched_a),
                "unmatched_b": int(unmatched_b),
                "status": "complete",
            }
        )
    return rows, notes


def recovery_summaries(frame: pd.DataFrame, confidence: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = ["initial_failure", "recovered", "new_failure"]
    if any(frame[column].isna().all() for column in required):
        return rows
    for group_key, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        metadata = dict(zip(GROUP_COLUMNS, group_key))
        evaluable = group.dropna(subset=required)
        if evaluable.empty:
            continue
        unit_keys = ["scene_id"] + (["seed"] if evaluable["seed"].notna().any() else [])
        state_counts = evaluable.groupby(unit_keys, dropna=False)[required].nunique(dropna=True)
        if (state_counts > 1).any(axis=None):
            raise ValueError(
                f"recovery flags conflict within scene/run for group {metadata}; "
                "narrow the canonical stage/variant/round metadata"
            )
        raw_record_count = len(evaluable)
        evaluable = evaluable.drop_duplicates(unit_keys, keep="first")
        initial = evaluable["initial_failure"].eq(True)
        recovered = initial & evaluable["recovered"].eq(True)
        persistent = initial & evaluable["recovered"].eq(False)
        initial_success = ~initial
        new_failure = initial_success & evaluable["new_failure"].eq(True)
        retained = initial_success & evaluable["new_failure"].eq(False)
        recovery_low, recovery_high = wilson_interval(int(recovered.sum()), int(initial.sum()), confidence)
        new_low, new_high = wilson_interval(int(new_failure.sum()), int(initial_success.sum()), confidence)
        rows.append(
            {
                **metadata,
                "recovered_initial_failures": int(recovered.sum()),
                "persistent_failures": int(persistent.sum()),
                "retained_successes": int(retained.sum()),
                "new_failures": int(new_failure.sum()),
                "net_recovery": int(recovered.sum() - new_failure.sum()),
                "initial_failure_count": int(initial.sum()),
                "initial_success_count": int(initial_success.sum()),
                "recovery_rate": float(recovered.sum() / initial.sum()) if initial.any() else None,
                "recovery_ci_low": recovery_low,
                "recovery_ci_high": recovery_high,
                "new_failure_rate": float(new_failure.sum() / initial_success.sum()) if initial_success.any() else None,
                "new_failure_ci_low": new_low,
                "new_failure_ci_high": new_high,
                "confidence": confidence,
                "ci_method": "Wilson score interval",
                "n_source_records": int(raw_record_count),
                "n_scenes": int(evaluable["scene_id"].nunique()),
                "n_runs": int(evaluable["seed"].dropna().nunique()),
            }
        )
    return rows


def credit_summaries(
    frame: pd.DataFrame, protected_pass_columns: list[str], confidence: float
) -> tuple[list[dict[str, Any]], list[str]]:
    if not protected_pass_columns:
        return [], ["Positive-credit violation rate is pending: no protected-pass columns are configured."]
    missing = [column for column in protected_pass_columns if column not in frame]
    if missing:
        raise ValueError(f"protected-pass columns are absent: {missing}")
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for group_key, group in frame.groupby(GROUP_COLUMNS, dropna=False):
        metadata = dict(zip(GROUP_COLUMNS, group_key))
        positive = group.loc[group["positive_credit"].eq(True)].copy()
        if positive.empty:
            continue
        evaluable = positive.dropna(subset=protected_pass_columns)
        if evaluable.empty:
            notes.append(f"{metadata}: positive-credit rows lack protected-condition observations.")
            continue
        violation = ~evaluable[protected_pass_columns].all(axis=1)
        violations = int(violation.sum())
        total = int(len(evaluable))
        low, high = wilson_interval(violations, total, confidence)
        dominated = (
            int(evaluable["pareto_front"].eq(False).sum())
            if evaluable["pareto_front"].notna().any()
            else None
        )
        rows.append(
            {
                **metadata,
                "positive_credit_rollouts": int(len(positive)),
                "evaluable_positive_credit_rollouts": total,
                "protected_condition_violations": violations,
                "positive_credit_violation_rate": violations / total,
                "violation_ci_low": low,
                "violation_ci_high": high,
                "dominated_positive_credit_rollouts": dominated,
                "dominated_positive_credit_rate": dominated / total if dominated is not None else None,
                "protected_pass_columns": ",".join(protected_pass_columns),
                "confidence": confidence,
                "ci_method": "Wilson score interval",
                "n_scenes": int(evaluable["scene_id"].nunique()),
                "n_runs": int(evaluable["seed"].dropna().nunique()),
            }
        )
    return rows, notes


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def write_records_csv(records: list[dict[str, Any]], path: Path, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0 or args.bootstrap_samples < 1 or args.minimum_runs < 1:
        raise ValueError("seed must be non-negative; bootstrap samples/minimum runs must be positive")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must lie strictly between zero and one")
    validation = load_json_object(args.validation_status, "validation status")
    if validation.get("status") == "invalid":
        raise RuntimeError("canonical dataset is invalid; inspect data_validation_report.md")
    if not args.input.exists():
        raise FileNotFoundError(f"canonical dataset not found: {args.input}")
    frame = read_table(args.input)
    config = load_json_object(args.comparison_config, "comparison config")
    rng = np.random.default_rng(args.seed)
    notes: list[str] = []

    if frame.empty:
        method_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        recovery_rows: list[dict[str, Any]] = []
        credit_rows: list[dict[str, Any]] = []
        notes.append("Canonical dataset is empty; all numerical analyses remain pending.")
        status = "pending"
    else:
        method_rows = method_summaries(
            frame,
            rng=rng,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            minimum_runs=args.minimum_runs,
        )
        comparison_rows, comparison_notes = comparison_summaries(
            frame,
            config.get("comparisons", []),
            rng=rng,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            minimum_runs=args.minimum_runs,
        )
        recovery_rows = recovery_summaries(frame, args.confidence)
        credit_rows, credit_notes = credit_summaries(
            frame, list(config.get("protected_pass_columns", [])), args.confidence
        )
        notes.extend(comparison_notes)
        notes.extend(credit_notes)
        status = "complete"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {
        "method_summaries": args.output_dir / "method_summaries.csv",
        "paired_comparisons": args.output_dir / "paired_comparisons.csv",
        "recovery_summaries": args.output_dir / "recovery_summaries.csv",
        "positive_credit_summaries": args.output_dir / "positive_credit_summaries.csv",
    }
    write_records_csv(
        method_rows,
        csv_paths["method_summaries"],
        [*GROUP_COLUMNS, "metric", "mean", "std_across_runs", "n_scenes", "n_runs"],
    )
    write_records_csv(
        comparison_rows,
        csv_paths["paired_comparisons"],
        [
            "comparison_id",
            "label_a",
            "label_b",
            "metric",
            "mean_a",
            "mean_b",
            "absolute_difference_b_minus_a",
            "ci_low_b_minus_a",
            "ci_high_b_minus_a",
            "n_scenes",
            "n_runs",
            "status",
        ],
    )
    write_records_csv(
        recovery_rows,
        csv_paths["recovery_summaries"],
        [
            *GROUP_COLUMNS,
            "recovered_initial_failures",
            "persistent_failures",
            "retained_successes",
            "new_failures",
            "net_recovery",
            "recovery_rate",
            "recovery_ci_low",
            "recovery_ci_high",
            "n_scenes",
            "n_runs",
        ],
    )
    write_records_csv(
        credit_rows,
        csv_paths["positive_credit_summaries"],
        [
            *GROUP_COLUMNS,
            "positive_credit_rollouts",
            "protected_condition_violations",
            "positive_credit_violation_rate",
            "violation_ci_low",
            "violation_ci_high",
            "n_scenes",
            "n_runs",
        ],
    )

    payload = json_safe(
        {
            "schema_version": 1,
            "status": status,
            "analysis_seed": args.seed,
            "analysis_seed_scope": "bootstrap resampling only; not a training or evaluation seed",
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": args.confidence,
            "minimum_runs_for_stability_claim": args.minimum_runs,
            "input": relative_display(args.input, repository_root()),
            "input_sha256": sha256_file(args.input),
            "method_summaries": method_rows,
            "paired_comparisons": comparison_rows,
            "recovery_summaries": recovery_rows,
            "positive_credit_summaries": credit_rows,
            "notes": notes,
        }
    )
    atomic_write_json(args.output_json, payload)
    outputs = [args.output_json, *csv_paths.values()]
    provenance = build_provenance(
        seed=args.seed,
        inputs=[args.input, args.validation_status, args.comparison_config],
        outputs=outputs,
        status=status,
        notes=notes,
    )
    atomic_write_json(args.provenance, provenance)
    logging.info(
        "statistics status=%s, method rows=%d, comparisons=%d",
        status,
        len(method_rows),
        len(comparison_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
