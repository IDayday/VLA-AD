#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


def _read_eval_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token" not in df.columns:
        raise KeyError(f"Missing token column in {path}")
    return df[df["token"].astype(str) != "average"].copy()


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _markdown_table(df: pd.DataFrame, *, float_digits: int = 4, max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_empty_\n"
    view = df.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}")
    columns = [str(column) for column in view.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("\n", " ")

    rows = [[fmt(value) for value in row] for row in view.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(columns[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body]) + "\n"


def _case_from_scores(train_score: float, test_score: float) -> str:
    if pd.isna(train_score) or pd.isna(test_score):
        return "unknown"
    if train_score > 0.95 and test_score > 0.95:
        return "train_high_test_high"
    if train_score > 0.95 and test_score < 0.85:
        return "train_high_test_low"
    if train_score < 0.90 and test_score < 0.85:
        return "train_low_test_low"
    if train_score < 0.90 and test_score > 0.95:
        return "train_low_test_high"
    if test_score < 0.85:
        return "train_mid_test_low"
    return "other"


def _safe_corr(df: pd.DataFrame, left: str, right: str, method: str = "pearson") -> float:
    valid = df[[left, right]].dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid.corr(method=method).iloc[0, 1])


def _parse_scores_json(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    scores: list[float] = []
    for item in raw:
        if item is None:
            continue
        try:
            score = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scores.append(score)
    return scores


def _sensitivity_class(row: pd.Series) -> str:
    if row["candidate_valid_count"] <= 0:
        return "invalid"
    score_min = row["sample_min"]
    score_max = row["sample_max"]
    score_range = row["sample_range"]
    if pd.isna(score_min) or pd.isna(score_max):
        return "invalid"
    if score_min <= 0.05 and score_max >= 0.85:
        return "knife_edge_zero_to_success"
    if score_range >= 0.50:
        return "large_range"
    if score_max < 0.85:
        return "stable_low"
    if score_min >= 0.95:
        return "stable_high"
    return "moderate_range"


def analyze_sensitivity(best_of_n_csv: Path, metadata_csv: Optional[Path], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_eval_csv(best_of_n_csv)
    if "candidate_scores_json" not in df.columns:
        raise KeyError(f"Missing candidate_scores_json in {best_of_n_csv}")
    score_lists = df["candidate_scores_json"].map(_parse_scores_json)
    df["sample_min"] = score_lists.map(lambda xs: min(xs) if xs else float("nan"))
    df["sample_max"] = score_lists.map(lambda xs: max(xs) if xs else float("nan"))
    df["sample_mean"] = score_lists.map(lambda xs: sum(xs) / len(xs) if xs else float("nan"))
    df["sample_range"] = df["sample_max"] - df["sample_min"]
    df["sample_zero_rate"] = score_lists.map(lambda xs: sum(score <= 0.05 for score in xs) / len(xs) if xs else float("nan"))
    df["sample_success_rate"] = score_lists.map(lambda xs: sum(score >= 0.85 for score in xs) / len(xs) if xs else float("nan"))
    df["sensitivity_class"] = df.apply(_sensitivity_class, axis=1)

    if metadata_csv is not None and metadata_csv.is_file():
        metadata = pd.read_csv(metadata_csv)
        if "token" not in metadata.columns and "navtest_token" in metadata.columns:
            metadata = metadata.rename(columns={"navtest_token": "token"})
        df = df.merge(metadata, on="token", how="left", suffixes=("", "_meta"))

    df.to_csv(output_dir / "best_of_n_sensitivity_per_scene.csv", index=False)
    group_cols = [column for column in ("source_case", "failure_reason", "motion_phenotype") if column in df.columns]
    if group_cols:
        grouped = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n=("token", "count"),
                candidate0=("candidate0_score", "mean"),
                best=("score", "mean"),
                sample_min=("sample_min", "mean"),
                sample_max=("sample_max", "mean"),
                sample_range=("sample_range", "mean"),
                zero_rate=("sample_zero_rate", "mean"),
                success_rate=("sample_success_rate", "mean"),
            )
            .reset_index()
        )
    else:
        grouped = pd.DataFrame(
            [
                {
                    "n": len(df),
                    "candidate0": pd.to_numeric(df.get("candidate0_score"), errors="coerce").mean(),
                    "best": pd.to_numeric(df.get("score"), errors="coerce").mean(),
                    "sample_range": df["sample_range"].mean(),
                    "zero_rate": df["sample_zero_rate"].mean(),
                    "success_rate": df["sample_success_rate"].mean(),
                }
            ]
        )
    class_summary = (
        df.groupby("sensitivity_class")
        .agg(
            n=("token", "count"),
            candidate0=("candidate0_score", "mean"),
            best=("score", "mean"),
            sample_range=("sample_range", "mean"),
            zero_rate=("sample_zero_rate", "mean"),
            success_rate=("sample_success_rate", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    grouped.to_csv(output_dir / "best_of_n_sensitivity_group_summary.csv", index=False)
    class_summary.to_csv(output_dir / "best_of_n_sensitivity_class_summary.csv", index=False)
    return grouped, class_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ReCogDrive train/test failure modes and optional best-of-N sensitivity.")
    parser.add_argument("--train-eval-csv", type=Path, required=True)
    parser.add_argument("--train-map-csv", type=Path, required=True)
    parser.add_argument("--navtest-long-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--best-of-n-csv", type=Path)
    parser.add_argument("--best-of-n-metadata-csv", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_eval = _read_eval_csv(args.train_eval_csv)
    train_map = pd.read_csv(args.train_map_csv)
    if "train_token" not in train_map.columns:
        raise KeyError(f"Missing train_token in {args.train_map_csv}")

    _to_numeric(
        train_eval,
        [
            "score",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ],
    )
    if "comfort" in train_eval.columns:
        train_eval["train_core"] = (
            5 * train_eval["ego_progress"] + 5 * train_eval["time_to_collision_within_bound"] + 2 * train_eval["comfort"]
        ) / 12.0

    joined = train_eval.merge(train_map, left_on="token", right_on="train_token", how="left", suffixes=("_train", "_linked"))
    _to_numeric(
        joined,
        [
            "mean_sota_score",
            "zero_ratio",
            "mean_similarity",
            "linked_navtest_count",
            "mean_core",
            "mean_ep",
            "mean_ttc",
            "mean_nc",
            "mean_dac",
        ],
    )
    joined["train_test_case"] = [
        _case_from_scores(train_score, test_score)
        for train_score, test_score in zip(joined["score"], joined["mean_sota_score"])
    ]
    joined.to_csv(args.output_dir / "train_eval_joined_with_navtest_mapping.csv", index=False)

    group_summary = (
        joined.groupby("test_outcome_proxy", dropna=False)
        .agg(
            n=("token", "count"),
            train_pdms=("score", "mean"),
            train_zero=("score", lambda s: float((s == 0).mean())),
            train_core=("train_core", "mean") if "train_core" in joined.columns else ("score", "mean"),
            train_nc=("no_at_fault_collisions", "mean"),
            train_dac=("drivable_area_compliance", "mean"),
            train_ep=("ego_progress", "mean"),
            train_ttc=("time_to_collision_within_bound", "mean"),
            train_ddc=("driving_direction_compliance", "mean"),
            linked_navtest=("mean_sota_score", "mean"),
            linked_zero=("zero_ratio", "mean"),
            linked_count=("linked_navtest_count", "mean"),
            similarity=("mean_similarity", "mean"),
        )
        .reset_index()
    )
    case_summary = (
        joined.groupby("train_test_case")
        .agg(
            n=("token", "count"),
            train_pdms=("score", "mean"),
            train_zero=("score", lambda s: float((s == 0).mean())),
            train_ep=("ego_progress", "mean"),
            train_dac=("drivable_area_compliance", "mean"),
            linked_navtest=("mean_sota_score", "mean"),
            linked_zero=("zero_ratio", "mean"),
            similarity=("mean_similarity", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    thresholds = []
    for threshold in (0.85, 0.90, 0.95):
        train_low = joined["score"] < threshold
        test_low = joined["mean_sota_score"] < 0.85
        thresholds.append(
            {
                "train_pdms_threshold": threshold,
                "pct_train_below": float(train_low.mean()),
                "pct_test_low_given_train_below": float(test_low[train_low].mean()) if train_low.any() else float("nan"),
                "pct_test_low_given_train_above": float(test_low[~train_low].mean()) if (~train_low).any() else float("nan"),
            }
        )
    threshold_summary = pd.DataFrame(thresholds)

    correlation = pd.DataFrame(
        [
            {
                "metric": "train_pdms_vs_linked_navtest_pdms",
                "pearson": _safe_corr(joined, "score", "mean_sota_score", "pearson"),
                "spearman": _safe_corr(joined, "score", "mean_sota_score", "spearman"),
            },
            {
                "metric": "train_pdms_vs_linked_zero_ratio",
                "pearson": _safe_corr(joined, "score", "zero_ratio", "pearson"),
                "spearman": _safe_corr(joined, "score", "zero_ratio", "spearman"),
            },
        ]
    )

    group_summary.to_csv(args.output_dir / "group_summary.csv", index=False)
    case_summary.to_csv(args.output_dir / "train_test_case_summary.csv", index=False)
    threshold_summary.to_csv(args.output_dir / "threshold_summary.csv", index=False)
    correlation.to_csv(args.output_dir / "correlation_summary.csv", index=False)

    navtest_long = pd.read_csv(
        args.navtest_long_csv,
        usecols=[
            "train_token",
            "navtest_token",
            "sota_score",
            "failure_reason",
            "motion_phenotype",
            "similarity",
            "rank",
            "sota_ep",
            "sota_ttc",
            "sota_nc",
            "sota_dac",
            "sota_ddc",
        ],
    )
    train_scores = joined[
        [
            "token",
            "test_outcome_proxy",
            "score",
            "ego_progress",
            "time_to_collision_within_bound",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
        ]
    ].rename(
        columns={
            "token": "train_token",
            "score": "train_score",
            "ego_progress": "train_ep",
            "time_to_collision_within_bound": "train_ttc",
            "no_at_fault_collisions": "train_nc",
            "drivable_area_compliance": "train_dac",
            "driving_direction_compliance": "train_ddc",
        }
    )
    linked_rows = navtest_long.merge(train_scores, on="train_token", how="inner")
    linked_rows["source_case"] = [
        _case_from_scores(train_score, test_score)
        for train_score, test_score in zip(linked_rows["train_score"], linked_rows["sota_score"])
    ]
    linked_rows.to_csv(args.output_dir / "linked_navtest_rows_with_train_scores.csv", index=False)
    linked_case_summary = (
        linked_rows[linked_rows["source_case"] != "other"]
        .groupby("source_case")
        .agg(
            linked_rows=("navtest_token", "count"),
            unique_train_tokens=("train_token", "nunique"),
            navtest_pdms=("sota_score", "mean"),
            navtest_zero=("sota_score", lambda s: float((s == 0).mean())),
            similarity=("similarity", "mean"),
            train_pdms=("train_score", "mean"),
            train_ep=("train_ep", "mean"),
        )
        .reset_index()
        .sort_values("linked_rows", ascending=False)
    )
    failure_dist = (
        linked_rows[linked_rows["source_case"] != "other"]
        .groupby(["source_case", "failure_reason"])
        .agg(n=("navtest_token", "count"), score=("sota_score", "mean"), similarity=("similarity", "mean"))
        .reset_index()
    )
    failure_dist["ratio"] = failure_dist.groupby("source_case")["n"].transform(lambda s: s / s.sum())
    failure_dist = failure_dist.sort_values(["source_case", "n"], ascending=[True, False])
    motion_dist = (
        linked_rows[linked_rows["source_case"] != "other"]
        .groupby(["source_case", "motion_phenotype"])
        .agg(n=("navtest_token", "count"), score=("sota_score", "mean"), similarity=("similarity", "mean"))
        .reset_index()
    )
    motion_dist["ratio"] = motion_dist.groupby("source_case")["n"].transform(lambda s: s / s.sum())
    motion_dist = motion_dist.sort_values(["source_case", "n"], ascending=[True, False])
    linked_case_summary.to_csv(args.output_dir / "linked_case_summary.csv", index=False)
    failure_dist.to_csv(args.output_dir / "linked_case_failure_distribution.csv", index=False)
    motion_dist.to_csv(args.output_dir / "linked_case_motion_distribution.csv", index=False)

    examples_cols = [
        "token",
        "test_outcome_proxy",
        "score",
        "ego_progress",
        "time_to_collision_within_bound",
        "drivable_area_compliance",
        "no_at_fault_collisions",
        "driving_direction_compliance",
        "mean_sota_score",
        "zero_ratio",
        "linked_navtest_count",
        "mean_similarity",
        "failure_mix",
        "motion_mix",
    ]
    for case_name in ("train_high_test_low", "train_low_test_low", "train_high_test_high"):
        examples = joined[joined["train_test_case"] == case_name].copy()
        examples = examples.sort_values(["mean_sota_score", "zero_ratio"], ascending=[True, False])
        examples[[column for column in examples_cols if column in examples.columns]].head(200).to_csv(
            args.output_dir / f"examples_{case_name}.csv", index=False
        )

    sensitivity_group = pd.DataFrame()
    sensitivity_class = pd.DataFrame()
    if args.best_of_n_csv is not None:
        sensitivity_group, sensitivity_class = analyze_sensitivity(args.best_of_n_csv, args.best_of_n_metadata_csv, args.output_dir)

    report_lines = [
        "# ReCogDrive Train/Test Failure Mode Analysis",
        "",
        "## Inputs",
        "",
        f"- train eval: `{args.train_eval_csv}`",
        f"- train/navtest map: `{args.train_map_csv}`",
        f"- navtest top-k long table: `{args.navtest_long_csv}`",
        f"- output dir: `{args.output_dir}`",
        "",
        "## Group Summary",
        "",
        _markdown_table(group_summary),
        "## Case Summary",
        "",
        _markdown_table(case_summary),
        "## Correlation",
        "",
        _markdown_table(correlation),
        "## Threshold Relation",
        "",
        _markdown_table(threshold_summary),
        "## Linked Navtest Case Summary",
        "",
        _markdown_table(linked_case_summary),
        "## Failure Distribution",
        "",
        _markdown_table(failure_dist, max_rows=30),
        "## Motion Distribution",
        "",
        _markdown_table(motion_dist, max_rows=30),
    ]
    if not sensitivity_group.empty:
        report_lines.extend(
            [
                "## Best-of-N Sensitivity",
                "",
                "Sensitivity classes are computed from per-scene candidate score min/max over repeated diffusion sampling.",
                "",
                _markdown_table(sensitivity_class),
                "### Sensitivity By Source Case / Failure / Motion",
                "",
                _markdown_table(sensitivity_group, max_rows=40),
            ]
        )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
