#!/usr/bin/env python3
"""Validate the canonical scene-level supplementary dataset.

Hard data-integrity failures return exit code 2 after writing a diagnostic
report.  A genuinely empty dataset is ``pending`` rather than invalid, enabling
the pipeline to expose missing evidence without manufacturing rows.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    BOOLEAN_COLUMNS,
    CANONICAL_SCHEMA,
    METRIC_COLUMNS,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    configure_logging,
    markdown_table,
    read_table,
    relative_display,
    repository_root,
    sha256_file,
    supplementary_root,
)


IDENTITY_COLUMNS = [
    "benchmark",
    "split",
    "method",
    "stage",
    "variant",
    "seed",
    "round",
    "sample_index",
    "scene_id",
]


@dataclass
class Finding:
    severity: str
    check: str
    message: str
    count: int | None = None


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Validate canonical scene metrics and stop on integrity errors."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=supp / "derived/canonical_scene_metrics.parquet",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=supp / "configs/validation.json",
        help="JSON configuration for formulas, expected counts, and matched sets.",
    )
    parser.add_argument(
        "--claims-csv",
        type=Path,
        default=supp / "audit/result_source_map.csv",
        help="Optional traceable paper claims to compare with scene means.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=supp / "derived/data_validation_report.md",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=supp / "derived/data_validation_status.json",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=supp / "derived/validation_provenance.json",
    )
    parser.add_argument("--metric-min", type=float, default=0.0)
    parser.add_argument("--metric-max", type=float, default=100.0)
    parser.add_argument("--aggregate-tolerance", type=float, default=1e-5)
    parser.add_argument("--claim-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--claim-mismatch-error",
        action="store_true",
        help="Treat paper-claim/scene-mean mismatches as errors rather than warnings.",
    )
    parser.add_argument(
        "--fail-on-warning", action="store_true", help="Return exit code 2 for warnings too."
    )
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], list[Finding]]:
    if not path.exists():
        return {}, [Finding("warning", "validation_config", f"Config not found: {path.name}")]
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("validation config root must be a JSON object")
    return config, []


def selector_mask(frame: pd.DataFrame, selector: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, expected in selector.items():
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
        if isinstance(expected, list):
            mask &= frame[column].isin(expected)
        elif expected is None:
            mask &= frame[column].isna()
        else:
            mask &= frame[column].astype(str).eq(str(expected))
    return mask


def check_schema(frame: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    missing = [column for column in CANONICAL_SCHEMA if column not in frame]
    if missing:
        findings.append(
            Finding("error", "schema", f"Missing canonical columns: {', '.join(missing)}", len(missing))
        )
    else:
        findings.append(Finding("pass", "schema", "All required canonical columns are present."))
    return findings


def check_identifiers_and_duplicates(frame: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    missing_id = frame["scene_id"].isna() | frame["scene_id"].astype(str).str.strip().eq("")
    if missing_id.any():
        findings.append(
            Finding("error", "scene_id", "Rows have missing scene identifiers.", int(missing_id.sum()))
        )
    else:
        findings.append(Finding("pass", "scene_id", "Every row has an explicit scene identifier."))

    available_keys = [column for column in IDENTITY_COLUMNS if column in frame]
    duplicate = frame.duplicated(available_keys, keep=False)
    if duplicate.any():
        findings.append(
            Finding(
                "error",
                "duplicates",
                "Duplicate records share the complete canonical identity key.",
                int(duplicate.sum()),
            )
        )
    else:
        findings.append(Finding("pass", "duplicates", "No duplicate canonical identity keys."))
    return findings


def check_metric_ranges(frame: pd.DataFrame, minimum: float, maximum: float) -> list[Finding]:
    findings: list[Finding] = []
    for column in METRIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_numeric = frame[column].notna() & values.isna()
        outside = values.notna() & ((values < minimum) | (values > maximum))
        if invalid_numeric.any():
            findings.append(
                Finding("error", f"range:{column}", "Non-numeric metric values found.", int(invalid_numeric.sum()))
            )
        elif outside.any():
            observed_min = float(values.min())
            observed_max = float(values.max())
            findings.append(
                Finding(
                    "error",
                    f"range:{column}",
                    f"Values fall outside [{minimum}, {maximum}] (observed {observed_min:g}--{observed_max:g}).",
                    int(outside.sum()),
                )
            )
    if not any(finding.check.startswith("range:") for finding in findings):
        findings.append(
            Finding("pass", "metric_ranges", f"All observed metrics lie in [{minimum}, {maximum}].")
        )
    return findings


def check_boolean_logic(frame: pd.DataFrame, tolerance: float) -> list[Finding]:
    findings: list[Finding] = []

    def error_if(mask: pd.Series, check: str, message: str) -> None:
        mask = mask.fillna(False)
        if mask.any():
            findings.append(Finding("error", check, message, int(mask.sum())))

    score = pd.to_numeric(frame["aggregate_score"], errors="coerce")
    error_if(
        frame["zero_score"].eq(True) & score.notna() & (score.abs() > tolerance),
        "zero_score_logic",
        "zero_score=True but aggregate_score is nonzero.",
    )
    error_if(
        frame["zero_score"].eq(False) & score.notna() & (score.abs() <= tolerance),
        "zero_score_logic",
        "zero_score=False but aggregate_score is zero.",
    )
    error_if(
        frame["recovered"].eq(True) & ~frame["initial_failure"].eq(True),
        "recovery_logic",
        "recovered=True requires initial_failure=True.",
    )
    error_if(
        frame["new_failure"].eq(True) & frame["initial_failure"].eq(True),
        "recovery_logic",
        "new_failure=True is incompatible with initial_failure=True.",
    )
    error_if(
        frame["recovered"].eq(True) & frame["feasible"].eq(False),
        "recovery_logic",
        "recovered=True is incompatible with feasible=False.",
    )
    error_if(
        frame["new_failure"].eq(True) & frame["feasible"].eq(True),
        "recovery_logic",
        "new_failure=True is incompatible with feasible=True.",
    )
    error_if(
        frame["positive_credit"].eq(True)
        & frame["advantage"].notna()
        & (pd.to_numeric(frame["advantage"], errors="coerce") <= 0),
        "positive_credit_logic",
        "positive_credit=True but advantage is not positive.",
    )
    if not findings:
        findings.append(Finding("pass", "state_logic", "Observed state and credit flags are internally consistent."))
    return findings


def check_aggregate_formulas(
    frame: pd.DataFrame, specs: list[dict[str, Any]], default_tolerance: float
) -> list[Finding]:
    if not specs:
        return [
            Finding(
                "warning",
                "aggregate_formula",
                "No benchmark-specific aggregate formula is configured; formula consistency was not evaluated.",
            )
        ]
    findings: list[Finding] = []
    for index, spec in enumerate(specs):
        name = str(spec.get("name", f"formula_{index}"))
        expression = spec.get("expression")
        selector = spec.get("selector", {})
        tolerance = float(spec.get("tolerance", default_tolerance))
        if not isinstance(expression, str) or not expression.strip():
            findings.append(Finding("error", f"aggregate_formula:{name}", "Missing expression."))
            continue
        subset = frame.loc[selector_mask(frame, selector)].copy()
        if subset.empty:
            findings.append(
                Finding("warning", f"aggregate_formula:{name}", "No rows matched the formula selector.")
            )
            continue
        try:
            calculated = subset.eval(expression, engine="python")
        except Exception as exc:
            findings.append(
                Finding("error", f"aggregate_formula:{name}", f"Expression failed: {type(exc).__name__}: {exc}")
            )
            continue
        observed = pd.to_numeric(subset["aggregate_score"], errors="coerce")
        comparable = observed.notna() & calculated.notna()
        mismatch = comparable & ((observed - calculated).abs() > tolerance)
        if mismatch.any():
            findings.append(
                Finding(
                    "error",
                    f"aggregate_formula:{name}",
                    f"Aggregate score differs from configured formula by more than {tolerance:g}.",
                    int(mismatch.sum()),
                )
            )
        else:
            findings.append(
                Finding(
                    "pass",
                    f"aggregate_formula:{name}",
                    f"Formula matches {int(comparable.sum())} comparable rows.",
                )
            )
    return findings


def check_expected_counts(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> list[Finding]:
    if not specs:
        return [
            Finding(
                "warning",
                "expected_scene_counts",
                "No evidence-backed expected scene counts are configured.",
            )
        ]
    findings: list[Finding] = []
    for index, spec in enumerate(specs):
        selector = spec.get("selector", {})
        expected = spec.get("expected_unique_scenes")
        name = str(spec.get("name", f"count_{index}"))
        if not isinstance(expected, int) or expected < 0:
            findings.append(Finding("error", f"scene_count:{name}", "Expected count must be non-negative integer."))
            continue
        observed = int(frame.loc[selector_mask(frame, selector), "scene_id"].nunique())
        if observed != expected:
            findings.append(
                Finding(
                    "error",
                    f"scene_count:{name}",
                    f"Expected {expected} unique scenes, observed {observed}.",
                    abs(observed - expected),
                )
            )
        else:
            findings.append(Finding("pass", f"scene_count:{name}", f"Observed expected {expected} scenes."))
    return findings


def check_scene_sets(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> list[Finding]:
    if not specs:
        return [
            Finding(
                "warning",
                "matched_scene_sets",
                "No protocol-specific matched scene-set checks are configured.",
            )
        ]
    findings: list[Finding] = []
    for index, spec in enumerate(specs):
        name = str(spec.get("name", f"set_{index}"))
        selectors = spec.get("selectors", [])
        if not isinstance(selectors, list) or len(selectors) < 2:
            findings.append(Finding("error", f"scene_set:{name}", "At least two selectors are required."))
            continue
        sets = [set(frame.loc[selector_mask(frame, selector), "scene_id"].dropna().astype(str)) for selector in selectors]
        if not sets or any(not values for values in sets):
            findings.append(Finding("error", f"scene_set:{name}", "One or more selectors match no scenes."))
            continue
        reference = sets[0]
        deltas = [len(reference.symmetric_difference(other)) for other in sets[1:]]
        if any(deltas):
            findings.append(
                Finding(
                    "error",
                    f"scene_set:{name}",
                    f"Scene sets differ; symmetric-difference sizes relative to first: {deltas}.",
                    sum(deltas),
                )
            )
        else:
            findings.append(Finding("pass", f"scene_set:{name}", f"All selectors share {len(reference)} scenes."))
    return findings


def check_candidate_matching(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> list[Finding]:
    if not specs:
        return [
            Finding(
                "warning",
                "candidate_matching",
                "No candidate-count/acceptance-rate matching checks are configured.",
            )
        ]
    findings: list[Finding] = []
    for index, spec in enumerate(specs):
        name = str(spec.get("name", f"candidate_match_{index}"))
        column = str(spec.get("column", "retained_candidates"))
        selectors = spec.get("selectors", [])
        tolerance = float(spec.get("tolerance", 0.0))
        if column not in frame:
            findings.append(Finding("error", f"candidate_match:{name}", f"Column {column!r} is absent."))
            continue
        means = []
        for selector in selectors:
            values = pd.to_numeric(frame.loc[selector_mask(frame, selector), column], errors="coerce").dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
        if len(means) < 2 or any(np.isnan(value) for value in means):
            findings.append(Finding("error", f"candidate_match:{name}", "Selectors lack comparable candidate counts."))
        elif max(means) - min(means) > tolerance:
            findings.append(
                Finding(
                    "error",
                    f"candidate_match:{name}",
                    f"Mean {column} values {means} exceed tolerance {tolerance:g}.",
                )
            )
        else:
            findings.append(Finding("pass", f"candidate_match:{name}", f"Mean values {means} are matched."))
    return findings


def check_claim_means(
    frame: pd.DataFrame, claims_path: Path, tolerance: float, mismatch_is_error: bool
) -> list[Finding]:
    if not claims_path.exists():
        return [Finding("warning", "paper_claim_means", "Traceable result-source map is not available yet.")]
    try:
        claims = pd.read_csv(claims_path)
    except Exception as exc:
        return [Finding("error", "paper_claim_means", f"Could not read claims CSV: {exc}")]
    required = {"method", "metric", "value"}
    if not required.issubset(claims.columns):
        return [
            Finding(
                "warning",
                "paper_claim_means",
                "Result-source map lacks method/metric/value columns required for numeric comparison.",
            )
        ]
    findings: list[Finding] = []
    compared = 0
    for row in claims.to_dict(orient="records"):
        method, metric = row.get("method"), row.get("metric")
        try:
            expected = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if metric not in frame.columns or pd.isna(method):
            continue
        selector: dict[str, Any] = {"method": method}
        for column in ("split", "benchmark", "seed", "stage", "variant", "round"):
            value = row.get(column)
            if column in frame and value is not None and not pd.isna(value) and str(value).strip():
                selector[column] = value
        values = pd.to_numeric(frame.loc[selector_mask(frame, selector), metric], errors="coerce").dropna()
        if values.empty:
            continue
        compared += 1
        observed = float(values.mean())
        if abs(observed - expected) > tolerance:
            severity = "error" if mismatch_is_error else "warning"
            source = row.get("source_file", "unknown source")
            findings.append(
                Finding(
                    severity,
                    "paper_claim_means",
                    f"{method}/{metric}: scene mean {observed:.6g} vs claim {expected:.6g} from {source}.",
                )
            )
    if compared == 0:
        findings.append(
            Finding("warning", "paper_claim_means", "No traceable claim row matched the canonical data.")
        )
    elif not findings:
        findings.append(
            Finding("pass", "paper_claim_means", f"All {compared} matched claims agree within {tolerance:g}.")
        )
    return findings


def coverage_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["benchmark", "split", "method", "stage", "variant"]
    coverage = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            rows=("scene_id", "size"),
            unique_scenes=("scene_id", "nunique"),
            seeds=("seed", lambda values: int(values.dropna().nunique())),
            rounds=("round", lambda values: int(values.dropna().nunique())),
        )
        .reset_index()
    )
    scene_counts = (
        frame.groupby(group_columns + ["seed", "round"], dropna=False)["scene_id"]
        .nunique()
        .rename("unique_scenes")
        .reset_index()
    )
    return coverage, scene_counts


def render_report(
    status: str,
    frame: pd.DataFrame,
    findings: list[Finding],
    coverage: pd.DataFrame,
    scene_counts: pd.DataFrame,
    input_hash: str,
) -> str:
    counts = Counter(finding.severity for finding in findings)
    lines = [
        "# Data Validation Report",
        "",
        f"**Status:** `{status}`",
        "",
        f"- Canonical rows: {len(frame)}",
        f"- Unique scenes: {frame['scene_id'].nunique() if 'scene_id' in frame else 0}",
        f"- Errors: {counts.get('error', 0)}",
        f"- Warnings: {counts.get('warning', 0)}",
        f"- Input SHA-256: `{input_hash}`",
        "",
    ]
    if status == "pending":
        lines.append(
            "No scene-level evidence has been ingested. Integrity checks that require observations remain pending."
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            markdown_table(
                ["Severity", "Check", "Count", "Message"],
                [
                    (finding.severity, finding.check, finding.count or "", finding.message)
                    for finding in findings
                ],
            ),
            "",
            "## Seed and round coverage",
            "",
            markdown_table(list(coverage.columns), coverage.itertuples(index=False, name=None))
            if not coverage.empty
            else "No records available.",
            "",
            "## Per-run scene counts",
            "",
            markdown_table(list(scene_counts.columns), scene_counts.itertuples(index=False, name=None))
            if not scene_counts.empty
            else "No records available.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.metric_min >= args.metric_max:
        raise ValueError("--metric-min must be less than --metric-max")
    if args.aggregate_tolerance < 0 or args.claim_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if not args.input.exists():
        raise FileNotFoundError(f"canonical dataset not found: {args.input}")

    frame = read_table(args.input)
    findings = check_schema(frame)
    config, config_findings = load_config(args.validation_config)
    findings.extend(config_findings)
    if not set(CANONICAL_SCHEMA).issubset(frame.columns):
        status = "invalid"
        coverage = pd.DataFrame()
        scene_counts = pd.DataFrame()
    elif frame.empty:
        status = "pending"
        coverage = pd.DataFrame()
        scene_counts = pd.DataFrame()
        findings.append(
            Finding("warning", "data_availability", "Canonical dataset contains zero records.")
        )
    else:
        findings.extend(check_identifiers_and_duplicates(frame))
        findings.extend(check_metric_ranges(frame, args.metric_min, args.metric_max))
        findings.extend(check_boolean_logic(frame, args.aggregate_tolerance))
        findings.extend(
            check_aggregate_formulas(
                frame, config.get("aggregate_formulas", []), args.aggregate_tolerance
            )
        )
        findings.extend(check_expected_counts(frame, config.get("expected_scene_counts", [])))
        findings.extend(check_scene_sets(frame, config.get("matched_scene_sets", [])))
        findings.extend(check_candidate_matching(frame, config.get("candidate_match_checks", [])))
        findings.extend(
            check_claim_means(
                frame, args.claims_csv, args.claim_tolerance, args.claim_mismatch_error
            )
        )
        coverage, scene_counts = coverage_tables(frame)
        error_count = sum(finding.severity == "error" for finding in findings)
        status = "invalid" if error_count else "valid"

    input_hash = sha256_file(args.input)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.report,
        render_report(status, frame, findings, coverage, scene_counts, input_hash),
    )
    status_payload = {
        "schema_version": 1,
        "status": status,
        "input": relative_display(args.input, repository_root()),
        "input_sha256": input_hash,
        "rows": int(len(frame)),
        "unique_scenes": int(frame["scene_id"].nunique()) if "scene_id" in frame else 0,
        "finding_counts": dict(Counter(finding.severity for finding in findings)),
        "findings": [asdict(finding) for finding in findings],
    }
    atomic_write_json(args.status_json, status_payload)
    provenance_inputs = [args.input]
    if args.validation_config.exists():
        provenance_inputs.append(args.validation_config)
    if args.claims_csv.exists():
        provenance_inputs.append(args.claims_csv)
    provenance = build_provenance(
        seed=args.seed,
        inputs=provenance_inputs,
        outputs=[args.report, args.status_json],
        status=status,
        notes=[finding.message for finding in findings if finding.severity != "pass"],
    )
    atomic_write_json(args.provenance, provenance)
    logging.info("validation status: %s", status)

    has_errors = any(finding.severity == "error" for finding in findings)
    has_warnings = any(finding.severity == "warning" for finding in findings)
    return 2 if has_errors or (args.fail_on_warning and has_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
