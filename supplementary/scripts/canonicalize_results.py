#!/usr/bin/env python3
"""Convert verified scene-level evaluator exports to the canonical long table.

This command never synthesizes scene identifiers, experimental seeds, metrics,
or feasibility labels.  Missing raw inputs produce a typed zero-row CSV and
Parquet file plus an explicit ``pending`` report so downstream code can run
without mistaking absent evidence for a result.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_common import (
    ANALYSIS_SEED,
    CANONICAL_SCHEMA,
    atomic_write_json,
    atomic_write_text,
    build_provenance,
    coerce_canonical_types,
    configure_logging,
    empty_canonical_frame,
    markdown_table,
    read_table,
    relative_display,
    repository_root,
    supplementary_root,
    write_dataframe,
)


COLUMN_ALIASES: dict[str, set[str]] = {
    "scene_id": {"scene_id", "scene_token", "scenario_id", "token"},
    "split": {"split", "dataset_split", "eval_split"},
    "benchmark": {"benchmark", "dataset", "navsim_version"},
    "method": {"method", "model", "policy", "approach"},
    "stage": {"stage", "training_stage"},
    "variant": {"variant", "ablation", "data_variant"},
    "seed": {"seed", "run_seed", "random_seed", "eval_seed"},
    "round": {"round", "apr_round", "iteration"},
    "sample_index": {"sample_index", "sample_idx", "rollout_index", "rollout_idx"},
    "aggregate_score": {
        "aggregate_score",
        "pdms",
        "epdms",
        "pdm_score",
        "epdm_score",
        "navsim_score",
    },
    "NC": {"nc", "no_at_fault_collision", "no_at_fault_collisions"},
    "DAC": {"dac", "drivable_area_compliance"},
    "DDC": {"ddc", "driving_direction_compliance"},
    "TLC": {"tlc", "traffic_light_compliance"},
    "TTC": {"ttc", "time_to_collision", "time_to_collision_within_bound"},
    "EP": {"ep", "ego_progress", "progress"},
    "C": {"c", "comfort", "comfort_score"},
    "LK": {"lk", "lane_keeping", "lane_keeping_score"},
    "HC": {"hc", "history_comfort", "history_comfort_score"},
    "EC": {"ec", "extended_comfort", "extended_comfort_score"},
    "feasible": {"feasible", "is_feasible", "hard_feasibility_pass"},
    "zero_score": {"zero_score", "is_zero_score"},
    "initial_failure": {"initial_failure", "is_initial_failure"},
    "recovered": {"recovered", "is_recovered"},
    "new_failure": {"new_failure", "is_new_failure"},
    "advantage": {"advantage", "trajectory_advantage", "group_advantage"},
    "positive_credit": {"positive_credit", "positive_advantage"},
    "pareto_front": {"pareto_front", "on_pareto_front", "is_pareto"},
    "reference_guard_pass": {
        "reference_guard_pass",
        "coherent_reference_guard_pass",
        "protected_guard_pass",
    },
    "teacher_source": {"teacher_source", "source_teacher"},
    "teacher_status": {"teacher_status", "teacher_lifecycle_status"},
}


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


NORMALIZED_ALIASES = {
    canonical: {normalized(alias) for alias in aliases | {canonical}}
    for canonical, aliases in COLUMN_ALIASES.items()
}


def parse_args() -> argparse.Namespace:
    supp = supplementary_root()
    parser = argparse.ArgumentParser(
        description="Canonicalize explicit scene-level result exports without imputing evidence."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input file or glob; repeatable. Supported: CSV/TSV/JSON/JSONL/Parquet/PKL.",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=supp / "raw_manifest/result_inputs.txt",
        help="Text file with one repository-relative input path/glob per line.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=supp / "raw_manifest/result_sources.json",
        help=(
            "JSON manifest with a 'sources' list. Each enabled source may declare "
            "path, defaults, and column_map; no metadata is inferred from filenames."
        ),
    )
    parser.add_argument(
        "--column-map",
        type=Path,
        help="Optional JSON object mapping raw column names to canonical names.",
    )
    parser.add_argument(
        "--default",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Fill a missing metadata column; never overwrites observed values.",
    )
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading trusted local pickle files (disabled by default).",
    )
    parser.add_argument(
        "--drop-missing-scene-id",
        action="store_true",
        help="Drop rows lacking scene_id; default is to fail on partial missing IDs.",
    )
    parser.add_argument(
        "--skip-unreadable", action="store_true", help="Record unreadable inputs and continue."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=supp / "derived/canonical_scene_metrics.csv",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=supp / "derived/canonical_scene_metrics.parquet",
    )
    parser.add_argument(
        "--report", type=Path, default=supp / "derived/canonicalization_report.md"
    )
    parser.add_argument(
        "--provenance", type=Path, default=supp / "derived/canonicalization_provenance.json"
    )
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_defaults(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--default must have COLUMN=VALUE form: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty default column in {item!r}")
        result[key] = value
    return result


def load_column_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("--column-map must contain a JSON string-to-string object")
    invalid = sorted(set(payload.values()) - set(CANONICAL_SCHEMA))
    if invalid:
        raise ValueError(f"column map contains unknown canonical targets: {invalid}")
    return payload


def load_source_manifest(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"source manifest not found: {relative_display(path, repository_root())}"]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("sources", []), list):
        raise ValueError("source manifest must be an object containing a 'sources' list")
    sources: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, source in enumerate(payload.get("sources", [])):
        if not isinstance(source, dict):
            raise ValueError(f"source manifest entry {index} must be an object")
        if source.get("enabled", True) is False:
            notes.append(f"source manifest entry {index} is disabled")
            continue
        pattern = source.get("path", source.get("glob"))
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"source manifest entry {index} requires a non-empty path or glob")
        defaults = source.get("defaults", {})
        column_map = source.get("column_map", {})
        if not isinstance(defaults, dict):
            raise ValueError(f"source manifest entry {index} defaults must be an object")
        if not isinstance(column_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in column_map.items()
        ):
            raise ValueError(f"source manifest entry {index} column_map must map strings to strings")
        invalid = sorted(set(column_map.values()) - set(CANONICAL_SCHEMA))
        if invalid:
            raise ValueError(f"source manifest entry {index} maps to unknown columns: {invalid}")
        sources.append(
            {
                "id": str(source.get("id", f"source_{index}")),
                "pattern": pattern,
                "defaults": defaults,
                "column_map": column_map,
            }
        )
    return sources, notes


def expand_inputs(
    patterns: list[str], list_path: Path, manifest_path: Path, repo: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_sources = [
        {"id": f"cli_{index}", "pattern": pattern, "defaults": {}, "column_map": {}}
        for index, pattern in enumerate(patterns)
    ]
    notes: list[str] = []
    if list_path.exists():
        for raw_line in list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                raw_sources.append(
                    {
                        "id": f"input_list_{len(raw_sources)}",
                        "pattern": line,
                        "defaults": {},
                        "column_map": {},
                    }
                )
    elif not patterns and not manifest_path.exists():
        notes.append(f"input list not found: {relative_display(list_path, repo)}")

    manifest_sources, manifest_notes = load_source_manifest(manifest_path)
    raw_sources.extend(manifest_sources)
    notes.extend(manifest_notes)

    inputs: list[dict[str, Any]] = []
    seen: dict[Path, dict[str, Any]] = {}
    for source in raw_sources:
        pattern = source["pattern"]
        candidate = Path(pattern)
        expanded_pattern = str(candidate if candidate.is_absolute() else repo / candidate)
        matches = [Path(value).resolve() for value in glob.glob(expanded_pattern, recursive=True)]
        if not matches:
            notes.append(f"input pattern matched no files: {pattern}")
            continue
        for match in matches:
            if match.is_file():
                normalized_source = {**source, "path": match}
                if match in seen:
                    previous = seen[match]
                    comparable_previous = {
                        key: previous[key] for key in ("defaults", "column_map")
                    }
                    comparable_current = {
                        key: normalized_source[key] for key in ("defaults", "column_map")
                    }
                    if comparable_previous != comparable_current:
                        raise ValueError(
                            f"input {relative_display(match, repo)} is declared more than once "
                            "with conflicting defaults or column maps"
                        )
                    notes.append(
                        f"duplicate source declaration ignored: {relative_display(match, repo)}"
                    )
                    continue
                seen[match] = normalized_source
                inputs.append(normalized_source)
            else:
                notes.append(f"input is not a file and was skipped: {pattern}")
    return sorted(inputs, key=lambda item: str(item["path"])), notes


def choose_mapping(columns: list[str], explicit: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    notes: list[str] = []
    occupied: dict[str, str] = {}
    for raw, canonical in explicit.items():
        if raw not in columns:
            notes.append(f"explicit column-map source absent: {raw}")
            continue
        if canonical in occupied:
            raise ValueError(
                f"columns {occupied[canonical]!r} and {raw!r} both map to {canonical!r}"
            )
        mapping[raw] = canonical
        occupied[canonical] = raw

    for raw in columns:
        if raw in mapping:
            continue
        raw_normalized = normalized(raw)
        leaf = normalized(raw.split(".")[-1])
        candidates = [
            canonical
            for canonical, aliases in NORMALIZED_ALIASES.items()
            if raw_normalized in aliases or leaf in aliases
        ]
        candidates = [canonical for canonical in candidates if canonical not in occupied]
        if len(candidates) == 1:
            canonical = candidates[0]
            mapping[raw] = canonical
            occupied[canonical] = raw
        elif len(candidates) > 1:
            raise ValueError(f"ambiguous automatic mapping for column {raw!r}: {candidates}")
    return mapping, notes


def canonicalize_one(
    path: Path,
    *,
    source_id: str,
    repo: Path,
    explicit_map: dict[str, str],
    defaults: dict[str, Any],
    allow_pickle: bool,
    drop_missing_scene_id: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_table(path, allow_pickle=allow_pickle)
    source_report: dict[str, Any] = {
        "source_id": source_id,
        "source_file": relative_display(path, repo),
        "input_rows": int(len(raw)),
        "output_rows": 0,
        "status": "pending",
        "column_mapping": {},
        "notes": [],
    }
    if raw.empty:
        source_report["status"] = "empty_input"
        source_report["notes"].append("The input contained no records.")
        return empty_canonical_frame(), source_report

    raw.columns = [str(column) for column in raw.columns]
    mapping, mapping_notes = choose_mapping(list(raw.columns), explicit_map)
    source_report["column_mapping"] = mapping
    source_report["notes"].extend(mapping_notes)
    renamed = raw.rename(columns=mapping).copy()
    if "scene_id" not in renamed:
        source_report["status"] = "non_scene_level"
        source_report["notes"].append(
            "No explicit scene identifier was found; aggregate-only records were not ingested."
        )
        return empty_canonical_frame(), source_report

    missing_scene = renamed["scene_id"].isna() | renamed["scene_id"].astype(str).str.strip().eq("")
    if missing_scene.any():
        if not drop_missing_scene_id:
            raise ValueError(
                f"{relative_display(path, repo)} has {int(missing_scene.sum())} rows without scene_id"
            )
        source_report["notes"].append(
            f"Dropped {int(missing_scene.sum())} rows without an observed scene_id."
        )
        renamed = renamed.loc[~missing_scene].copy()

    for column, value in defaults.items():
        if column not in renamed:
            renamed[column] = value
        else:
            renamed[column] = renamed[column].fillna(value)

    # Preserve unmapped diagnostics, but make their names deterministic and
    # avoid accidental collisions with canonical columns.
    occupied = set(CANONICAL_SCHEMA)
    clean_names: dict[str, str] = {}
    for column in renamed.columns:
        if column in occupied:
            continue
        base = normalized(column) or "unnamed"
        candidate = base
        index = 2
        while candidate in occupied or candidate in clean_names.values():
            candidate = f"{base}_{index}"
            index += 1
        clean_names[column] = candidate
    renamed = renamed.rename(columns=clean_names)
    renamed["source_file"] = relative_display(path, repo)
    renamed["source_record"] = [str(index) for index in raw.index[~missing_scene]]
    result = coerce_canonical_types(renamed)
    source_report["output_rows"] = int(len(result))
    source_report["status"] = "ingested" if len(result) else "empty_after_filtering"
    return result, source_report


def render_report(
    *, status: str, frame: pd.DataFrame, source_reports: list[dict[str, Any]], notes: list[str]
) -> str:
    lines = [
        "# Canonicalization Report",
        "",
        f"**Status:** `{status}`",
        "",
        f"Canonical scene-level records: **{len(frame)}**.",
        "",
        "No experimental value, seed, scene identifier, or feasibility label is imputed by this script.",
    ]
    if status == "pending":
        lines.extend(
            [
                "",
                "No verified scene-level records were available. Typed empty CSV and Parquet files were emitted so schema checks remain reproducible.",
            ]
        )
    if notes:
        lines.extend(["", "## Input resolution notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Per-source ingestion", ""])
    lines.append(
        markdown_table(
            ["Source", "Input rows", "Output rows", "Status", "Notes"],
            [
                (
                    report["source_file"],
                    report["input_rows"],
                    report["output_rows"],
                    report["status"],
                    "; ".join(report["notes"]),
                )
                for report in source_reports
            ],
        )
        if source_reports
        else "No input files were resolved."
    )
    lines.extend(
        [
            "",
            "## Output schema",
            "",
            markdown_table(
                ["Column", "Pandas dtype"],
                [(column, str(frame[column].dtype)) for column in CANONICAL_SCHEMA],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    repo = repository_root()
    defaults = parse_defaults(args.default)
    explicit_map = load_column_map(args.column_map)
    sources, notes = expand_inputs(args.input, args.input_list, args.source_manifest, repo)
    input_paths = [source["path"] for source in sources]
    logging.info("resolved %d input files", len(sources))

    frames: list[pd.DataFrame] = []
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        path = source["path"]
        source_defaults = {**defaults, **source.get("defaults", {})}
        source_column_map = {**explicit_map, **source.get("column_map", {})}
        try:
            frame, report = canonicalize_one(
                path,
                source_id=source["id"],
                repo=repo,
                explicit_map=source_column_map,
                defaults=source_defaults,
                allow_pickle=args.allow_pickle,
                drop_missing_scene_id=args.drop_missing_scene_id,
            )
            frames.append(frame)
            source_reports.append(report)
        except Exception as exc:
            if not args.skip_unreadable:
                raise
            logging.exception("failed to ingest %s", path)
            source_reports.append(
                {
                    "source_id": source["id"],
                    "source_file": relative_display(path, repo),
                    "input_rows": "unknown",
                    "output_rows": 0,
                    "status": "read_error",
                    "column_mapping": {},
                    "notes": [f"{type(exc).__name__}: {exc}"],
                }
            )

    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        all_columns = sorted(set().union(*(frame.columns for frame in nonempty)))
        extras = [column for column in all_columns if column not in CANONICAL_SCHEMA]
        aligned = [frame.reindex(columns=list(CANONICAL_SCHEMA) + extras) for frame in nonempty]
        canonical = coerce_canonical_types(pd.concat(aligned, ignore_index=True))
    else:
        canonical = empty_canonical_frame()
    status = "complete" if len(canonical) else "pending"

    write_dataframe(canonical, args.output_csv, args.output_parquet)
    atomic_write_text(
        args.report,
        render_report(
            status=status, frame=canonical, source_reports=source_reports, notes=notes
        ),
    )
    provenance = build_provenance(
        seed=args.seed,
        inputs=input_paths
        + ([args.source_manifest] if args.source_manifest.exists() else [])
        + ([args.input_list] if args.input_list.exists() else [])
        + ([args.column_map] if args.column_map else []),
        outputs=[args.output_csv, args.output_parquet, args.report],
        status=status,
        notes=notes,
        root=repo,
    )
    provenance["sources"] = source_reports
    atomic_write_json(args.provenance, provenance)
    logging.info("wrote %d canonical rows (%s)", len(canonical), status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
