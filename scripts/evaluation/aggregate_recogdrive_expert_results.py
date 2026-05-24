#!/usr/bin/env python3
"""Aggregate ReCogDrive expert-token ablation results.

The script accepts NAVSIM PDM result JSON/CSV files or output directories. For
directories, it recursively searches for result-like JSON/CSV files and Hydra
config files, then writes a compact CSV and Markdown comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VARIANT_ORDER = {
    "baseline": 0,
    "jepa": 1,
    "vggt": 2,
    "jepa_vggt": 3,
    "jepa_vggt_alignment": 4,
}

VARIANT_DISPLAY = {
    "baseline": "baseline",
    "jepa": "JEPA only",
    "vggt": "VGGT only",
    "jepa_vggt": "JEPA+VGGT",
    "jepa_vggt_alignment": "JEPA+VGGT+alignment",
}

METRIC_ALIASES = {
    "PDMS": ("pdms", "pdm_score", "score", "final_score"),
    "NC": ("nc", "no_at_fault_collisions", "no_collision", "no_collisions"),
    "DAC": ("dac", "drivable_area_compliance", "drivable_area"),
    "TTC": ("ttc", "time_to_collision_within_bound", "time_to_collision"),
    "CF": ("cf", "comfort", "comfortable"),
    "EP": ("ep", "ego_progress", "progress"),
}

CONFIG_COLUMNS = (
    "use_expert_features",
    "use_jepa",
    "use_vggt",
    "jepa_dim",
    "vggt_dim",
    "num_jepa_tokens",
    "num_vggt_tokens",
    "expert_alignment_weight",
    "jepa_alignment_weight",
    "vggt_alignment_weight",
    "checkpoint_path",
)

CONFIG_ALIASES = {
    "use_expert_features": ("agent.use_expert_features", "use_expert_features"),
    "use_jepa": ("agent.use_jepa", "use_jepa"),
    "use_vggt": ("agent.use_vggt", "use_vggt"),
    "jepa_dim": ("agent.jepa_dim", "jepa_dim"),
    "vggt_dim": ("agent.vggt_dim", "vggt_dim"),
    "num_jepa_tokens": ("agent.num_jepa_tokens", "num_jepa_tokens"),
    "num_vggt_tokens": ("agent.num_vggt_tokens", "num_vggt_tokens"),
    "expert_alignment_weight": ("agent.expert_alignment_weight", "expert_alignment_weight"),
    "jepa_alignment_weight": ("agent.jepa_alignment_weight", "jepa_alignment_weight"),
    "vggt_alignment_weight": ("agent.vggt_alignment_weight", "vggt_alignment_weight"),
    "checkpoint_path": ("agent.checkpoint_path", "checkpoint_path"),
}

RESULT_SUFFIXES = {".json", ".csv"}
CONFIG_NAMES = {"config.yaml", "config.yml", "config.json"}
HYDRA_DIR_NAMES = {"hydra", ".hydra"}


@dataclass
class AggregatedRun:
    label: str
    variant: str
    source: Path
    result_file: Optional[Path] = None
    config_file: Optional[Path] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def as_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null"}:
            return None
        try:
            number = float(stripped)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off", "none", "null", ""}:
            return False
    return None


def deep_get(data: Any, dotted_key: str) -> Any:
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def recursive_find_key(data: Any, key: str) -> Any:
    target = normalize_key(key)
    if isinstance(data, dict):
        for candidate_key, value in data.items():
            if normalize_key(candidate_key) == target:
                return value
        for value in data.values():
            found = recursive_find_key(value, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = recursive_find_key(item, key)
            if found is not None:
                return found
    return None


def get_config_value(config: Dict[str, Any], column: str) -> Any:
    for alias in CONFIG_ALIASES[column]:
        value = deep_get(config, alias) if "." in alias else recursive_find_key(config, alias)
        if value is not None:
            return value
    return None


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except ImportError:
        warn(f"PyYAML is not available; cannot read config YAML: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # pragma: no cover - message matters more than type here
        warn(f"Failed to parse YAML config {path}: {exc}")
        return None

    if isinstance(data, dict):
        return data
    warn(f"Config file {path} did not contain a mapping; ignoring it.")
    return None


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        warn(f"Failed to parse JSON {path}: {exc}")
        return None


def is_config_file(path: Path) -> bool:
    if path.name not in CONFIG_NAMES:
        return False
    parent_names = {parent.name for parent in path.parents}
    return bool(parent_names & HYDRA_DIR_NAMES) or "code" in parent_names or path.name == "config.json"


def load_config(path: Path) -> Optional[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            return data
        warn(f"Config JSON {path} did not contain a mapping; ignoring it.")
        return None
    return load_yaml(path)


def config_from_result_data(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    for key in ("config", "cfg", "training_config", "agent_config"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def flatten_metric_row(data: Any) -> Optional[Dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("average", "summary", "metrics", "pdm", "pdm_score", "results", "pdm_results"):
            value = data.get(key)
            if key in {"results", "pdm_results"} and isinstance(value, list):
                return row_from_records(value)
            if isinstance(value, dict):
                nested = flatten_metric_row(value)
                if nested is not None:
                    return nested

        if any(find_metric_value(data, aliases) is not None for aliases in METRIC_ALIASES.values()):
            return data

        for value in data.values():
            nested = flatten_metric_row(value)
            if nested is not None:
                return nested

    if isinstance(data, list):
        return row_from_records(data)

    return None


def find_metric_value(data: Any, aliases: Sequence[str]) -> Any:
    alias_set = {normalize_key(alias) for alias in aliases}
    if isinstance(data, dict):
        for key, value in data.items():
            if normalize_key(key) in alias_set:
                return value
        for value in data.values():
            found = find_metric_value(value, aliases)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = find_metric_value(item, aliases)
            if found is not None:
                return found
    return None


def row_from_records(records: Iterable[Any]) -> Optional[Dict[str, Any]]:
    rows = [row for row in records if isinstance(row, dict)]
    if not rows:
        return None

    for row in rows:
        token = str(row.get("token", row.get("name", ""))).strip().lower()
        if token in {"average", "avg", "mean", "summary"}:
            return row

    aggregated: Dict[str, Any] = {}
    keys = sorted({key for row in rows for key in row.keys()})
    for key in keys:
        values = [as_number(row.get(key)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            aggregated[key] = sum(numeric_values) / len(numeric_values)
    return aggregated if aggregated else rows[-1]


def read_csv_result(path: Path) -> Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        warn(f"Failed to parse CSV {path}: {exc}")
        return {}, None

    selected_row = row_from_records(rows)
    if selected_row is None:
        warn(f"No rows found in CSV result file: {path}")
        return {}, None
    return extract_metrics(selected_row, path), None


def read_json_result(path: Path) -> Tuple[Dict[str, Optional[float]], Optional[Dict[str, Any]]]:
    data = load_json(path)
    if data is None:
        return {}, None
    row = flatten_metric_row(data)
    if row is None:
        warn(f"Could not find NAVSIM PDM metrics in JSON result file: {path}")
        return {}, config_from_result_data(data)
    return extract_metrics(row, path), config_from_result_data(data)


def extract_metrics(row: Dict[str, Any], path: Path) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for metric_name, aliases in METRIC_ALIASES.items():
        value = find_metric_value(row, aliases)
        number = as_number(value)
        if number is None:
            warn(f"Missing metric {metric_name} in {path}")
        metrics[metric_name] = number
    return metrics


def find_result_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in RESULT_SUFFIXES else []

    candidates: List[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in RESULT_SUFFIXES:
            continue
        if is_config_file(candidate):
            continue
        lower_name = candidate.name.lower()
        lower_parent = str(candidate.parent).lower()
        looks_like_result = (
            "pdm" in lower_name
            or "score" in lower_name
            or "result" in lower_name
            or "eval" in lower_parent
            or candidate.suffix.lower() == ".csv"
        )
        if looks_like_result:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)


def find_config_files(base_path: Path) -> List[Path]:
    search_root = base_path.parent if base_path.is_file() else base_path
    candidates: List[Path] = []
    for relative in (
        Path("code/hydra/config.yaml"),
        Path("code/hydra/config.yml"),
        Path(".hydra/config.yaml"),
        Path(".hydra/config.yml"),
        Path("config.yaml"),
        Path("config.yml"),
        Path("config.json"),
    ):
        candidate = search_root / relative
        if candidate.is_file():
            candidates.append(candidate)

    for candidate in search_root.rglob("*"):
        if candidate.is_file() and is_config_file(candidate) and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def choose_result_file(candidates: List[Path], input_path: Path) -> Optional[Path]:
    if not candidates:
        warn(f"No result JSON/CSV file found for input: {input_path}")
        return None
    if len(candidates) > 1:
        warn(
            f"Found {len(candidates)} result files under {input_path}; "
            f"using newest: {candidates[0]}"
        )
    return candidates[0]


def choose_config_file(candidates: List[Path], input_path: Path) -> Optional[Path]:
    if not candidates:
        return None
    if len(candidates) > 1:
        warn(
            f"Found {len(candidates)} config files under {input_path}; "
            f"using first: {candidates[0]}"
        )
    return candidates[0]


def parse_input_spec(spec: str) -> Tuple[Optional[str], Path]:
    if "=" in spec:
        label, path_text = spec.split("=", 1)
        if label and path_text:
            return label, Path(path_text).expanduser()
    return None, Path(spec).expanduser()


def infer_variant(path: Path, config: Dict[str, Any], label_hint: Optional[str]) -> str:
    if label_hint:
        normalized_label = normalize_key(label_hint)
        if normalized_label in VARIANT_ORDER:
            return normalized_label

    use_expert = parse_bool(get_config_value(config, "use_expert_features"))
    use_jepa = parse_bool(get_config_value(config, "use_jepa"))
    use_vggt = parse_bool(get_config_value(config, "use_vggt"))
    jepa_align = as_number(get_config_value(config, "jepa_alignment_weight")) or 0.0
    vggt_align = as_number(get_config_value(config, "vggt_alignment_weight")) or 0.0
    expert_align = as_number(get_config_value(config, "expert_alignment_weight")) or 0.0

    if use_expert is False:
        return "baseline"
    if use_jepa is True and use_vggt is False:
        return "jepa"
    if use_jepa is False and use_vggt is True:
        return "vggt"
    if use_jepa is True and use_vggt is True:
        if jepa_align > 0 or vggt_align > 0 or expert_align > 0:
            return "jepa_vggt_alignment"
        return "jepa_vggt"

    text = normalize_key(f"{label_hint or ''} {path}")
    if "alignment" in text and "jepa" in text and "vggt" in text:
        return "jepa_vggt_alignment"
    if "jepa_vggt" in text or ("jepa" in text and "vggt" in text):
        return "jepa_vggt"
    if "vggt" in text:
        return "vggt"
    if "jepa" in text:
        return "jepa"
    if "baseline" in text:
        return "baseline"
    return normalize_key(label_hint or path.stem) or "unknown"


def display_label(variant: str, label_hint: Optional[str]) -> str:
    if label_hint and normalize_key(label_hint) not in VARIANT_ORDER:
        return label_hint
    return VARIANT_DISPLAY.get(variant, variant)


def aggregate_input(spec: str) -> AggregatedRun:
    label_hint, input_path = parse_input_spec(spec)
    result_file = choose_result_file(find_result_files(input_path), input_path)

    metrics: Dict[str, Optional[float]] = {metric: None for metric in METRIC_ALIASES}
    inline_config: Optional[Dict[str, Any]] = None
    if result_file is not None:
        if result_file.suffix.lower() == ".csv":
            metrics, inline_config = read_csv_result(result_file)
        elif result_file.suffix.lower() == ".json":
            metrics, inline_config = read_json_result(result_file)
        else:
            warn(f"Unsupported result file type: {result_file}")

    config_file = choose_config_file(find_config_files(input_path), input_path)
    config: Dict[str, Any] = {}
    if inline_config:
        config.update(inline_config)
    if config_file is not None:
        loaded_config = load_config(config_file)
        if loaded_config:
            config.update(loaded_config)
    elif not inline_config:
        warn(f"No Hydra/config file found for input: {input_path}; config summary fields may be blank.")

    variant = infer_variant(input_path, config, label_hint)
    label = display_label(variant, label_hint)

    summary = {column: get_config_value(config, column) for column in CONFIG_COLUMNS}
    return AggregatedRun(
        label=label,
        variant=variant,
        source=input_path,
        result_file=result_file,
        config_file=config_file,
        metrics=metrics,
        config=summary,
    )


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return ""
    return str(value)


def output_rows(runs: Sequence[AggregatedRun]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run in runs:
        row: Dict[str, str] = {
            "variant": run.label,
            "source": str(run.source),
            "result_file": str(run.result_file or ""),
            "config_file": str(run.config_file or ""),
        }
        for metric in METRIC_ALIASES:
            row[metric] = format_value(run.metrics.get(metric))
        for column in CONFIG_COLUMNS:
            row[column] = format_value(run.config.get(column))
        rows.append(row)
    return rows


def sort_runs(runs: Sequence[AggregatedRun]) -> List[AggregatedRun]:
    return sorted(
        runs,
        key=lambda run: (
            VARIANT_ORDER.get(run.variant, 100),
            run.label,
            str(run.source),
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", *METRIC_ALIASES.keys(), *CONFIG_COLUMNS, "source", "result_file", "config_file"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["variant", *METRIC_ALIASES.keys(), *CONFIG_COLUMNS, "checkpoint_path"]
    # checkpoint_path is already part of CONFIG_COLUMNS; keep a stable order without duplication.
    columns = [column for index, column in enumerate(columns) if column not in columns[:index]]

    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(markdown_escape(row.get(column, "")) for column in columns) + " |\n")


def print_missing_expected_variants(runs: Sequence[AggregatedRun]) -> None:
    present = {run.variant for run in runs}
    for variant in VARIANT_ORDER:
        if variant not in present:
            warn(f"No input resolved to expected variant: {VARIANT_DISPLAY[variant]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "PDM result JSON/CSV files or output directories. Use label=path to force a row label, "
            "for example baseline=/exp/baseline jepa=/exp/jepa."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("recogdrive_expert_ablation_results.csv"),
        help="Path for the aggregated CSV table.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("recogdrive_expert_ablation_results.md"),
        help="Path for the aggregated Markdown table.",
    )
    parser.add_argument(
        "--allow-missing-variants",
        action="store_true",
        help="Do not warn when one of the standard five expert ablation variants is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = sort_runs([aggregate_input(spec) for spec in args.inputs])
    if not args.allow_missing_variants:
        print_missing_expected_variants(runs)

    rows = output_rows(runs)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
