from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml


TOKEN_COLUMNS = ("token", "sample_token", "scene_token")
PDM_COLUMNS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
]
BLOCKED_TRAIN_SPLIT_MARKERS = ("test", "navtest", "challenge", "eval-only")


def _as_path(value: Any) -> Optional[Path]:
    return Path(str(value)).expanduser() if value not in (None, "") else None


def _split_blocked(split: str) -> bool:
    lowered = split.lower()
    return any(marker in lowered for marker in BLOCKED_TRAIN_SPLIT_MARKERS)


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")
    return data


def token_column(columns: Iterable[str]) -> Optional[str]:
    available = set(columns)
    for column in TOKEN_COLUMNS:
        if column in available:
            return column
    return None


def inspect_pdm_csv(path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "readable": False,
        "row_count": None,
        "columns": [],
        "token_column": None,
        "missing_columns": [],
        "has_required_pdm_columns": False,
        "warnings": [],
    }
    if not path.is_file():
        result["warnings"].append(f"PDM CSV not found: {path}")
        result["missing_columns"] = ["token", *PDM_COLUMNS]
        return result
    try:
        header = pd.read_csv(path, nrows=0)
        result["columns"] = [str(col) for col in header.columns]
        result["token_column"] = token_column(result["columns"])
        missing = []
        if result["token_column"] is None:
            missing.append("token")
        missing.extend(column for column in PDM_COLUMNS if column not in result["columns"])
        result["missing_columns"] = missing
        try:
            result["row_count"] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
        except Exception:
            result["row_count"] = int(len(pd.read_csv(path)))
        result["readable"] = True
        result["has_required_pdm_columns"] = not missing
    except Exception as exc:
        result["warnings"].append(f"Could not inspect PDM CSV {path}: {exc}")
        result["missing_columns"] = ["token", *PDM_COLUMNS]
    return result


def _validate_pdm_entry(
    *,
    section: str,
    name: str,
    entry: Dict[str, Any],
    require_training_safe: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    csv_path = _as_path(entry.get("csv"))
    split = str(entry.get("split") or "").strip()
    use_for_training = bool(entry.get("use_for_training", False))
    if csv_path is None:
        info = {"path": None, "exists": False, "has_required_pdm_columns": False, "missing_columns": ["csv"]}
        errors.append(f"{section}.{name}: missing csv path.")
    else:
        info = inspect_pdm_csv(csv_path)
        if not info["exists"]:
            errors.append(f"{section}.{name}: CSV does not exist: {csv_path}")
        if info["exists"] and not info["has_required_pdm_columns"]:
            errors.append(f"{section}.{name}: missing PDM columns: {', '.join(info['missing_columns'])}")
        warnings.extend(info.get("warnings", []))
    if use_for_training and _split_blocked(split):
        errors.append(f"{section}.{name}: split={split!r} cannot be used for training labels.")
    if require_training_safe:
        if not use_for_training:
            warnings.append(f"{section}.{name}: train/val entry has use_for_training=false.")
        if not split:
            errors.append(f"{section}.{name}: training entry must declare split.")
        elif _split_blocked(split):
            errors.append(f"{section}.{name}: training split is blocked: {split!r}.")
    else:
        if use_for_training:
            warnings.append(f"{section}.{name}: analysis PDM is marked use_for_training=true; this is not recommended.")
    return {
        "name": name,
        "csv": str(csv_path) if csv_path else None,
        "split": split,
        "use_for_training": use_for_training,
        "pdm": info,
        "errors": errors,
        "warnings": warnings,
    }


def validate_manifest(path: Path) -> Dict[str, Any]:
    data = load_manifest(path)
    errors: List[str] = []
    warnings: List[str] = []
    analysis_entries: Dict[str, Any] = {}
    trainval_entries: Dict[str, Any] = {}

    for name, entry in (data.get("analysis_pdm") or {}).items():
        if not isinstance(entry, dict):
            errors.append(f"analysis_pdm.{name}: entry must be a mapping.")
            continue
        checked = _validate_pdm_entry(section="analysis_pdm", name=str(name), entry=entry, require_training_safe=False)
        analysis_entries[str(name)] = checked
        errors.extend(checked["errors"])
        warnings.extend(checked["warnings"])

    for name, entry in (data.get("trainval_pdm") or {}).items():
        if not isinstance(entry, dict):
            errors.append(f"trainval_pdm.{name}: entry must be a mapping.")
            continue
        checked = _validate_pdm_entry(section="trainval_pdm", name=str(name), entry=entry, require_training_safe=True)
        trainval_entries[str(name)] = checked
        errors.extend(checked["errors"])
        warnings.extend(checked["warnings"])
    if not trainval_entries:
        warnings.append("No trainval_pdm entries are defined; R0 training cannot run from this manifest.")

    cache = data.get("cache") or {}
    source_cache = _as_path(cache.get("source_chunk_cache_dir"))
    overlay = _as_path(cache.get("overlay_output_dir"))
    if source_cache is None:
        errors.append("cache.source_chunk_cache_dir is required.")
    elif not source_cache.is_dir():
        errors.append(f"cache.source_chunk_cache_dir does not exist: {source_cache}")
    if overlay is None:
        warnings.append("cache.overlay_output_dir is missing; wrappers will use BIT_CACHE_ROOT default overlay.")

    checkpoints = data.get("checkpoints") or {}
    checkpoint = _as_path(checkpoints.get("base_or_bit_checkpoint"))
    if checkpoint is None:
        warnings.append("checkpoints.base_or_bit_checkpoint is missing; R0 command will rely on defaults.")
    elif not checkpoint.exists():
        errors.append(f"checkpoints.base_or_bit_checkpoint does not exist: {checkpoint}")

    found_inputs = manifest_found_inputs(analysis_entries, trainval_entries, source_cache, overlay, checkpoint)
    return {
        "manifest_path": str(path),
        "round": data.get("round"),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "analysis_pdm": analysis_entries,
        "trainval_pdm": trainval_entries,
        "cache": {
            "source_chunk_cache_dir": str(source_cache) if source_cache else None,
            "overlay_output_dir": str(overlay) if overlay else None,
        },
        "checkpoints": {"base_or_bit_checkpoint": str(checkpoint) if checkpoint else None},
        "found_inputs": found_inputs,
        "missing_inputs": manifest_missing_inputs(found_inputs),
    }


def manifest_found_inputs(
    analysis_entries: Dict[str, Any],
    trainval_entries: Dict[str, Any],
    source_cache: Optional[Path],
    overlay: Optional[Path],
    checkpoint: Optional[Path],
) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    for key, aliases in {
        "a0_pdm_csv": ("A0_base", "A0", "base"),
        "bit_pdm_csv": ("B3_direct_bit", "B3", "bit", "direct_bit"),
    }.items():
        for alias in aliases:
            entry = analysis_entries.get(alias)
            if entry and entry.get("csv"):
                found[key] = entry["csv"]
                break
    for name, entry in trainval_entries.items():
        if entry.get("csv"):
            found[f"{name}_pdm_csv"] = entry["csv"]
            found[f"{name}_split"] = entry.get("split")
    if source_cache is not None:
        found["chunk_cache_dir"] = str(source_cache)
    if overlay is not None:
        found["overlay_output_dir"] = str(overlay)
    if checkpoint is not None:
        found["checkpoint_dir"] = str(checkpoint)
        found["checkpoint_path"] = str(checkpoint)
    return found


def manifest_missing_inputs(found: Dict[str, Any]) -> List[str]:
    required = ["a0_pdm_csv", "bit_pdm_csv", "chunk_cache_dir"]
    return [key for key in required if not found.get(key)]


def markdown_report(report: Dict[str, Any]) -> str:
    lines = ["# RISK-VLA Round 1 Input Manifest Validation", "", f"Valid: `{report['valid']}`", ""]
    lines.extend(["## Analysis PDM", ""])
    analysis = report.get("analysis_pdm", {})
    if analysis:
        for name, entry in analysis.items():
            lines.append(
                f"- `{name}`: csv=`{entry.get('csv')}`, split=`{entry.get('split')}`, "
                f"use_for_training=`{entry.get('use_for_training')}`"
            )
    else:
        lines.append("_None._")
    lines.extend(["", "## Train/Val PDM", ""])
    trainval = report.get("trainval_pdm", {})
    if trainval:
        for name, entry in trainval.items():
            lines.append(
                f"- `{name}`: csv=`{entry.get('csv')}`, split=`{entry.get('split')}`, "
                f"use_for_training=`{entry.get('use_for_training')}`"
            )
    else:
        lines.append("_None._")
    lines.extend(["", "## Cache And Checkpoints", ""])
    lines.append(f"- source chunk cache: `{report.get('cache', {}).get('source_chunk_cache_dir')}`")
    lines.append(f"- overlay output: `{report.get('cache', {}).get('overlay_output_dir')}`")
    lines.append(f"- checkpoint: `{report.get('checkpoints', {}).get('base_or_bit_checkpoint')}`")
    lines.extend(["", "## Errors", ""])
    errors = report.get("errors", [])
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("_None._")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("_None._")
    lines.append("")
    return "\n".join(lines)
