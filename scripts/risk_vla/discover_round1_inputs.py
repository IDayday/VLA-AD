#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PDM_METRIC_COLUMNS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
]
CHECKPOINT_SUFFIXES = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")


def env_path(name: str, default: Optional[str] = None) -> Optional[Path]:
    value = os.getenv(name, default)
    return Path(value) if value else None


def token_column(columns: Iterable[str]) -> Optional[str]:
    for name in ("token", "sample_token", "scene_token"):
        if name in columns:
            return name
    return None


def validate_pdm_csv(path: Path) -> tuple[bool, List[str]]:
    warnings: List[str] = []
    if not path.is_file():
        return False, [f"PDM CSV not found: {path}"]
    try:
        df = pd.read_csv(path, nrows=10)
    except Exception as exc:
        return False, [f"Could not read PDM CSV {path}: {exc}"]
    if token_column(df.columns) is None:
        warnings.append(f"{path} has no token/sample_token/scene_token column.")
    missing_metrics = [col for col in PDM_METRIC_COLUMNS if col not in df.columns]
    if len(missing_metrics) == len(PDM_METRIC_COLUMNS):
        warnings.append(f"{path} has none of the expected PDM metric columns.")
    elif missing_metrics:
        warnings.append(f"{path} is missing optional PDM columns: {', '.join(missing_metrics)}")
    return True, warnings


def validate_chunk_cache(path: Path) -> tuple[bool, List[str]]:
    warnings: List[str] = []
    if not path.is_dir():
        return False, [f"Chunk cache dir not found: {path}"]
    has_index = (path / "index.jsonl").is_file()
    has_samples = any(path.glob("*.pt"))
    if (path / "samples").exists():
        has_samples = has_samples or any((path / "samples").glob("*.pt"))
    if not has_index and not has_samples:
        warnings.append(f"{path} does not look like a chunk cache: no index.jsonl or .pt samples found near root.")
    return True, warnings


def checkpoint_candidates(path: Path, *, deep_search: bool = False) -> List[Path]:
    if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES:
        return [path]
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if deep_search else path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix in CHECKPOINT_SUFFIXES)


def validate_checkpoint_dir(path: Path, *, deep_search: bool = False) -> tuple[bool, List[str], List[str]]:
    if not path.exists():
        return False, [f"Checkpoint path not found: {path}"], []
    candidates = checkpoint_candidates(path, deep_search=deep_search)
    warnings = [] if candidates else [f"No checkpoint candidates found under {path}."]
    return True, warnings, [str(item) for item in candidates[:10]]


def likely_csv(root: Optional[Path], keywords: Iterable[str], *, deep_search: bool = False) -> Optional[Path]:
    if root is None or not root.is_dir():
        return None
    patterns = ["*.csv"] if not deep_search else ["**/*.csv"]
    lowered = [kw.lower() for kw in keywords]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            name = path.name.lower()
            if all(kw in name for kw in lowered):
                return path
    return None


def discover_inputs(
    *,
    a0_pdm_csv: Optional[Path],
    bit_pdm_csv: Optional[Path],
    chunk_cache_dir: Optional[Path],
    checkpoint_dir: Optional[Path],
    deep_search: bool = False,
) -> Dict[str, Any]:
    bit_exp_root = env_path("BIT_EXP_ROOT", "/mnt/project/bit_drive_left_tail/experiments/risk_vla")
    shared_chunk = env_path("SHARED_CHUNK_CACHE_ROOT", "/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1")
    checkpoint_root = env_path("CHECKPOINT_ROOT", "/mnt/project/VLA-AD/checkpoints")

    if a0_pdm_csv is None:
        a0_pdm_csv = likely_csv(bit_exp_root, ("a0", "pdm"), deep_search=deep_search)
    if bit_pdm_csv is None:
        bit_pdm_csv = likely_csv(bit_exp_root, ("bit", "pdm"), deep_search=deep_search) or likely_csv(
            bit_exp_root, ("b3", "pdm"), deep_search=deep_search
        )
    if chunk_cache_dir is None:
        chunk_cache_dir = shared_chunk
    if checkpoint_dir is None:
        checkpoint_dir = checkpoint_root / "recogdrive" / "ReCogDrive-2B-IL" if checkpoint_root else None

    found: Dict[str, Any] = {}
    missing: List[str] = []
    warnings: List[str] = []

    for key, path in (("a0_pdm_csv", a0_pdm_csv), ("bit_pdm_csv", bit_pdm_csv)):
        if path is None:
            missing.append(key)
            continue
        ok, warn = validate_pdm_csv(path)
        warnings.extend(warn)
        if ok:
            found[key] = str(path)
        else:
            missing.append(key)

    if chunk_cache_dir is None:
        missing.append("chunk_cache_dir")
    else:
        ok, warn = validate_chunk_cache(chunk_cache_dir)
        warnings.extend(warn)
        if ok:
            found["chunk_cache_dir"] = str(chunk_cache_dir)
        else:
            missing.append("chunk_cache_dir")

    if checkpoint_dir is None:
        missing.append("checkpoint_dir")
    else:
        ok, warn, candidates = validate_checkpoint_dir(checkpoint_dir, deep_search=deep_search)
        warnings.extend(warn)
        if ok:
            found["checkpoint_dir"] = str(checkpoint_dir)
            found["checkpoint_candidates"] = candidates
        else:
            missing.append("checkpoint_dir")

    recommended = (
        "python scripts/risk_vla/build_round1_execution_plan.py "
        "--input-report-json <this_report.json> --output-dir ${BIT_EXP_ROOT}/round1/plan --write"
        if not missing
        else "Provide missing inputs explicitly, then rerun discover_round1_inputs.py with --strict."
    )
    return {
        "env": {
            "BIT_EXP_ROOT": str(bit_exp_root) if bit_exp_root else None,
            "SHARED_CHUNK_CACHE_ROOT": str(shared_chunk) if shared_chunk else None,
            "CHECKPOINT_ROOT": str(checkpoint_root) if checkpoint_root else None,
        },
        "found_inputs": found,
        "missing_inputs": missing,
        "warnings": warnings,
        "recommended_next_command": recommended,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# RISK-VLA Round 1 Input Discovery", "", "## Found Inputs", ""]
    found = report.get("found_inputs", {})
    if found:
        for key, value in found.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("_None._")
    lines.extend(["", "## Missing Inputs", ""])
    missing = report.get("missing_inputs", [])
    lines.extend(f"- `{item}`" for item in missing) if missing else lines.append("_None._")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("_None._")
    lines.extend(["", "## Recommended Next Command", "", f"```bash\n{report.get('recommended_next_command', '')}\n```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover or validate real inputs for RISK-VLA Round 1.")
    parser.add_argument("--a0-pdm-csv", type=Path, default=None)
    parser.add_argument("--bit-pdm-csv", type=Path, default=None)
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--deep-search", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = discover_inputs(
        a0_pdm_csv=args.a0_pdm_csv,
        bit_pdm_csv=args.bit_pdm_csv,
        chunk_cache_dir=args.chunk_cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        deep_search=bool(args.deep_search),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["missing_inputs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
