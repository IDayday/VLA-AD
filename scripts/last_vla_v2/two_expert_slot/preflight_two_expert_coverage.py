#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import load_path_index


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Two-Expert Teacher/Base Coverage",
        "",
        f"- OK: `{report['ok']}`",
        f"- Base count: `{report['base_count']}`",
        f"- JEPA count: `{report['jepa_count']}`",
        f"- VGGT count: `{report['vggt_count']}`",
        f"- Intersection count: `{report['intersection_count']}`",
        f"- JEPA/base coverage: `{report['jepa_base_coverage']:.6f}`",
        f"- VGGT/base coverage: `{report['vggt_base_coverage']:.6f}`",
        f"- All-teacher coverage: `{report['all_teacher_coverage']:.6f}`",
        f"- Min train coverage: `{report['min_train_coverage']:.6f}`",
        "",
        "Errors:",
    ]
    if report["errors"]:
        lines.extend(f"- `{item}`" for item in report["errors"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coverage(numerator: Set[str], denominator: Set[str]) -> float:
    return float(len(numerator)) / float(len(denominator)) if denominator else 0.0


def coverage_report(
    *,
    base_chunk_root: Path,
    jepa_cache_root: Path,
    vggt_cache_root: Path,
    min_train_coverage: float = 0.99,
    max_records: Optional[int] = None,
    chunk_name_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    base = set(load_path_index(base_chunk_root, pattern=chunk_name_pattern, max_records=max_records))
    jepa = set(load_path_index(jepa_cache_root, max_records=max_records))
    vggt = set(load_path_index(vggt_cache_root, max_records=max_records))
    base_jepa = base.intersection(jepa)
    base_vggt = base.intersection(vggt)
    base_all = base.intersection(jepa).intersection(vggt)
    errors = []
    if not base:
        errors.append("base_count_zero")
    all_coverage = _coverage(base_all, base)
    if all_coverage < float(min_train_coverage):
        errors.append(f"all_teacher_coverage_below_min:{all_coverage:.6f}<{float(min_train_coverage):.6f}")
    return {
        "ok": not errors,
        "base_chunk_root": str(base_chunk_root),
        "jepa_cache_root": str(jepa_cache_root),
        "vggt_cache_root": str(vggt_cache_root),
        "base_count": len(base),
        "jepa_count": len(jepa),
        "vggt_count": len(vggt),
        "intersection_count": len(base_all),
        "jepa_base_intersection_count": len(base_jepa),
        "vggt_base_intersection_count": len(base_vggt),
        "jepa_base_coverage": _coverage(base_jepa, base),
        "vggt_base_coverage": _coverage(base_vggt, base),
        "all_teacher_coverage": all_coverage,
        "min_train_coverage": float(min_train_coverage),
        "max_records": max_records,
        "chunk_name_pattern": chunk_name_pattern,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight base/teacher coverage for two_expert_slot Stage1.")
    parser.add_argument("--base-chunk-root", type=Path, required=True)
    parser.add_argument("--jepa-cache-root", type=Path, required=True)
    parser.add_argument("--vggt-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train-coverage", type=float, default=0.99)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--chunk-name-pattern", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = coverage_report(
        base_chunk_root=args.base_chunk_root,
        jepa_cache_root=args.jepa_cache_root,
        vggt_cache_root=args.vggt_cache_root,
        min_train_coverage=float(args.min_train_coverage),
        max_records=args.max_records,
        chunk_name_pattern=args.chunk_name_pattern,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "coverage.json", report)
    write_markdown(args.output_dir / "coverage.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
