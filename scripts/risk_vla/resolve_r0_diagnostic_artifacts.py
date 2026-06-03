#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKPOINT_SUFFIXES = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")


def find_files(root: Path, names: List[str]) -> List[Path]:
    if not root.is_dir():
        return []
    found: List[Path] = []
    for name in names:
        found.extend(path for path in root.rglob(name) if path.is_file())
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def find_checkpoints(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_artifacts(r0_output_dir: Path) -> Dict[str, Any]:
    checkpoints = find_checkpoints(r0_output_dir)
    best = next((path for path in checkpoints if "best" in path.name.lower()), None)
    latest = checkpoints[0] if checkpoints else None
    logs = find_files(r0_output_dir, ["train.log", "resolved_train_command.sh"])
    data_reports = find_files(r0_output_dir, ["data_report.json"])
    precision_reports = find_files(r0_output_dir, ["precision_report.json"])
    return {
        "r0_output_dir": str(r0_output_dir),
        "best_checkpoint": str(best) if best else None,
        "latest_checkpoint": str(latest) if latest else None,
        "checkpoint_candidates": [str(path) for path in checkpoints[:10]],
        "training_logs": [str(path) for path in logs],
        "data_report": str(data_reports[0]) if data_reports else None,
        "precision_report": str(precision_reports[0]) if precision_reports else None,
        "has_checkpoint": bool(checkpoints),
        "recommendation": (
            "Use best_checkpoint if available, otherwise latest_checkpoint for prediction export."
            if checkpoints
            else "No R0 checkpoint found. Use synthetic-smoke export only, or run run_10_execute_r0_diagnostic_small.sh with EXECUTE=1."
        ),
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# R0 Diagnostic Artifact Resolver", "", f"R0 output dir: `{report['r0_output_dir']}`", ""]
    lines.extend(
        [
            f"- best checkpoint: `{report.get('best_checkpoint')}`",
            f"- latest checkpoint: `{report.get('latest_checkpoint')}`",
            f"- data report: `{report.get('data_report')}`",
            f"- precision report: `{report.get('precision_report')}`",
            "",
            "## Checkpoint Candidates",
            "",
        ]
    )
    candidates = report.get("checkpoint_candidates", [])
    lines.extend(f"- `{item}`" for item in candidates) if candidates else lines.append("_None._")
    lines.extend(["", "## Recommendation", "", report.get("recommendation", ""), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve R0 diagnostic checkpoints and logs.")
    parser.add_argument("--r0-output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = resolve_artifacts(args.r0_output_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
