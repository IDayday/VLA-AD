#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.risk_vla.round1_manifest import inspect_pdm_csv


def git_info() -> Dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "<unknown>", "commit": "<unknown>"}


def file_status(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def optional_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def build_dashboard(round_dir: Path) -> Dict[str, Any]:
    pdm_inputs = round_dir / "pdm_inputs"
    r0_dir = round_dir / "R0_risk_head_diagnostic"
    pdm_csvs = {
        "A0_base": pdm_inputs / "A0_base" / "pdm.csv",
        "B3_direct_bit": pdm_inputs / "B3_direct_bit" / "pdm.csv",
        "train": pdm_inputs / "train" / "pdm.csv",
        "val": pdm_inputs / "val" / "pdm.csv",
    }
    dashboard: Dict[str, Any] = {
        "git": git_info(),
        "round_dir": str(round_dir),
        "pdm_csvs": {name: inspect_pdm_csv(path) for name, path in pdm_csvs.items()},
        "manifest": {
            "json": file_status(round_dir / "input_manifest_resolved.json"),
            "md": file_status(round_dir / "input_manifest_resolved.md"),
            "local_yaml": file_status(round_dir / "round1_input_manifest.local.yaml"),
        },
        "pdm_generation": {
            "commands_md": file_status(pdm_inputs / "small_pdm_eval_commands.md"),
            "blockers_md": file_status(pdm_inputs / "pdm_generation_blockers.md"),
            "evaluator_report_md": file_status(pdm_inputs / "evaluator_path_report.md"),
            "blockers_text": optional_text(pdm_inputs / "pdm_generation_blockers.md"),
        },
        "r0": {
            "output_dir": str(r0_dir),
            "risk_diagnostics_md": file_status(r0_dir / "risk_vla_diagnostics.md"),
            "risk_metrics_csv": file_status(r0_dir / "risk_prediction_metrics.csv"),
            "strategy_activation_csv": file_status(r0_dir / "strategy_activation_by_subset.csv"),
            "go_no_go_md": file_status(r0_dir / "go_no_go_report.md"),
            "checkpoint_candidates": [str(path) for path in sorted(r0_dir.rglob("*.ckpt"))[:10]] if r0_dir.is_dir() else [],
        },
        "reports": {
            "evidence_report": file_status(round_dir / "ROUND1_EVIDENCE_REPORT.md"),
            "artifact_dashboard": file_status(round_dir / "ROUND1_ARTIFACT_DASHBOARD.md"),
        },
    }
    all_pdm_ready = all(item.get("has_required_pdm_columns") for item in dashboard["pdm_csvs"].values())
    manifest_ready = dashboard["manifest"]["json"]["exists"]
    r0_ready = dashboard["r0"]["risk_diagnostics_md"]["exists"]
    dashboard["summary"] = {
        "all_required_pdm_csvs_ready": bool(all_pdm_ready),
        "manifest_present": bool(manifest_ready),
        "r0_diagnostic_present": bool(r0_ready),
    }
    return dashboard


def write_markdown(dashboard: Dict[str, Any], output_md: Path) -> None:
    lines = [
        "# RISK-VLA Round 1 Artifact Dashboard",
        "",
        "This dashboard tracks input materialization for low-score risk scenarios / critical-risk subsets. It is not a final PDM performance claim.",
        "",
        f"- branch: `{dashboard['git']['branch']}`",
        f"- commit: `{dashboard['git']['commit']}`",
        f"- round dir: `{dashboard['round_dir']}`",
        "",
        "## PDM CSVs",
        "",
        "| artifact | exists | required columns | rows | missing columns | path |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for name, info in dashboard["pdm_csvs"].items():
        lines.append(
            f"| {name} | {info.get('exists')} | {info.get('has_required_pdm_columns')} | "
            f"{info.get('row_count')} | {info.get('missing_columns')} | `{info.get('path')}` |"
        )
    lines.extend(["", "## Manifest", ""])
    for key, info in dashboard["manifest"].items():
        lines.append(f"- `{key}`: exists=`{info['exists']}`, path=`{info['path']}`")
    lines.extend(["", "## PDM Generation", ""])
    for key, info in dashboard["pdm_generation"].items():
        if isinstance(info, dict):
            lines.append(f"- `{key}`: exists=`{info['exists']}`, path=`{info['path']}`")
    lines.extend(["", "## R0 Diagnostic", ""])
    for key, info in dashboard["r0"].items():
        if isinstance(info, dict):
            lines.append(f"- `{key}`: exists=`{info['exists']}`, path=`{info['path']}`")
    lines.extend(["", "## Summary", ""])
    for key, value in dashboard["summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dashboard of RISK-VLA Round 1 artifacts.")
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    dashboard = build_dashboard(args.round_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dashboard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(dashboard, args.output_md)
    print(json.dumps(dashboard["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
