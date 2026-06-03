#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.risk_vla.round1_manifest import validate_manifest


def _cmd(script: str, env: Dict[str, str]) -> str:
    exports = " ".join(f"{key}={value!r}" for key, value in env.items())
    return f"{exports} bash {script}"


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _manifest_label_command(report: Dict[str, Any], round_dir: Path) -> str:
    label_dir = round_dir / "risk_labels"
    pieces = ["if [[ \"${EXECUTE:-0}\" == \"1\" ]]; then", f"  mkdir -p {_q(label_dir)}"]
    label_files: List[Path] = []
    for name, entry in sorted((report.get("trainval_pdm") or {}).items()):
        csv_path = entry.get("csv")
        split = entry.get("split") or name
        if not csv_path:
            continue
        output_jsonl = label_dir / f"{name}_risk_labels.jsonl"
        output_csv = label_dir / f"{name}_risk_label_summary.csv"
        output_md = label_dir / f"{name}_risk_label_report.md"
        label_files.append(output_jsonl)
        pieces.append(
            " ".join(
                [
                    "  python scripts/risk_vla/build_risk_labels_from_pdm.py",
                    "--input-csv",
                    _q(csv_path),
                    "--output-jsonl",
                    _q(output_jsonl),
                    "--output-csv",
                    _q(output_csv),
                    "--output-md",
                    _q(output_md),
                    "--schema both",
                    "--split",
                    _q(split),
                ]
            )
        )
    if label_files:
        pieces.append("  cat " + " ".join(_q(path) for path in label_files) + f" > {_q(label_dir / 'risk_labels.jsonl')}")
    else:
        pieces.append("  echo 'No train/val PDM entries available; cannot build R0 labels.'")
    pieces.extend(["else", "  echo 'EXECUTE=0; would build train/val risk labels from manifest PDM CSVs.'", "fi"])
    return "\n".join(pieces)


def build_plan(
    report: Dict[str, Any],
    output_dir: Path,
    *,
    max_samples: int,
    limit_train_batches: int,
    limit_val_batches: int,
    input_manifest_yaml: Path | None = None,
) -> Dict[str, Any]:
    found = report.get("found_inputs", {})
    round_dir = output_dir / "round1_outputs"
    cache_dir = output_dir / "cache"
    labels_jsonl = round_dir / "risk_labels" / "risk_labels.jsonl"
    overlay_output = found.get("overlay_output_dir", str(cache_dir / "risk_vla_round1_labeled_overlay"))
    comments: List[str] = []
    if input_manifest_yaml is not None:
        comments.append(f"Input manifest: {input_manifest_yaml}")
        analysis = report.get("analysis_pdm", {})
        trainval = report.get("trainval_pdm", {})
        for name, entry in analysis.items():
            comments.append(f"analysis-only PDM {name}: {entry.get('csv')} split={entry.get('split')}")
        for name, entry in trainval.items():
            comments.append(f"train/val label PDM {name}: {entry.get('csv')} split={entry.get('split')}")
    commands = [
        {
            "name": "00_collect_matched_pdm",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_00_collect_matched_pdm.sh",
            "env": {
                "DRY_RUN": "1",
                "ROUND_DIR": str(round_dir),
                "PDM_INPUTS": f"A0_base={found.get('a0_pdm_csv', '<missing>')} B3_direct_bit={found.get('bit_pdm_csv', '<missing>')}",
            },
            "comments": comments,
        },
        {
            "name": "01_build_risk_labels",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_01_build_risk_labels.sh",
            "env": {"DRY_RUN": "1", "ROUND_DIR": str(round_dir)},
            "raw_command": _manifest_label_command(report, round_dir) if input_manifest_yaml is not None else None,
            "comments": comments,
        },
        {
            "name": "02_check_leakage",
            "expensive": False,
            "analysis_only": False,
            "script": "python scripts/risk_vla/check_label_leakage.py",
            "raw_command": (
                "if [[ \"${EXECUTE:-0}\" == \"1\" ]]; then\n"
                f"  python scripts/risk_vla/check_label_leakage.py --labels-jsonl {_q(labels_jsonl)} "
                f"--for-training --output-md {_q(round_dir / 'risk_labels' / 'leakage_check.md')}\n"
                "else\n"
                "  echo 'EXECUTE=0; would check train/val risk-label leakage.'\n"
                "fi"
            ),
            "comments": ["Rejects navtest/test/challenge/eval-only labels for training."],
        },
        {
            "name": "03_create_overlay",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_02_merge_labels_to_chunk_cache.sh",
            "env": {
                "DRY_RUN": "1",
                "ROUND_DIR": str(round_dir),
                "SOURCE_CACHE_DIR": found.get("chunk_cache_dir", "<missing>"),
                "OUTPUT_CACHE_DIR": overlay_output,
                "MAX_SAMPLES": str(max_samples),
            },
            "comments": ["Creates an isolated overlay; source cache is never mutated."],
        },
        {
            "name": "04_train_r0_diagnostic",
            "expensive": True,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_10_execute_r0_diagnostic_small.sh",
            "env": {
                "EXECUTE": "0",
                "ROUND_DIR": str(round_dir),
                "CACHE_PATH": overlay_output,
                "CHECKPOINT_PATH": found.get("checkpoint_dir", "<missing>"),
                "MAX_SAMPLES": str(max_samples),
                "LIMIT_TRAIN_BATCHES": str(limit_train_batches),
                "LIMIT_VAL_BATCHES": str(limit_val_batches),
            },
            "comments": ["R0 diagnostic only: strategy token/residual scales remain zero."],
        },
        {
            "name": "05_export_predictions",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_11_export_r0_predictions_small.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "CACHE_PATH": overlay_output, "MAX_SAMPLES": str(max_samples)},
        },
        {
            "name": "06_aggregate_diagnostics",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_12_aggregate_r0_diagnostics.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir)},
        },
        {
            "name": "07_oracle_router_pilot",
            "expensive": True,
            "analysis_only": True,
            "script": "scripts/risk_vla/round1/run_13_execute_r1_oracle_router_small.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "CHUNK_CACHE_DIR": overlay_output, "MAX_SAMPLES": str(min(max_samples, 256))},
            "comments": ["Oracle-router is analysis-only."],
        },
        {
            "name": "08_predicted_router_pilot",
            "expensive": True,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_14_execute_r2_predicted_router_small.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "CHUNK_CACHE_DIR": overlay_output, "MAX_SAMPLES": str(min(max_samples, 256))},
            "comments": ["Predicted-router pilot should run only after GO/NO-GO permits it."],
        },
        {
            "name": "09_build_report",
            "expensive": False,
            "analysis_only": False,
            "script": "python scripts/risk_vla/build_round1_evidence_report.py",
            "raw_command": (
                "if [[ \"${EXECUTE:-0}\" == \"1\" ]]; then\n"
                f"  python scripts/risk_vla/build_round1_evidence_report.py --input-discovery-md {_q(output_dir / 'input_discovery.md')} "
                f"--dryrun-summary-md {_q(round_dir / 'logs' / 'dryrun' / 'round1_dryrun_summary.md')} "
                f"--r0-diagnostics-md {_q(round_dir / 'R0_risk_head_diagnostic' / 'risk_vla_diagnostics.md')} "
                f"--go-no-go-md {_q(round_dir / 'go_no_go_report.md')} --output-md {_q(round_dir / 'ROUND1_EVIDENCE_REPORT.md')}\n"
                "else\n"
                "  echo 'EXECUTE=0; would build Round 1 evidence report.'\n"
                "fi"
            ),
        },
    ]
    for item in commands:
        if not item.get("raw_command"):
            item["raw_command"] = _cmd(item["script"], item["env"])
    return {
        "output_dir": str(output_dir),
        "round_dir": str(round_dir),
        "commands": commands,
        "missing_inputs": report.get("missing_inputs", []),
        "input_manifest_yaml": str(input_manifest_yaml) if input_manifest_yaml is not None else None,
        "manifest_valid": report.get("valid") if input_manifest_yaml is not None else None,
    }


def write_plan(plan: Dict[str, Any], output_dir: Path) -> None:
    commands_dir = output_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    for item in plan["commands"]:
        path = commands_dir / f"{item['name']}.sh"
        note = "analysis-only" if item.get("analysis_only") else ("expensive" if item.get("expensive") else "safe")
        script_text = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"# RISK-VLA Round 1 command: {item['name']} ({note}).\n"
            "# Defaults are dry-run/no-execute. Edit only after reviewing inputs.\n"
            + "".join(f"# {comment}\n" for comment in item.get("comments", []))
            + f"{item['raw_command']}\n"
        )
        path.write_text(
            script_text,
            encoding="utf-8",
        )
        path.chmod(0o755)
    (output_dir / "round1_execution_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# RISK-VLA Round 1 Execution Plan", "", f"Output dir: `{output_dir}`", "", "## Missing Inputs", ""]
    missing = plan.get("missing_inputs", [])
    lines.extend(f"- `{item}`" for item in missing) if missing else lines.append("_None._")
    lines.extend(["", "## Commands", ""])
    for item in plan["commands"]:
        lines.append(f"### {item['name']}")
        lines.append("")
        if item.get("analysis_only"):
            lines.append("Analysis-only command.")
        elif item.get("expensive"):
            lines.append("Potentially expensive; defaults to no execution.")
        else:
            lines.append("Safe dry-run/check command.")
        for comment in item.get("comments", []):
            lines.append(f"- {comment}")
        lines.append("")
        lines.append("```bash")
        lines.append(item["raw_command"])
        lines.append("```")
        lines.append("")
    (output_dir / "round1_execution_plan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dry-run-first RISK-VLA Round 1 execution plan.")
    parser.add_argument("--input-report-json", type=Path, default=None)
    parser.add_argument("--input-manifest-yaml", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--limit-train-batches", type=int, default=50)
    parser.add_argument("--limit-val-batches", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.input_manifest_yaml is None and args.input_report_json is None:
        parser.error("Set --input-report-json or --input-manifest-yaml.")
    report = (
        validate_manifest(args.input_manifest_yaml)
        if args.input_manifest_yaml is not None
        else json.loads(args.input_report_json.read_text(encoding="utf-8"))
    )
    plan = build_plan(
        report,
        args.output_dir,
        max_samples=args.max_samples,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        input_manifest_yaml=args.input_manifest_yaml,
    )
    if args.write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_plan(plan, args.output_dir)
        print(f"Wrote execution plan to {args.output_dir}")
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("Dry-run only. Pass --write to materialize command files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
