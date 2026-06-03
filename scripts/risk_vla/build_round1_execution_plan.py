#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _cmd(script: str, env: Dict[str, str]) -> str:
    exports = " ".join(f"{key}={value!r}" for key, value in env.items())
    return f"{exports} bash {script}"


def build_plan(report: Dict[str, Any], output_dir: Path, *, max_samples: int, limit_train_batches: int, limit_val_batches: int) -> Dict[str, Any]:
    found = report.get("found_inputs", {})
    round_dir = output_dir / "round1_outputs"
    cache_dir = output_dir / "cache"
    labels_jsonl = round_dir / "risk_labels" / "risk_labels.jsonl"
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
        },
        {
            "name": "01_build_risk_labels",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_01_build_risk_labels.sh",
            "env": {"DRY_RUN": "1", "ROUND_DIR": str(round_dir)},
        },
        {
            "name": "02_check_leakage",
            "expensive": False,
            "analysis_only": False,
            "script": "python scripts/risk_vla/check_label_leakage.py",
            "raw_command": (
                f"python scripts/risk_vla/check_label_leakage.py --labels-jsonl {labels_jsonl} "
                f"--for-training --output-md {round_dir / 'risk_labels' / 'leakage_check.md'}"
            ),
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
                "BIT_CACHE_ROOT": str(cache_dir),
                "MAX_SAMPLES": str(max_samples),
            },
        },
        {
            "name": "04_train_r0_diagnostic",
            "expensive": True,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_10_execute_r0_diagnostic_small.sh",
            "env": {
                "EXECUTE": "0",
                "ROUND_DIR": str(round_dir),
                "BIT_CACHE_ROOT": str(cache_dir),
                "CHECKPOINT_PATH": found.get("checkpoint_dir", "<missing>"),
                "MAX_SAMPLES": str(max_samples),
                "LIMIT_TRAIN_BATCHES": str(limit_train_batches),
                "LIMIT_VAL_BATCHES": str(limit_val_batches),
            },
        },
        {
            "name": "05_export_predictions",
            "expensive": False,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_11_export_r0_predictions_small.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "BIT_CACHE_ROOT": str(cache_dir), "MAX_SAMPLES": str(max_samples)},
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
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "BIT_CACHE_ROOT": str(cache_dir), "MAX_SAMPLES": str(min(max_samples, 256))},
        },
        {
            "name": "08_predicted_router_pilot",
            "expensive": True,
            "analysis_only": False,
            "script": "scripts/risk_vla/round1/run_14_execute_r2_predicted_router_small.sh",
            "env": {"EXECUTE": "0", "ROUND_DIR": str(round_dir), "BIT_CACHE_ROOT": str(cache_dir), "MAX_SAMPLES": str(min(max_samples, 256))},
        },
        {
            "name": "09_build_report",
            "expensive": False,
            "analysis_only": False,
            "script": "python scripts/risk_vla/build_round1_evidence_report.py",
            "raw_command": (
                f"python scripts/risk_vla/build_round1_evidence_report.py --input-discovery-md {output_dir / 'input_discovery.md'} "
                f"--dryrun-summary-md {round_dir / 'logs' / 'dryrun' / 'round1_dryrun_summary.md'} "
                f"--r0-diagnostics-md {round_dir / 'R0_risk_head_diagnostic' / 'risk_vla_diagnostics.md'} "
                f"--go-no-go-md {round_dir / 'go_no_go_report.md'} --output-md {round_dir / 'ROUND1_EVIDENCE_REPORT.md'}"
            ),
        },
    ]
    for item in commands:
        if "raw_command" not in item:
            item["raw_command"] = _cmd(item["script"], item["env"])
    return {"output_dir": str(output_dir), "round_dir": str(round_dir), "commands": commands, "missing_inputs": report.get("missing_inputs", [])}


def write_plan(plan: Dict[str, Any], output_dir: Path) -> None:
    commands_dir = output_dir / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    for item in plan["commands"]:
        path = commands_dir / f"{item['name']}.sh"
        note = "analysis-only" if item.get("analysis_only") else ("expensive" if item.get("expensive") else "safe")
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"# RISK-VLA Round 1 command: {item['name']} ({note}).\n"
            "# Defaults are dry-run/no-execute. Edit only after reviewing inputs.\n"
            f"{item['raw_command']}\n",
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
        lines.append("")
        lines.append("```bash")
        lines.append(item["raw_command"])
        lines.append("```")
        lines.append("")
    (output_dir / "round1_execution_plan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dry-run-first RISK-VLA Round 1 execution plan.")
    parser.add_argument("--input-report-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--limit-train-batches", type=int, default=50)
    parser.add_argument("--limit-val-batches", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.input_report_json.read_text(encoding="utf-8"))
    plan = build_plan(
        report,
        args.output_dir,
        max_samples=args.max_samples,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
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
