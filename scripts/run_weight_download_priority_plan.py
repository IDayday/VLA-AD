#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER = REPO_ROOT / "scripts" / "download_required_weights.py"
DEFAULT_OUTPUT_ROOT = Path("/mnt/project/VLA-AD/checkpoints")
DEFAULT_WEIGHTS_CONFIG = REPO_ROOT / "configs" / "weights.yaml"
TARGET_ORDER = ["recogdrive_2b_il", "recogdrive_vlm_2b", "vjepa2", "vggt_1b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run required weight downloads in gate priority order.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-config", type=Path, default=DEFAULT_WEIGHTS_CONFIG)
    parser.add_argument("--source", choices=("auto", "huggingface", "modelscope", "hf-original", "hf-mirror"), default="auto")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--http-proxy", default=None)
    parser.add_argument("--https-proxy", default=None)
    parser.add_argument("--all-proxy", default=None)
    parser.add_argument("--repo-workers", type=int, default=1)
    parser.add_argument("--file-workers", type=int, default=None, help="Default is 8, except vggt_1b uses 4.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--stop-after-first-failure", action="store_true")
    parser.add_argument("--continue-on-failure", dest="continue_on_failure", action="store_true", default=True)
    parser.add_argument("--no-continue-on-failure", dest="continue_on_failure", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_workers_for(target: str, requested: Optional[int]) -> int:
    if requested is not None:
        return requested
    return 4 if target == "vggt_1b" else 8


def command_for(args: argparse.Namespace, target: str) -> List[str]:
    cmd = [
        sys.executable,
        str(DOWNLOADER),
        "--output-root",
        str(args.output_root),
        "--weights-config",
        str(args.weights_config),
        "--source",
        args.source,
        "--only",
        target,
        "--repo-workers",
        str(args.repo_workers),
        "--file-workers",
        str(file_workers_for(target, args.file_workers)),
        "--timeout",
        str(args.timeout),
    ]
    if args.proxy:
        cmd += ["--proxy", args.proxy]
    if args.http_proxy:
        cmd += ["--http-proxy", args.http_proxy]
    if args.https_proxy:
        cmd += ["--https-proxy", args.https_proxy]
    if args.all_proxy:
        cmd += ["--all-proxy", args.all_proxy]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_command(cmd: List[str]) -> int:
    print(f"\n$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def refresh_status(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(DOWNLOADER),
        "--status",
        "--output-root",
        str(args.output_root),
        "--weights-config",
        str(args.weights_config),
    ]
    print(f"\n$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    status_path = args.output_root / "download_status.json"
    if status_path.is_file():
        return json.loads(status_path.read_text(encoding="utf-8"))
    return {"records": []}


def status_by_target(status: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {record.get("target"): record for record in status.get("records", [])}


def retry_command(args: argparse.Namespace, target: str) -> str:
    cmd = command_for(args, target)
    if "--dry-run" in cmd:
        cmd = [part for part in cmd if part != "--dry-run"]
    return shlex.join(cmd)


def write_reports(args: argparse.Namespace, steps: List[Dict[str, Any]], status: Dict[str, Any]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = status_by_target(status)
    complete = [target for target in TARGET_ORDER if records.get(target, {}).get("complete_plausible")]
    incomplete = [target for target in TARGET_ORDER if not records.get(target, {}).get("complete_plausible")]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root),
        "source": args.source,
        "dry_run": bool(args.dry_run),
        "steps": steps,
        "complete": complete,
        "incomplete": incomplete,
        "status": status,
        "retry_commands": {target: retry_command(args, target) for target in incomplete},
    }
    (args.output_root / "priority_download_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Priority Weight Download Report",
        "",
        f"Created: {payload['created_at']}",
        f"Output root: `{args.output_root}`",
        f"Source: `{args.source}`",
        f"Dry run: `{args.dry_run}`",
        "",
        "## Commands",
        "",
    ]
    for step in steps:
        lines.append(f"- `{step['target']}` return code: `{step['returncode']}`")
        lines.append("")
        lines.append("```bash")
        lines.append(step["command"])
        lines.append("```")
        lines.append("")
    lines.extend(["## Complete Models", ""])
    lines.extend([f"- `{target}`" for target in complete] or ["- none"])
    lines.extend(["", "## Incomplete Models", ""])
    lines.extend([f"- `{target}`" for target in incomplete] or ["- none"])
    lines.extend(["", "## Next Retry Commands", ""])
    for target in incomplete:
        lines.append(f"### {target}")
        lines.append("")
        lines.append("```bash")
        lines.append(payload["retry_commands"][target])
        lines.append("```")
        lines.append("")
    (args.output_root / "priority_download_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    steps: List[Dict[str, Any]] = []
    any_failed = False
    for target in TARGET_ORDER:
        cmd = command_for(args, target)
        rc = 0 if args.dry_run else run_command(cmd)
        if args.dry_run:
            print(f"$ {shlex.join(cmd)}")
        steps.append({"target": target, "command": shlex.join(cmd), "returncode": rc})
        if rc != 0:
            any_failed = True
            if args.stop_after_first_failure or not args.continue_on_failure:
                break
    status = refresh_status(args)
    write_reports(args, steps, status)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
