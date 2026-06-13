#!/usr/bin/env python3
"""Refresh Stage3 summaries and optionally launch the gated next experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_WORKDIR = Path("/mnt/project/VLA-AD_stage3_algo_clean_4f3eb73")
DEFAULT_OUTPUTS_ROOT = Path("/mnt/project/VLA-AD/outputs")
DEFAULT_CURRENT_RUN = "stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run(cmd: list[str], cwd: Path, log_path: Path, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    started = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, check=False)
    elapsed = time.time() - started
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc()}] cmd={' '.join(cmd)}\n")
        log.write(f"rc={result.returncode} elapsed_sec={elapsed:.1f}\n")
        if result.stdout:
            log.write("stdout:\n")
            log.write(result.stdout)
            if not result.stdout.endswith("\n"):
                log.write("\n")
        if result.stderr:
            log.write("stderr:\n")
            log.write(result.stderr)
            if not result.stderr.endswith("\n"):
                log.write("\n")
    return result


def _script(workdir: Path, name: str) -> str:
    return str(workdir / "scripts" / "training" / name)


def _refresh_once(args: argparse.Namespace) -> dict:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n=== Stage3 supervisor iteration {_utc()} ===\n")

    summary_cmd = [
        sys.executable,
        _script(args.workdir, "summarize_recogdrive_stage3_runs.py"),
        "--outputs-root",
        str(args.outputs_root),
        "--max-runs",
        str(args.max_runs),
        "--print-limit",
        str(args.print_limit),
        "--output-tsv",
        str(args.summary_tsv),
        "--output-json",
        str(args.summary_json),
    ]
    result = _run(summary_cmd, cwd=args.workdir, log_path=args.log_file, timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(f"summary command failed rc={result.returncode}; see {args.log_file}")

    report_cmd = [
        sys.executable,
        _script(args.workdir, "report_recogdrive_stage3_algorithm_status.py"),
        "--summary-tsv",
        str(args.summary_tsv),
        "--output-md",
        str(args.report_md),
        "--output-json",
        str(args.report_json),
        "--baseline-pdms",
        str(args.baseline_pdms),
        "--original-stage3-pdms",
        str(args.original_stage3_pdms),
        "--active-event-age-sec",
        str(args.active_event_age_sec),
    ]
    result = _run(report_cmd, cwd=args.workdir, log_path=args.log_file, timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(f"report command failed rc={result.returncode}; see {args.log_file}")

    decision_cmd = [
        sys.executable,
        _script(args.workdir, "decide_recogdrive_stage3_next_action.py"),
        "--summary-tsv",
        str(args.summary_tsv),
        "--current-run",
        args.current_run,
        "--baseline-pdms",
        str(args.baseline_pdms),
        "--continue-margin",
        str(args.continue_margin),
        "--switch-margin",
        str(args.switch_margin),
        "--min-safe-ratio",
        str(args.min_safe_ratio),
        "--min-eval-rows",
        str(args.min_eval_rows),
        "--output-json",
        str(args.decision_json),
        "--command-file",
        str(args.command_file),
    ]
    result = _run(decision_cmd, cwd=args.workdir, log_path=args.log_file, timeout=args.command_timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(f"decision command failed rc={result.returncode}; see {args.log_file}")

    decision = json.loads(args.decision_json.read_text(encoding="utf-8"))
    status = {
        "timestamp_utc": _utc(),
        "decision": decision,
        "summary_tsv": str(args.summary_tsv),
        "report_md": str(args.report_md),
        "command_file": str(args.command_file),
        "allow_launch": bool(args.allow_launch),
    }
    args.status_json.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    if decision.get("should_launch_now"):
        if args.allow_launch:
            with args.log_file.open("a", encoding="utf-8") as log:
                log.write(f"[{_utc()}] gate allows launch; executing {args.command_file}\n")
            result = _run(["bash", str(args.command_file)], cwd=args.workdir, log_path=args.log_file, timeout=None)
            status["launch_rc"] = result.returncode
            args.status_json.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"launch command failed rc={result.returncode}; see {args.log_file}")
        else:
            with args.log_file.open("a", encoding="utf-8") as log:
                log.write(f"[{_utc()}] gate allows launch, but --allow-launch is false; dry-run only\n")
    else:
        with args.log_file.open("a", encoding="utf-8") as log:
            log.write(
                f"[{_utc()}] no launch: action={decision.get('action')} "
                f"reason={decision.get('reason')}\n"
            )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--current-run", default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_runs_summary_latest.tsv")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_runs_summary_latest.json")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_algorithm_status_latest.md")
    parser.add_argument("--report-json", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_algorithm_status_latest.json")
    parser.add_argument("--decision-json", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_next_action_latest.json")
    parser.add_argument("--command-file", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_next_action_command.sh")
    parser.add_argument("--status-json", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_supervisor_status_latest.json")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_OUTPUTS_ROOT / "stage3_supervisor.log")
    parser.add_argument("--baseline-pdms", type=float, default=0.9061843202874436)
    parser.add_argument("--original-stage3-pdms", type=float, default=0.9055)
    parser.add_argument("--continue-margin", type=float, default=0.003)
    parser.add_argument("--switch-margin", type=float, default=0.015)
    parser.add_argument("--min-safe-ratio", type=float, default=0.88)
    parser.add_argument("--min-eval-rows", type=int, default=1)
    parser.add_argument("--active-event-age-sec", type=float, default=1800.0)
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--print-limit", type=int, default=12)
    parser.add_argument("--interval-sec", type=float, default=300.0)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means run forever unless --once is set.")
    parser.add_argument("--command-timeout-sec", type=float, default=300.0)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not args.workdir.exists():
        raise SystemExit(f"workdir does not exist: {args.workdir}")

    iteration = 0
    while True:
        iteration += 1
        status = _refresh_once(args)
        decision = status["decision"]
        print(
            f"iteration={iteration} action={decision.get('action')} "
            f"should_launch_now={decision.get('should_launch_now')} "
            f"allow_launch={args.allow_launch}"
        )
        print(f"reason={decision.get('reason')}")
        print(f"status_json={args.status_json}")
        if args.once or (args.max_iterations > 0 and iteration >= args.max_iterations):
            break
        if args.allow_launch and decision.get("should_launch_now"):
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
