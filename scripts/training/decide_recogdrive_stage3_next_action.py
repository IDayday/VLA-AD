#!/usr/bin/env python3
"""Decide the next ReCogDrive Stage3 experiment action from current evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SUMMARY_TSV = Path("/mnt/project/VLA-AD/outputs/stage3_runs_summary_latest.tsv")
DEFAULT_OUTPUT_JSON = Path("/mnt/project/VLA-AD/outputs/stage3_next_action_latest.json")
DEFAULT_COMMAND_FILE = Path("/mnt/project/VLA-AD/outputs/stage3_next_action_command.sh")
DEFAULT_CURRENT_RUN = "stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z"
STRICT_GSPO_COMMAND = (
    "cd /mnt/project/VLA-AD_stage3_algo_clean_4f3eb73 && "
    "RUN_TRAIN=1 LAUNCH_EVAL_WATCHERS=1 "
    "bash scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh"
)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _find_run(rows: list[dict[str, str]], run_name: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("run_name") == run_name:
            return row
    return None


def _metric(row: dict[str, str], field: str) -> float | None:
    return _float_or_none(row.get(field))


def decide_next_action(
    row: dict[str, str],
    *,
    baseline_pdms: float,
    continue_margin: float,
    switch_margin: float,
    min_safe_ratio: float,
    min_eval_rows: int,
    launch_running_policy: str,
) -> dict[str, object]:
    state = row.get("training_state", "")
    ckpts = int(row.get("checkpoint_count") or 0)
    eval_rows = int(row.get("eval_rows") or 0)
    pdms = _metric(row, "best_pdms")
    safe_ratio = _metric(row, "safe_ratio")
    ddc = _metric(row, "best_ddc")
    ttc = _metric(row, "best_ttc")
    ep = _metric(row, "best_ep")

    result: dict[str, object] = {
        "run_name": row.get("run_name", ""),
        "state": state,
        "checkpoint_count": ckpts,
        "eval_rows": eval_rows,
        "best_pdms": pdms,
        "safe_ratio": safe_ratio,
        "baseline_pdms": baseline_pdms,
        "strict_gspo_command": STRICT_GSPO_COMMAND,
        "launch_running_policy": launch_running_policy,
        "should_launch_now": False,
        "action": "monitor",
        "reason": "",
    }

    if state == "running" and ckpts == 0:
        result["action"] = "wait_for_first_checkpoint"
        result["reason"] = "当前 run 还没有 epoch checkpoint；训练标量不能替代 navtest PDMS。"
        return result

    if ckpts > 0 and eval_rows < min_eval_rows:
        result["action"] = "wait_for_checkpoint_eval"
        result["reason"] = "已有 checkpoint 但 watcher 评估未完成；等待 PDMS/submetrics。"
        return result

    if pdms is None:
        if state in {"failed", "done", "stopped_for_fast_refkl_relaunch"}:
            result["action"] = "launch_strict_gspo"
            result["should_launch_now"] = True
            result["reason"] = "当前路线没有可用 PDMS 且已结束/失败；下一步应切 strict GSPO。"
            return result
        result["action"] = "wait_for_pdms"
        result["reason"] = "缺少可比较 PDMS；先补齐评估。"
        return result

    delta = pdms - baseline_pdms
    result["delta_vs_baseline"] = delta
    result["best_ddc"] = ddc
    result["best_ttc"] = ttc
    result["best_ep"] = ep

    if delta >= -continue_margin:
        result["action"] = "continue_current"
        result["reason"] = "PDMS 已接近或达到历史强基线；继续当前 run 并观察后续 epoch。"
        return result

    allow_running_launch = state == "running" and launch_running_policy == "allow_after_eval"

    if safe_ratio is not None and safe_ratio < min_safe_ratio:
        result["action"] = "launch_strict_gspo_after_current_checkpoint"
        result["should_launch_now"] = state != "running" or allow_running_launch
        if allow_running_launch:
            result["reason"] = "训练安全比例偏低；已允许并行启动 strict GSPO 以充分利用资源。"
        else:
            result["reason"] = "训练安全比例偏低，下一轮需要更强 trust region 与行为策略 ratio 控制。"
        return result

    if delta <= -switch_margin:
        result["action"] = "launch_strict_gspo_after_current_checkpoint"
        result["should_launch_now"] = state != "running" or allow_running_launch
        if allow_running_launch:
            result["reason"] = "PDMS 明显低于历史强基线；已允许并行启动 strict GSPO。"
        else:
            result["reason"] = "PDMS 明显低于历史强基线；下一轮切 strict GSPO。"
        return result

    result["action"] = "continue_one_more_epoch_then_compare"
    result["reason"] = "PDMS 低于强基线但未达到强切换阈值；再观察一个 epoch。"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--current-run", default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--baseline-pdms", type=float, default=0.9061843202874436)
    parser.add_argument("--continue-margin", type=float, default=0.003)
    parser.add_argument("--switch-margin", type=float, default=0.015)
    parser.add_argument("--min-safe-ratio", type=float, default=0.88)
    parser.add_argument("--min-eval-rows", type=int, default=1)
    parser.add_argument(
        "--launch-running-policy",
        choices=("wait_until_finished", "allow_after_eval"),
        default="wait_until_finished",
        help="Whether a poor evaluated checkpoint may trigger a parallel strict-GSPO launch while the current run is still running.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND_FILE)
    args = parser.parse_args()

    rows = _read_tsv(args.summary_tsv)
    row = _find_run(rows, args.current_run)
    if row is None:
        raise SystemExit(f"Run not found in summary: {args.current_run}")

    decision = decide_next_action(
        row,
        baseline_pdms=args.baseline_pdms,
        continue_margin=args.continue_margin,
        switch_margin=args.switch_margin,
        min_safe_ratio=args.min_safe_ratio,
        min_eval_rows=args.min_eval_rows,
        launch_running_policy=args.launch_running_policy,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    if decision["should_launch_now"]:
        command_text = f"#!/usr/bin/env bash\nset -euo pipefail\n{STRICT_GSPO_COMMAND}\n"
    else:
        command_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"echo 'Stage3 next-action gate: {decision['action']}'\n"
            f"echo 'Reason: {decision['reason']}'\n"
            "echo 'Strict GSPO command is intentionally gated until should_launch_now=true.'\n"
            f"# {STRICT_GSPO_COMMAND}\n"
            "exit 2\n"
        )
    args.command_file.write_text(command_text, encoding="utf-8")
    args.command_file.chmod(0o755)

    print(f"action={decision['action']}")
    print(f"should_launch_now={decision['should_launch_now']}")
    print(f"reason={decision['reason']}")
    if decision["action"].startswith("launch_strict_gspo"):
        print(f"command={STRICT_GSPO_COMMAND}")
    print(f"json={args.output_json}")
    print(f"command_file={args.command_file}")


if __name__ == "__main__":
    main()
