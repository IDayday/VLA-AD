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
DEFAULT_LAUNCH_LOCK_FILE = Path("/mnt/project/VLA-AD/outputs/stage3_next_action_launch.lock")
STRICT_GSPO_COMMAND = (
    "cd /mnt/project/VLA-AD_stage3_algo_clean_4f3eb73 && "
    "RUN_TRAIN=1 LAUNCH_EVAL_WATCHERS=1 "
    "bash scripts/training/launch_recogdrive_stage3_grpo_gspo_2b_local_stable.sh"
)


def _resource_guard_script(gpu_list: str, max_mem_used_mb: int, max_util_pct: int) -> str:
    return f"""\
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi not found; refusing guarded Stage3 launch.' >&2
  exit 3
fi
blocked="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
  awk -F, -v gpu_list='{gpu_list}' -v max_mem={max_mem_used_mb} -v max_util={max_util_pct} '
    BEGIN {{
      n = split(gpu_list, wanted, ",");
      for (i = 1; i <= n; ++i) allow[wanted[i] + 0] = 1;
    }}
    {{
      idx = $1 + 0;
      mem = $2 + 0;
      util = $3 + 0;
      if (idx in allow && (mem > max_mem || util > max_util)) {{
        printf("gpu=%d mem=%dMB util=%d%%\\n", idx, mem, util);
      }}
    }}')"
if [[ -n "${{blocked}}" ]]; then
  echo 'Refusing strict GSPO launch because target GPUs are busy:' >&2
  echo "${{blocked}}" >&2
  exit 3
fi
"""


def _launch_lock_script(lock_file: Path) -> str:
    return f"""\
mkdir -p $(dirname {str(lock_file)!r})
exec 9>{str(lock_file)!r}
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo 'Refusing strict GSPO launch because another gated launch holds the lock: {lock_file}' >&2
    exit 4
  fi
else
  echo 'flock not found; refusing guarded Stage3 launch.' >&2
  exit 4
fi
"""


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
    launch_resource_policy: str,
    launch_gpu_list: str,
    launch_gpu_max_mem_used_mb: int,
    launch_gpu_max_util: int,
    launch_lock_file: Path,
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
        "launch_resource_policy": launch_resource_policy,
        "launch_gpu_list": launch_gpu_list,
        "launch_gpu_max_mem_used_mb": launch_gpu_max_mem_used_mb,
        "launch_gpu_max_util": launch_gpu_max_util,
        "launch_lock_file": str(launch_lock_file),
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
    parser.add_argument(
        "--launch-resource-policy",
        choices=("local_gpu_free", "none"),
        default="local_gpu_free",
        help="Resource guard for generated launch command.",
    )
    parser.add_argument("--launch-gpu-list", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--launch-gpu-max-mem-used-mb", type=int, default=2000)
    parser.add_argument("--launch-gpu-max-util", type=int, default=5)
    parser.add_argument("--launch-lock-file", type=Path, default=DEFAULT_LAUNCH_LOCK_FILE)
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
        launch_resource_policy=args.launch_resource_policy,
        launch_gpu_list=args.launch_gpu_list,
        launch_gpu_max_mem_used_mb=args.launch_gpu_max_mem_used_mb,
        launch_gpu_max_util=args.launch_gpu_max_util,
        launch_lock_file=args.launch_lock_file,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    args.command_file.parent.mkdir(parents=True, exist_ok=True)
    if decision["should_launch_now"]:
        guard = _launch_lock_script(args.launch_lock_file)
        if args.launch_resource_policy == "local_gpu_free":
            guard += _resource_guard_script(
                gpu_list=args.launch_gpu_list,
                max_mem_used_mb=args.launch_gpu_max_mem_used_mb,
                max_util_pct=args.launch_gpu_max_util,
            )
        command_text = f"#!/usr/bin/env bash\nset -euo pipefail\n{guard}{STRICT_GSPO_COMMAND}\n"
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
