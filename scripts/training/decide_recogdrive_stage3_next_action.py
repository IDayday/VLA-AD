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
DEFAULT_CURRENT_RUN = "stage3_grpo_refkl_s16_lr1e4_b2acc4_currentrepo_8gpu_20260614T191227Z"
DEFAULT_LAUNCH_LOCK_FILE = Path("/mnt/project/VLA-AD/outputs/stage3_buffer_dpo_next_action_launch.lock")
DEFAULT_BUFFER_DPO_RUN_PREFIX = "stage3_grpo_buffer_dpo_refctrl"
NEXT_EXPERIMENT_COMMAND = (
    "cd /mnt/project/VLA-AD_last_vla_dev && "
    "RUN_TRAIN=1 LAUNCH_EVAL_WATCHERS=1 "
    "bash scripts/training/launch_recogdrive_stage3_grpo_buffer_dpo_2b_local_stable.sh"
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
  echo 'Refusing buffer-guided strict GSPO launch because target GPUs are busy:' >&2
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
    echo 'Refusing buffer-guided strict GSPO launch because another gated launch holds the lock: {lock_file}' >&2
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


def _find_active_buffer_dpo_run(
    rows: list[dict[str, str]],
    *,
    run_prefix: str,
    current_run: str,
) -> dict[str, str] | None:
    for row in rows:
        run_name = row.get("run_name", "")
        if run_name == current_run or not run_name.startswith(run_prefix):
            continue
        state = row.get("training_state", "")
        if state.startswith("queued") or "waiting_for_free" in state or state == "running":
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
        "next_experiment_command": NEXT_EXPERIMENT_COMMAND,
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
        result["reason"] = "当前 run 还没有可评估 checkpoint；训练标量不能替代 navtest PDMS。"
        return result

    if ckpts > 0 and eval_rows < min_eval_rows:
        result["action"] = "wait_for_checkpoint_eval"
        result["reason"] = "已有 checkpoint 但 watcher 评估未完成；等待 PDMS/submetrics。"
        return result

    if pdms is None:
        if state in {"failed", "done", "stopped_for_fast_refkl_relaunch"}:
            result["action"] = "launch_buffer_dpo"
            result["should_launch_now"] = True
            result["reason"] = "当前路线没有可用 PDMS 且已结束/失败；下一步应切 current-repo control-aligned buffer-DPO。"
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
        result["action"] = "launch_buffer_dpo_after_current_checkpoint"
        result["should_launch_now"] = state != "running" or allow_running_launch
        if allow_running_launch:
            result["reason"] = "训练安全比例偏低；已允许并行启动 current-repo control-aligned buffer-DPO 以充分利用资源。"
        else:
            result["reason"] = "训练安全比例偏低，下一轮需要在 current-repo control 形状上验证 train-buffer preference absorption。"
        return result

    if delta <= -switch_margin:
        result["action"] = "launch_buffer_dpo_after_current_checkpoint"
        result["should_launch_now"] = state != "running" or allow_running_launch
        if allow_running_launch:
            result["reason"] = "PDMS 明显低于历史强基线；已允许并行启动 current-repo control-aligned buffer-DPO。"
        else:
            result["reason"] = "PDMS 明显低于历史强基线；下一轮切 current-repo control-aligned buffer-DPO。"
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
    parser.add_argument("--buffer-dpo-run-prefix", default=DEFAULT_BUFFER_DPO_RUN_PREFIX)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--command-file", type=Path, default=DEFAULT_COMMAND_FILE)
    args = parser.parse_args()

    rows = _read_tsv(args.summary_tsv)
    row = _find_run(rows, args.current_run)
    if row is None:
        raise SystemExit(f"Run not found in summary: {args.current_run}")

    active_buffer_dpo = _find_active_buffer_dpo_run(
        rows,
        run_prefix=args.buffer_dpo_run_prefix,
        current_run=args.current_run,
    )
    if active_buffer_dpo is not None:
        decision: dict[str, object] = {
            "run_name": args.current_run,
            "state": row.get("training_state", ""),
            "checkpoint_count": int(row.get("checkpoint_count") or 0),
            "eval_rows": int(row.get("eval_rows") or 0),
            "best_pdms": _metric(row, "best_pdms"),
            "baseline_pdms": args.baseline_pdms,
            "next_experiment_command": NEXT_EXPERIMENT_COMMAND,
            "launch_running_policy": args.launch_running_policy,
            "launch_resource_policy": args.launch_resource_policy,
            "launch_gpu_list": args.launch_gpu_list,
            "launch_gpu_max_mem_used_mb": args.launch_gpu_max_mem_used_mb,
            "launch_gpu_max_util": args.launch_gpu_max_util,
            "launch_lock_file": str(args.launch_lock_file),
            "should_launch_now": False,
            "action": "wait_for_buffer_dpo_queue",
            "reason": "Buffer-DPO 已经排队或运行；不要重复启动，等待其进入训练并观察 DPO 诊断。",
            "active_buffer_dpo_run": active_buffer_dpo.get("run_name", ""),
            "active_buffer_dpo_state": active_buffer_dpo.get("training_state", ""),
            "active_buffer_dpo_pid": active_buffer_dpo.get("train_pid", ""),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
        args.command_file.parent.mkdir(parents=True, exist_ok=True)
        command_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"echo 'Stage3 next-action gate: {decision['action']}'\n"
            f"echo 'Reason: {decision['reason']}'\n"
            f"echo 'Active Buffer-DPO run: {decision['active_buffer_dpo_run']} ({decision['active_buffer_dpo_state']})'\n"
            "exit 2\n"
        )
        args.command_file.write_text(command_text, encoding="utf-8")
        args.command_file.chmod(0o755)
        print(f"action={decision['action']}")
        print(f"should_launch_now={decision['should_launch_now']}")
        print(f"reason={decision['reason']}")
        print(f"active_buffer_dpo_run={decision['active_buffer_dpo_run']}")
        print(f"json={args.output_json}")
        print(f"command_file={args.command_file}")
        return

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
        command_text = f"#!/usr/bin/env bash\nset -euo pipefail\n{guard}{NEXT_EXPERIMENT_COMMAND}\n"
    else:
        command_text = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"echo 'Stage3 next-action gate: {decision['action']}'\n"
            f"echo 'Reason: {decision['reason']}'\n"
            "echo 'Buffer-DPO command is intentionally gated until should_launch_now=true.'\n"
            f"# {NEXT_EXPERIMENT_COMMAND}\n"
            "exit 2\n"
        )
    args.command_file.write_text(command_text, encoding="utf-8")
    args.command_file.chmod(0o755)

    print(f"action={decision['action']}")
    print(f"should_launch_now={decision['should_launch_now']}")
    print(f"reason={decision['reason']}")
    if decision["action"].startswith("launch_buffer_dpo"):
        print(f"command={NEXT_EXPERIMENT_COMMAND}")
    print(f"json={args.output_json}")
    print(f"command_file={args.command_file}")


if __name__ == "__main__":
    main()
