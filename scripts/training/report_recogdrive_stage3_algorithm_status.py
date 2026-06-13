#!/usr/bin/env python3
"""Create a compact Stage3 algorithm comparison report from run summaries."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path


DEFAULT_SUMMARY_TSV = Path("/mnt/project/VLA-AD/outputs/stage3_runs_summary_latest.tsv")
DEFAULT_OUTPUT_MD = Path("/mnt/project/VLA-AD/outputs/stage3_algorithm_status_latest.md")
DEFAULT_OUTPUT_JSON = Path("/mnt/project/VLA-AD/outputs/stage3_algorithm_status_latest.json")
METRIC_FIELDS = ("best_pdms", "best_nc", "best_dac", "best_ttc", "best_ep", "best_comfort", "best_ddc", "best_tlc")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _method_family(run_name: str) -> str:
    name = run_name.lower()
    if "gspo" in name:
        return "Strict GSPO"
    if "grpo_refkl" in name:
        return "GRPO ref-KL"
    if "safe_diffgrpo" in name:
        return "Safe DiffGRPO"
    if "awac" in name and "dpo" in name:
        return "AWAC/IQL + DPO"
    if "awac" in name:
        return "AWAC/IQL"
    if "stage3_rl" in name:
        return "Stage3 RL"
    return "Other"


def _sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    running = 1 if row.get("training_state") == "running" else 0
    pdms = _float_or_none(row.get("best_pdms")) or -1.0
    mtime = _float_or_none(row.get("mtime_epoch")) or 0.0
    return running, pdms, mtime


def _best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = [row for row in rows if _float_or_none(row.get("best_pdms")) is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: _float_or_none(row.get("best_pdms")) or -1.0)


def _is_active_running(row: dict[str, str], active_event_age_sec: float) -> bool:
    if row.get("training_state") != "running":
        return False
    event_age = _float_or_none(row.get("event_age_sec"))
    if row.get("event_file") and event_age is not None:
        return event_age <= active_event_age_sec
    return False


def _current_rows(rows: list[dict[str, str]], active_event_age_sec: float) -> list[dict[str, str]]:
    current = [row for row in rows if _is_active_running(row, active_event_age_sec)]
    return sorted(current, key=lambda row: _sort_key(row), reverse=True)


def _stale_running_rows(rows: list[dict[str, str]], active_event_age_sec: float) -> list[dict[str, str]]:
    stale = [row for row in rows if row.get("training_state") == "running" and not _is_active_running(row, active_event_age_sec)]
    return sorted(stale, key=lambda row: row.get("run_name", ""))


def _format_float(value: object, digits: int = 6) -> str:
    number = _float_or_none(value)
    return "" if number is None else f"{number:.{digits}f}"


def _delta(value: object, baseline: float) -> str:
    number = _float_or_none(value)
    if number is None:
        return ""
    return f"{number - baseline:+.6f}"


def _decision_for_current(row: dict[str, str], baseline_pdms: float, launch_margin: float) -> str:
    state = row.get("training_state", "")
    ckpts = int(row.get("checkpoint_count") or 0)
    eval_rows = int(row.get("eval_rows") or 0)
    pdms = _float_or_none(row.get("best_pdms"))
    safe_ratio = _float_or_none(row.get("safe_ratio"))
    group_std = _float_or_none(row.get("group_reward_std"))
    all_unsafe = _float_or_none(row.get("all_unsafe_group_ratio"))

    if state == "running" and ckpts == 0:
        return "等待 epoch0 checkpoint；当前只能看训练 reward/safety，不能判定 navtest PDMS。"
    if state == "running" and eval_rows == 0:
        return "等待 watcher 完成 checkpoint PDMS 评估。"
    if pdms is None:
        return "缺少 PDMS，先补评估或检查 watcher 输出。"
    if pdms >= baseline_pdms:
        return "达到或超过历史强基线，继续当前路线并优先观察后续 epoch 是否保持。"
    if pdms >= baseline_pdms - launch_margin:
        return "接近历史强基线，继续至少一个 epoch，同时检查 DDC/TTC 是否退化。"
    if all_unsafe is not None and all_unsafe > 0.10:
        return "安全组退化明显，下一轮应加强 safety guard/reference trust region。"
    if safe_ratio is not None and safe_ratio < 0.88:
        return "训练安全比例偏低，下一轮优先收紧安全 shaping 或降低更新强度。"
    if group_std is not None and group_std < 0.05:
        return "组内 reward 区分度低，下一轮应提高采样多样性或改用更强偏好信号。"
    return "明显低于强基线，下一轮优先尝试 strict GSPO/偏好学习变体，而不是继续 AWAC 调参。"


def build_report(
    rows: list[dict[str, str]],
    baseline_pdms: float,
    original_pdms: float,
    launch_margin: float,
    active_event_age_sec: float,
) -> dict:
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[_method_family(row.get("run_name", ""))].append(row)

    best_overall = _best_row(rows)
    current = _current_rows(rows, active_event_age_sec)
    stale_running = _stale_running_rows(rows, active_event_age_sec)
    family_best = {}
    for family, family_rows in families.items():
        best = _best_row(family_rows)
        if best is not None:
            family_best[family] = best

    current_decisions = [
        {
            "run_name": row.get("run_name", ""),
            "decision": _decision_for_current(row, baseline_pdms=baseline_pdms, launch_margin=launch_margin),
        }
        for row in current
    ]

    return {
        "generated_at_utc": _utc(),
        "baseline_pdms": baseline_pdms,
        "original_stage3_pdms": original_pdms,
        "best_overall": best_overall or {},
        "current_runs": current,
        "stale_running_runs": stale_running,
        "family_best": family_best,
        "current_decisions": current_decisions,
        "num_runs": len(rows),
    }


def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def write_markdown(path: Path, report: dict, top_rows: list[dict[str, str]], baseline_pdms: float) -> None:
    lines: list[str] = []
    lines.append("# ReCogDrive Stage3 Algorithm Status")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at_utc']}`")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append(f"- Historical strong Stage3/Safe DiffGRPO best PDMS: `{baseline_pdms:.6f}`")
    lines.append(f"- Original Stage3 reference PDMS: `{report['original_stage3_pdms']:.6f}`")
    best = report.get("best_overall") or {}
    if best:
        lines.append(
            f"- Best discovered run: `{best.get('run_name', '')}` / `{best.get('best_checkpoint_id', '')}` "
            f"PDMS `{_format_float(best.get('best_pdms'))}`"
        )
    lines.append("")

    lines.append("## Current Running Runs")
    lines.append("")
    lines.append(_table_row(["run", "step", "ckpts", "reward", "safe", "eval rows", "best PDMS", "decision"]))
    lines.append(_table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]))
    current = report.get("current_runs") or []
    decisions = {item["run_name"]: item["decision"] for item in report.get("current_decisions", [])}
    if not current:
        lines.append(_table_row(["", "", "", "", "", "", "", "no running Stage3 run found"]))
    for row in current:
        lines.append(
            _table_row(
                [
                    row.get("run_name", ""),
                    row.get("latest_step", ""),
                    row.get("checkpoint_count", ""),
                    _format_float(row.get("train_reward")),
                    _format_float(row.get("safe_ratio")),
                    row.get("eval_rows", ""),
                    _format_float(row.get("best_pdms")),
                    decisions.get(row.get("run_name", ""), ""),
                ]
            )
        )
    lines.append("")

    stale_running = report.get("stale_running_runs") or []
    if stale_running:
        lines.append("## Stale Running Records")
        lines.append("")
        lines.append("These directories still report `running`, but their TensorBoard event file is stale or missing.")
        lines.append("")
        lines.append(_table_row(["run", "step", "event age sec", "ckpts", "best PDMS"]))
        lines.append(_table_row(["---", "---:", "---:", "---:", "---:"]))
        for row in stale_running[:12]:
            lines.append(
                _table_row(
                    [
                        row.get("run_name", ""),
                        row.get("latest_step", ""),
                        row.get("event_age_sec", ""),
                        row.get("checkpoint_count", ""),
                        _format_float(row.get("best_pdms")),
                    ]
                )
            )
        lines.append("")

    lines.append("## Best By Method Family")
    lines.append("")
    lines.append(_table_row(["family", "run", "ckpt", "PDMS", "delta vs baseline", "DDC", "TTC", "EP"]))
    lines.append(_table_row(["---", "---", "---", "---:", "---:", "---:", "---:", "---:"]))
    for family, row in sorted((report.get("family_best") or {}).items()):
        lines.append(
            _table_row(
                [
                    family,
                    row.get("run_name", ""),
                    row.get("best_checkpoint_id", ""),
                    _format_float(row.get("best_pdms")),
                    _delta(row.get("best_pdms"), baseline_pdms),
                    _format_float(row.get("best_ddc")),
                    _format_float(row.get("best_ttc")),
                    _format_float(row.get("best_ep")),
                ]
            )
        )
    lines.append("")

    lines.append("## Top Evaluated Runs")
    lines.append("")
    lines.append(_table_row(["rank", "run", "family", "ckpt", "PDMS", "delta vs baseline", "NC", "DAC", "TTC", "EP", "DDC"]))
    lines.append(_table_row(["---:", "---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    for idx, row in enumerate(top_rows, start=1):
        lines.append(
            _table_row(
                [
                    str(idx),
                    row.get("run_name", ""),
                    _method_family(row.get("run_name", "")),
                    row.get("best_checkpoint_id", ""),
                    _format_float(row.get("best_pdms")),
                    _delta(row.get("best_pdms"), baseline_pdms),
                    _format_float(row.get("best_nc")),
                    _format_float(row.get("best_dac")),
                    _format_float(row.get("best_ttc")),
                    _format_float(row.get("best_ep")),
                    _format_float(row.get("best_ddc")),
                ]
            )
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- AWAC/IQL and AWAC+DPO runs often show high training reward but much lower navtest PDMS, so the next change should not be another pure AWAC weight tweak.")
    lines.append("- Current GRPO ref-KL run still has no checkpoint evaluation; wait for epoch0 PDMS before deciding whether to continue or switch.")
    lines.append("- If current GRPO is clearly below the historical strong baseline, the prepared strict GSPO path is the next most grounded variant because it controls policy ratio drift instead of relying on offline targets.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-tsv", type=Path, default=DEFAULT_SUMMARY_TSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--baseline-pdms", type=float, default=0.9061843202874436)
    parser.add_argument("--original-stage3-pdms", type=float, default=0.9055)
    parser.add_argument("--launch-margin", type=float, default=0.01)
    parser.add_argument("--active-event-age-sec", type=float, default=1800.0)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows = _read_tsv(args.summary_tsv)
    report = build_report(
        rows,
        baseline_pdms=args.baseline_pdms,
        original_pdms=args.original_stage3_pdms,
        launch_margin=args.launch_margin,
        active_event_age_sec=args.active_event_age_sec,
    )
    top_rows = sorted(
        [row for row in rows if _float_or_none(row.get("best_pdms")) is not None],
        key=lambda row: _float_or_none(row.get("best_pdms")) or -1.0,
        reverse=True,
    )[: args.top_k]

    write_markdown(args.output_md, report, top_rows, baseline_pdms=args.baseline_pdms)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"report_md={args.output_md}")
    print(f"report_json={args.output_json}")
    if report.get("current_decisions"):
        for item in report["current_decisions"]:
            print(f"current_decision[{item['run_name']}]={item['decision']}")


if __name__ == "__main__":
    main()
