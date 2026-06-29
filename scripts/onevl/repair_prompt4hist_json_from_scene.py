#!/usr/bin/env python3
"""Repair/check OneVL prompt4hist JSON prompts against NAVSIM scene tensors.

This is a dataset preparation utility. It may use NAVSIM SceneLoader to create a
corrected JSON/JSONL file, but the stage2 cache bridge can then build planner
tensors from JSON text only.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.onevl.bridge_ar_answer_to_recogdrive_dit import (  # noqa: E402
    build_qwen_prompt_from_scene_tensors,
    build_scene_mapping,
    extract_navsim_planner_tensors,
    parse_prompt_fields,
    select_current_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--navsim-log-path", required=True, type=Path)
    parser.add_argument("--mode", choices=("repair", "check"), default="repair")
    parser.add_argument(
        "--current-image-policy",
        choices=("first", "last", "single_or_last", "strict_single"),
        default="single_or_last",
    )
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows, "jsonl"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"expected list in JSON file: {path}")
    return loaded, "json"


def write_rows(path: Path, rows: list[dict[str, Any]], file_format: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if file_format == "jsonl":
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def command_or_error(prompt: str) -> str:
    try:
        return str(parse_prompt_fields(prompt)["command"])
    except Exception as exc:  # noqa: BLE001 - report parse failures.
        return f"PARSE_ERROR:{exc}"


def main() -> int:
    args = parse_args()
    rows, file_format = load_rows(args.input)
    selected_rows = rows if args.max_samples <= 0 else rows[: args.max_samples]
    scene_mapping = build_scene_mapping(
        selected_rows,
        args.navsim_log_path,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        current_image_policy=args.current_image_policy,
    )

    output_rows: list[dict[str, Any]] = []
    command_before = collections.Counter()
    command_after = collections.Counter()
    mismatch_examples: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    changed = 0

    for line_index, row in enumerate(rows):
        if args.max_samples > 0 and line_index >= args.max_samples:
            if args.mode == "repair":
                output_rows.append(row)
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {line_index}: missing messages")
        old_prompt = str(messages[0].get("content", ""))
        before_command = command_or_error(old_prompt)
        command_before[before_command] += 1

        current_image = select_current_image(row, args.current_image_policy)
        scene_item = scene_mapping[current_image]
        features, _targets, _meta = extract_navsim_planner_tensors(
            row,
            scene_mapping,
            num_history_frames=args.num_history_frames,
            action_horizon=args.action_horizon,
            use_answer_target=False,
            current_image_policy=args.current_image_policy,
            target_index=None,
            stage2_support_mode="single_target",
        )
        new_prompt = build_qwen_prompt_from_scene_tensors(
            features["history_trajectory"],
            features["high_command_one_hot"],
            features["status_feature"],
            include_control_convention=False,
        )
        after_command = command_or_error(new_prompt)
        command_after[after_command] += 1
        if before_command.startswith("PARSE_ERROR"):
            parse_errors.append({"line_index": line_index, "error": before_command, "image": current_image})
        if old_prompt != new_prompt:
            changed += 1
            if len(mismatch_examples) < 20:
                mismatch_examples.append(
                    {
                        "line_index": line_index,
                        "image": current_image,
                        "token": str(scene_item["token"]),
                        "command_before": before_command,
                        "command_after": after_command,
                        "prompt_before": old_prompt,
                        "prompt_after": new_prompt,
                    }
                )

        if args.mode == "repair":
            new_messages = [dict(message) for message in messages]
            new_messages[0]["content"] = new_prompt
            new_row = dict(row)
            new_row["messages"] = new_messages
            output_rows.append(new_row)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "mode": args.mode,
        "file_format": file_format,
        "navsim_log_path": str(args.navsim_log_path),
        "current_image_policy": args.current_image_policy,
        "rows": len(rows),
        "checked_rows": len(selected_rows),
        "changed_rows": changed,
        "commands_before": dict(command_before),
        "commands_after": dict(command_after),
        "parse_errors": parse_errors[:50],
        "mismatch_examples": mismatch_examples,
    }
    report_path = args.report or args.output.with_suffix(args.output.suffix + ".repair_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "repair":
        write_rows(args.output, output_rows, file_format, args.overwrite)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
