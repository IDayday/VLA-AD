#!/usr/bin/env python3
"""Prepare the OneVL AR Answer prompt variant for stage1 retraining.

This keeps the original OneVL AR Answer data format and targets, while applying
the prompt-only alignment points needed before using AR Answer as the stage1 VLM
for ReCogDrive-style stage2 cache generation.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "/mnt/project/onevl_navsim_data/navsim_answer_official_paths.jsonl"
DEFAULT_OUTPUT = "/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_official_paths.jsonl"

PROMPT_RE = re.compile(
    r"^<image> is the front view\. "
    r"Command: (?P<command>[^.]+)\. "
    r"Velocity: (?P<velocity>\[[^\]]+\])\. "
    r"Acceleration: (?P<acceleration>\[[^\]]+\])\. "
    r"Historical trajectory: (?P<history>.*?)\. "
    r"Output the reasoning in <think></think> and the predicted trajectory in <answer></answer>\. "
    r"For the content in <answer></answer>, Only generate.*$"
)
POINT_RE = re.compile(r"\[[^\]]+\]")
COMMAND_MAP = {
    "MOVE FORWARD": "MOVE FORWARD",
    "TURN LEFT": "TURN LEFT",
    "TURN RIGHT": "TURN RIGHT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default="")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "json"), default="auto")
    parser.add_argument("--skip-answer-validation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_answer(answer: str) -> None:
    if not (answer.startswith("<answer>") and answer.endswith("</answer>")):
        raise ValueError(f"assistant answer must be wrapped in <answer>: {answer[:120]}")
    inner = answer[len("<answer>") : -len("</answer>")].strip()
    waypoints = ast.literal_eval(f"[{inner}]")
    if len(waypoints) != 8:
        raise ValueError(f"expected 8 answer waypoints, got {len(waypoints)}")
    for waypoint in waypoints:
        if not isinstance(waypoint, list) or len(waypoint) != 3:
            raise ValueError(f"invalid waypoint shape: {waypoint!r}")
        for value in waypoint:
            if not isinstance(value, (int, float)):
                raise ValueError(f"invalid waypoint value: {waypoint!r}")


def format_point(point: list[float]) -> str:
    values = []
    for value in point:
        value = float(value)
        if abs(value) < 0.005:
            value = 0.0
        values.append(f"{value:.2f}")
    return f"[{values[0]}, {values[1]}, {values[2]}]"


def parse_history_points(history: str) -> list[str]:
    history = history.strip()
    if history.startswith("[["):
        parsed = ast.literal_eval(history)
        if not isinstance(parsed, list):
            raise ValueError(f"invalid history list: {history}")
        points = []
        for point in parsed:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"invalid history point: {point!r}")
            points.append(format_point(point))
        return points
    return POINT_RE.findall(history)


def transform_prompt(prompt: str) -> tuple[str, dict[str, Any]]:
    match = PROMPT_RE.match(prompt)
    if match is None:
        raise ValueError(f"unrecognized AR Answer prompt: {prompt[:240]}")

    command = match.group("command")
    if command not in COMMAND_MAP:
        raise ValueError(f"unsupported command: {command}")

    history_points = parse_history_points(match.group("history"))
    if len(history_points) != 3:
        raise ValueError(f"expected 3 historical trajectory points, got {len(history_points)}")

    new_command = COMMAND_MAP[command]
    history = ", ".join([*history_points, "[0.00, 0.00, 0.00]"])
    new_prompt = (
        f"<image> is the front view. Command: {new_command}. "
        f"Velocity: {match.group('velocity')}. "
        f"Acceleration: {match.group('acceleration')}. "
        f"Historical trajectory: {history}. "
        "Predict the future trajectory in <answer></answer>. "
        "For the content in <answer></answer>, only generate 8 future waypoints in pure text format: "
        "[x_1, y_1, heading_1], [x_2, y_2, heading_2], ..., [x_8, y_8, heading_8]. "
        "Each waypoint must be [x, y, heading] with exactly 2 digits after the decimal point. "
        "Separate waypoints with commas. "
        "Do not include reasoning, extra text, an extra outer list, or invalid values."
    )
    return new_prompt, {
        "command_before": command,
        "command_after": new_command,
        "history_points_before": len(history_points),
        "history_points_after": 4,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(output_path.suffix + ".report.json")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    stats: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": 0,
        "commands_before": collections.Counter(),
        "commands_after": collections.Counter(),
        "history_points_before": collections.Counter(),
        "history_points_after": collections.Counter(),
        "prompt_with_think_before": 0,
        "prompt_with_think_after": 0,
        "answer_with_think": 0,
        "valid_answer_rows": 0,
        "examples": [],
    }

    input_format = args.input_format
    if input_format == "auto":
        input_format = "jsonl" if input_path.suffix == ".jsonl" else "json"
    output_rows: list[dict[str, Any]] = []

    def iter_rows() -> list[tuple[int, dict[str, Any]]]:
        if input_format == "jsonl":
            rows = []
            with input_path.open("r", encoding="utf-8") as src:
                for line_index, line in enumerate(src):
                    if line.strip():
                        rows.append((line_index, json.loads(line)))
            return rows
        with input_path.open("r", encoding="utf-8") as src:
            loaded = json.load(src)
        if not isinstance(loaded, list):
            raise ValueError(f"expected list for JSON input: {input_path}")
        return list(enumerate(loaded))

    for line_index, row in iter_rows():
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"row {line_index}: expected at least user and assistant messages")
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError(f"row {line_index}: unexpected message roles")

            old_prompt = str(messages[0]["content"])
            new_prompt, meta = transform_prompt(old_prompt)
            answer = str(messages[1]["content"])
            if not args.skip_answer_validation:
                validate_answer(answer)

            new_messages = [dict(message) for message in messages]
            new_messages[0]["content"] = new_prompt
            new_row = dict(row)
            new_row["messages"] = new_messages

            output_rows.append(new_row)

            stats["rows"] += 1
            stats["commands_before"][meta["command_before"]] += 1
            stats["commands_after"][meta["command_after"]] += 1
            stats["history_points_before"][str(meta["history_points_before"])] += 1
            stats["history_points_after"][str(meta["history_points_after"])] += 1
            stats["prompt_with_think_before"] += int("<think>" in old_prompt)
            stats["prompt_with_think_after"] += int("<think>" in new_prompt)
            stats["answer_with_think"] += int("<think>" in answer)
            stats["valid_answer_rows"] += int(not args.skip_answer_validation)
            if len(stats["examples"]) < 3:
                stats["examples"].append(
                    {
                        "line_index": line_index,
                        "before": old_prompt,
                        "after": new_prompt,
                        "answer": answer,
                    }
                )

    with tmp_path.open("w", encoding="utf-8") as dst:
        if input_format == "jsonl":
            for row in output_rows:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(output_rows, dst, ensure_ascii=False, indent=2)

    tmp_path.replace(output_path)
    serializable_stats = {
        key: dict(value) if isinstance(value, collections.Counter) else value
        for key, value in stats.items()
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(serializable_stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
