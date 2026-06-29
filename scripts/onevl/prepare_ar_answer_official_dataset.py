#!/usr/bin/env python3
"""Prepare and validate an AR Answer NAVSIM jsonl in OneVL demo path style."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PUBLIC_PREFIX = "/mnt/public_data/navsim/trainval_all/trainval_sensor_blobs/trainval/"
LOCAL_PREFIX = "/mnt/navsim/trainval_sensor_blobs/trainval/"
OFFICIAL_PREFIX = "navsim_v1.1_all/dataset/sensor_blobs/trainval/"
ANSWER_RE = re.compile(r"^<answer>(?P<body>.*)</answer>$", re.S)
WAYPOINT_RE = re.compile(
    r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-cwd",
        type=Path,
        default=Path("/mnt/project/OneVL_training"),
        help="Directory from which the official training script is launched.",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=Path("/mnt/project/OneVL_training/demo_data/navsim/navsim_answer_demo100.jsonl"),
    )
    return parser.parse_args()


def official_image_path(path: str) -> str:
    if path.startswith(PUBLIC_PREFIX):
        return OFFICIAL_PREFIX + path[len(PUBLIC_PREFIX):]
    if path.startswith(LOCAL_PREFIX):
        return OFFICIAL_PREFIX + path[len(LOCAL_PREFIX):]
    if path.startswith(OFFICIAL_PREFIX):
        return path
    raise ValueError(f"unsupported image path prefix: {path}")


def answer_has_eight_waypoints(text: str) -> bool:
    match = ANSWER_RE.match(text)
    if not match:
        return False
    return len(WAYPOINT_RE.findall(match.group("body"))) == 8


def demo_instruction_suffix(demo_path: Path) -> str:
    obj = json.loads(demo_path.read_text(encoding="utf-8").splitlines()[0])
    prompt = obj["messages"][0]["content"]
    marker = "Output the reasoning in <think></think>"
    return prompt[prompt.index(marker):]


def main() -> None:
    args = parse_args()
    suffix = demo_instruction_suffix(args.demo)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "training_cwd": str(args.training_cwd),
        "rows": 0,
        "written": 0,
        "bad_roles": 0,
        "bad_prompt_suffix": 0,
        "bad_answer_format": 0,
        "cot_in_answer": 0,
        "latent_in_answer": 0,
        "bad_image_count": 0,
        "bad_image_prefix": 0,
        "missing_images": 0,
        "keys_seen": {},
    }

    with args.input.open(encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            stats["rows"] += 1
            obj = json.loads(line)
            key = ",".join(sorted(obj.keys()))
            stats["keys_seen"][key] = stats["keys_seen"].get(key, 0) + 1

            messages = obj.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                stats["bad_roles"] += 1
                continue

            user_text = messages[0].get("content", "")
            if suffix not in user_text:
                stats["bad_prompt_suffix"] += 1

            answer_text = messages[1].get("content", "")
            if "<think>" in answer_text or "</think>" in answer_text:
                stats["cot_in_answer"] += 1
            if "<|latent" in answer_text:
                stats["latent_in_answer"] += 1
            if not answer_has_eight_waypoints(answer_text):
                stats["bad_answer_format"] += 1

            images = obj.get("images")
            if not isinstance(images, list) or len(images) != 1:
                stats["bad_image_count"] += 1
                continue
            try:
                rel_image = official_image_path(images[0])
            except ValueError:
                stats["bad_image_prefix"] += 1
                continue
            if not (args.training_cwd / rel_image).exists():
                stats["missing_images"] += 1

            # Keep only the fields used by the official qwen3_vl AR Answer SFT path.
            out_obj = {"messages": messages, "images": [rel_image]}
            dst.write(json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written"] += 1

    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if any(
        stats[key]
        for key in (
            "bad_roles",
            "bad_prompt_suffix",
            "bad_answer_format",
            "cot_in_answer",
            "latent_in_answer",
            "bad_image_count",
            "bad_image_prefix",
            "missing_images",
        )
    ):
        raise SystemExit(f"dataset validation failed; see {report_path}")


if __name__ == "__main__":
    main()
