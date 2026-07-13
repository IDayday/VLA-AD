#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample, write_json  # noqa: E402
from navsim.agents.recogdrive.trajectory_text_replay import try_parse_trajectory_answer  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records, load_path_index  # noqa: E402


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    jepa_index = load_path_index(args.jepa_cache_root) if args.jepa_cache_root else {}
    vggt_index = load_path_index(args.vggt_cache_root) if args.vggt_cache_root else {}
    total = 0
    prompt_ok = 0
    answer_ok = 0
    parse_ok = 0
    image_exists = 0
    official = 0
    teacher_intersection = 0
    bad_examples = []
    for _, sample_path, record in iter_indexed_records(args.replay_cache_root, max_records=args.max_samples):
        total += 1
        token = str(record.get("sample_token") or sample_path.stem)
        try:
            payload = load_sample(sample_path)
            prompt = payload.get("prompt")
            answer_text = payload.get("answer_text")
            if isinstance(prompt, str) and prompt.strip():
                prompt_ok += 1
            if isinstance(answer_text, str) and answer_text.strip():
                answer_ok += 1
            parsed = try_parse_trajectory_answer(answer_text if isinstance(answer_text, str) else "")
            if parsed.parse_ok:
                parse_ok += 1
            image_path = payload.get("image_path") or payload.get("image_path_tensor")
            if isinstance(image_path, str) and Path(image_path).is_file():
                image_exists += 1
            if bool(payload.get("official_recogdrive_stage1", False)):
                official += 1
            if (not jepa_index or token in jepa_index) and (not vggt_index or token in vggt_index):
                teacher_intersection += 1
            if len(bad_examples) < 8 and (not parsed.parse_ok or not prompt or not answer_text):
                bad_examples.append({"sample_token": token, "path": str(sample_path), "parse_error": parsed.error})
        except Exception as exc:  # pragma: no cover - audit should keep reporting bad records.
            if len(bad_examples) < 8:
                bad_examples.append({"sample_token": token, "path": str(sample_path), "error": repr(exc)})
    ratios = {
        "prompt_ok": prompt_ok / max(1, total),
        "answer_ok": answer_ok / max(1, total),
        "parse_ok": parse_ok / max(1, total),
        "image_exists": image_exists / max(1, total),
        "official_recogdrive_stage1": official / max(1, total),
        "teacher_intersection": teacher_intersection / max(1, total),
    }
    ok = bool(
        total > 0
        and ratios["prompt_ok"] >= 0.99
        and ratios["answer_ok"] >= 0.99
        and ratios["parse_ok"] >= float(args.min_parse_ok)
        and (not args.require_image_exists or ratios["image_exists"] >= 0.99)
        and (not jepa_index and not vggt_index or ratios["teacher_intersection"] >= float(args.min_teacher_intersection))
    )
    return {
        "ok": ok,
        "replay_cache_root": str(args.replay_cache_root),
        "total": total,
        "counts": {
            "prompt_ok": prompt_ok,
            "answer_ok": answer_ok,
            "parse_ok": parse_ok,
            "image_exists": image_exists,
            "official_recogdrive_stage1": official,
            "teacher_intersection": teacher_intersection,
        },
        "ratios": ratios,
        "bad_examples": bad_examples,
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Stage1 Replay Data Audit",
        "",
        f"- ok: `{result['ok']}`",
        f"- replay_cache_root: `{result['replay_cache_root']}`",
        f"- total: `{result['total']}`",
        "",
        "| ratio | value |",
        "| --- | ---: |",
    ]
    for key, value in sorted(result["ratios"].items()):
        lines.append(f"| `{key}` | {value:.6f} |")
    if result["bad_examples"]:
        lines.extend(["", "## Bad Examples", "", "```json", json.dumps(result["bad_examples"], indent=2, sort_keys=True), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit indexed ReCogDrive Stage1 replay cache.")
    parser.add_argument("--replay-cache-root", type=Path, required=True)
    parser.add_argument("--jepa-cache-root", type=Path, default=None)
    parser.add_argument("--vggt-cache-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-parse-ok", type=float, default=0.99)
    parser.add_argument("--min-teacher-intersection", type=float, default=0.99)
    parser.add_argument("--require-image-exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit(args)
    write_json(args.output_json, result)
    write_markdown(args.output_md, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
