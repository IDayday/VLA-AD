#!/usr/bin/env python3
"""Load real official B2D samples through the exact InternVL Stage1 data path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import AutoTokenizer

from internvl.train.constants import (
    BACK_LEFT_VIEW_TOKEN,
    BACK_RIGHT_VIEW_TOKEN,
    BACK_VIEW_TOKEN,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    FRONT_LEFT_VIEW_TOKEN,
    FRONT_RIGHT_VIEW_TOKEN,
    FRONT_VIEW_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    LOC_END_TOKEN,
    LOC_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl.train.internvl_chat_finetune import LazySupervisedDataset


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--max-dynamic-patch", type=int, default=16)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def _special_tokens() -> list[str]:
    return [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        LOC_START_TOKEN,
        LOC_END_TOKEN,
        FRONT_VIEW_TOKEN,
        FRONT_LEFT_VIEW_TOKEN,
        FRONT_RIGHT_VIEW_TOKEN,
        BACK_LEFT_VIEW_TOKEN,
        BACK_RIGHT_VIEW_TOKEN,
        BACK_VIEW_TOKEN,
    ]


def run_preflight(
    meta_path: Path,
    base_vlm: Path,
    max_seq_length: int,
    max_dynamic_patch: int,
) -> Dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        base_vlm,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = int(max_seq_length)
    tokenizer.add_tokens(_special_tokens(), special_tokens=True)
    image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    checks: list[Dict[str, Any]] = []
    for name, spec in meta.items():
        dataset = LazySupervisedDataset(
            "internvl2_5",
            spec,
            tokenizer,
            None,
            name,
            256,
            image_size=448,
            is_train=False,
            group_by_length=True,
            dynamic_image_size=True,
            use_thumbnail=True,
            min_dynamic_patch=1,
            max_dynamic_patch=max_dynamic_patch,
            repeat_time=1,
        )
        for index in sorted({0, len(dataset) - 1}):
            sample = dataset[index]
            pixels = sample["pixel_values"]
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            image_ends = int((input_ids == image_end_token_id).sum())
            supervised_tokens = int((labels != -100).sum())
            ok = (
                tuple(pixels.shape) == (18, 3, 448, 448)
                and bool(torch.isfinite(pixels).all())
                and image_ends == 6
                and 0 < input_ids.numel() <= max_seq_length
                and supervised_tokens > 0
            )
            checks.append(
                {
                    "dataset": name,
                    "index": index,
                    "ok": ok,
                    "input_tokens": int(input_ids.numel()),
                    "supervised_tokens": supervised_tokens,
                    "pixel_shape": list(pixels.shape),
                    "image_end_tokens": image_ends,
                    "pixels_finite": bool(torch.isfinite(pixels).all()),
                }
            )
    return {
        "ok": all(item["ok"] for item in checks),
        "meta": str(meta_path.resolve()),
        "base_vlm": str(base_vlm.resolve()),
        "max_seq_length": int(max_seq_length),
        "max_dynamic_patch": int(max_dynamic_patch),
        "expected_camera_views": 6,
        "expected_tiles_per_sample": 18,
        "checks": checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_preflight(
            args.meta,
            args.base_vlm,
            args.max_seq_length,
            args.max_dynamic_patch,
        )
    except Exception as exc:
        print(f"official Stage1 preflight failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
