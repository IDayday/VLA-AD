#!/usr/bin/env python3
"""Rewrite navsim_answer image paths to paths available in this workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PATH_REWRITES = (
    (
        "/mnt/public_data/navsim/trainval_all/trainval_sensor_blobs/trainval/",
        "/mnt/navsim/trainval_sensor_blobs/trainval/",
    ),
)


def normalize_image(path: str) -> str:
    for src, dst in PATH_REWRITES:
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    rows = 0
    rewritten = 0
    missing = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            images = []
            for image in row.get("images", []):
                normalized = normalize_image(image)
                if normalized != image:
                    rewritten += 1
                if args.check_images and not Path(normalized).exists():
                    missing += 1
                    if missing <= 5:
                        print(f"missing image after rewrite at line {line_no}: {normalized}")
                images.append(normalized)
            row["images"] = images
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows={rows} rewritten_images={rewritten} missing_images={missing} output={args.output}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
