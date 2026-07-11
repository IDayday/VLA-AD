#!/usr/bin/env python3
"""Audit the Stage2 proxy against released trajectory rows and raw transforms."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bench2drive_contract import (  # noqa: E402
    BENCH2DRIVE_CAMERA_ORDER,
    BENCH2DRIVE_SYSTEM_MESSAGE,
    build_bench2drive_stage1_question,
    planner_to_stage1_text_trajectory,
    relative_lidar_poses,
)


POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def _relative_image(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    for prefix in ("./Bench2drive/v1/", "Bench2drive/v1/"):
        if normalized.startswith(prefix):
            return PurePosixPath(normalized[len(prefix) :])
    raise ValueError(f"Unexpected released image path: {value!r}")


def _annotation(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def audit_record(record: dict[str, Any], data_root: Path) -> dict[str, Any]:
    images = record.get("image")
    conversations = record.get("conversations")
    if not isinstance(images, list) or len(images) != 6:
        raise ValueError("released row does not contain six images")
    relative_images = [_relative_image(value) for value in images]
    camera_order = [path.parts[-2] for path in relative_images]
    if camera_order != list(BENCH2DRIVE_CAMERA_ORDER):
        raise ValueError(f"camera order mismatch: {camera_order}")
    frame_indices = {int(path.stem) for path in relative_images}
    clip_names = {path.parts[0] for path in relative_images}
    if len(frame_indices) != 1 or len(clip_names) != 1:
        raise ValueError("six images do not share one clip/frame")
    frame_index = next(iter(frame_indices))
    clip_name = next(iter(clip_names))
    annotation_dir = data_root / clip_name / "anno"
    current = _annotation(annotation_dir / f"{frame_index:05d}.json.gz")
    future = [
        _annotation(annotation_dir / f"{frame_index + 5 * offset:05d}.json.gz")
        for offset in range(1, 7)
    ]
    expected_question = build_bench2drive_stage1_question(
        speed=float(current["speed"]),
        acceleration_xy=[float(current["acceleration"][0]), -float(current["acceleration"][1])],
        command=current["command_near"],
    )
    if conversations[0] != {"from": "system", "value": BENCH2DRIVE_SYSTEM_MESSAGE}:
        raise ValueError("system prompt mismatch")
    if conversations[1] != {"from": "human", "value": expected_question}:
        raise ValueError("human prompt mismatch")
    released_points = np.asarray(
        [[float(value) for value in match] for match in POINT_RE.findall(conversations[2]["value"])],
        dtype=np.float32,
    )
    if released_points.shape != (6, 3):
        raise ValueError(f"released answer has shape {released_points.shape}")
    planner_trajectory = relative_lidar_poses(current, future)
    text_trajectory = planner_to_stage1_text_trajectory(planner_trajectory)
    xy_matches = np.allclose(np.round(text_trajectory[:, :2], 2), released_points[:, :2], atol=1e-5)
    heading_difference = np.arctan2(
        np.sin(text_trajectory[:, 2] - released_points[:, 2]),
        np.cos(text_trajectory[:, 2] - released_points[:, 2]),
    )
    # A small number of released answers use an equivalent heading shifted by
    # +/-2pi (for example -4.66 instead of 1.62).  Planner yaw is normalized;
    # validate angular equivalence instead of copying that textual unwrap.
    heading_matches = bool(np.max(np.abs(heading_difference)) <= 0.006)
    if not xy_matches or not heading_matches:
        raise ValueError(
            "trajectory transform mismatch; "
            f"xy_matches={xy_matches}, max_heading_mod_2pi={float(np.max(np.abs(heading_difference)))}"
        )
    return {"clip": clip_name, "frame_index": frame_index}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-jsonl",
        type=Path,
        default=Path("/mnt/project/recogdrive_pretraining/Bench2drive_Traj/Bench2drive_Traj.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--expected-rows", type=int, default=196761)
    parser.add_argument("--expected-raw-clips", type=int, default=1000)
    parser.add_argument("--expected-raw-windows", type=int, default=202656)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_count <= 0 or args.expected_rows <= 0:
        raise ValueError("sample-count and expected-rows must be positive")
    stride = max(1, args.expected_rows // args.sample_count)
    checked = []
    rows = 0
    with args.traj_jsonl.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            if index % stride == 0 and len(checked) < args.sample_count:
                checked.append(audit_record(json.loads(line), args.data_root))
            rows += 1
    if rows != args.expected_rows:
        raise RuntimeError(f"Released trajectory row count changed: {rows} != {args.expected_rows}")
    if len(checked) != args.sample_count:
        raise RuntimeError(f"Only checked {len(checked)} records, expected {args.sample_count}")

    raw_clip_count = 0
    raw_window_count = 0
    for clip in sorted(args.data_root.iterdir()):
        annotation_dir = clip / "anno"
        if not clip.is_dir() or clip.name == "maps" or not annotation_dir.is_dir():
            continue
        annotation_paths = sorted(annotation_dir.glob("*.json.gz"))
        frame_indices = [int(path.name.split(".", 1)[0]) for path in annotation_paths]
        if frame_indices != list(range(len(frame_indices))):
            raise RuntimeError(f"Raw annotation frames are not contiguous from zero: {clip}")
        raw_clip_count += 1
        # current frames [15, n-31] inclusive: four history poses at offsets
        # -15/-10/-5/0 and six future poses at +5..+30.
        raw_window_count += max(0, len(frame_indices) - 45)
    if raw_clip_count != args.expected_raw_clips:
        raise RuntimeError(f"Raw clip count changed: {raw_clip_count} != {args.expected_raw_clips}")
    if raw_window_count != args.expected_raw_windows:
        raise RuntimeError(f"Raw Stage2 window count changed: {raw_window_count} != {args.expected_raw_windows}")
    report = {
        "ok": True,
        "traj_jsonl": str(args.traj_jsonl.resolve()),
        "data_root": str(args.data_root.resolve()),
        "total_rows": rows,
        "checked_records": len(checked),
        "raw_clips": raw_clip_count,
        "raw_stage2_windows": raw_window_count,
        "sampling_stride": stride,
        "first": checked[0],
        "last": checked[-1],
        "verified": [
            "six-camera order",
            "byte-equivalent system prompt",
            "byte-equivalent human prompt",
            "six future frames at 0.5-second intervals",
            "world2lidar planner/text coordinate conversion",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
