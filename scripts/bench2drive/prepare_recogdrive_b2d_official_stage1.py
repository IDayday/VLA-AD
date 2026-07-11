#!/usr/bin/env python3
"""Prepare a read-only data view and InternVL meta files for official B2D SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


class OfficialStage1DataError(RuntimeError):
    """Raised when the official Stage1 data view cannot be prepared safely."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj-jsonl",
        type=Path,
        default=Path("/mnt/project/recogdrive_pretraining/Bench2drive_Traj/Bench2drive_Traj.jsonl"),
    )
    parser.add_argument(
        "--qa-jsonl",
        type=Path,
        default=Path("/mnt/project/recogdrive_pretraining/Bench2drive_QA/Bench2drive_QA.jsonl"),
    )
    parser.add_argument("--raw-data-root", type=Path, default=Path("/mnt/data/Bench2Drive-Base"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bench2drive_recogdrive_stage1_official_data"),
    )
    parser.add_argument(
        "--smoke-records-per-dataset",
        type=int,
        default=8,
        help="Write deterministic smoke JSONLs; set to zero to omit them.",
    )
    parser.add_argument(
        "--max-dynamic-patch",
        type=int,
        default=16,
        help="Match the public ReCogDrive SFT launcher; six views receive floor(value / 6) patches each.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[tuple[str, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfficialStage1DataError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise OfficialStage1DataError(f"record at {path}:{line_number} is not an object")
            yield line if line.endswith("\n") else line + "\n", value


def _select_smoke_rows(path: Path, count: int, *, include_longest_conversation: bool) -> list[str]:
    if count <= 0:
        return []
    first_rows: list[tuple[str, Dict[str, Any]]] = []
    longest: Optional[tuple[str, Dict[str, Any]]] = None
    for line, record in _read_jsonl(path):
        if len(first_rows) < count:
            first_rows.append((line, record))
        if include_longest_conversation:
            conversations = record.get("conversations")
            length = len(conversations) if isinstance(conversations, list) else -1
            if longest is None or length > len(longest[1].get("conversations", [])):
                longest = (line, record)
        if len(first_rows) >= count and not include_longest_conversation:
            break
    if len(first_rows) < count:
        raise OfficialStage1DataError(f"{path} has only {len(first_rows)} records; requested {count}")
    if include_longest_conversation and longest is not None:
        longest_id = (longest[1].get("id"), tuple(longest[1].get("image", [])))
        selected_ids = {(row.get("id"), tuple(row.get("image", []))) for _, row in first_rows}
        if longest_id not in selected_ids:
            first_rows[-1] = longest
    return [line for line, _ in first_rows]


def _ensure_dataset_symlink(view_root: Path, raw_root: Path) -> Path:
    link = view_root / "Bench2drive" / "v1"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != raw_root.resolve():
            raise OfficialStage1DataError(
                f"dataset view symlink points to {link.resolve()}, expected {raw_root.resolve()}"
            )
    elif link.exists():
        raise OfficialStage1DataError(f"refusing to replace non-symlink dataset view: {link}")
    else:
        os.symlink(raw_root.resolve(), link, target_is_directory=True)
    return link


def _meta(
    root: Path,
    traj: Path,
    qa: Path,
    traj_length: int,
    qa_length: int,
    max_dynamic_patch: int,
) -> Dict[str, Any]:
    common = {
        "root": str(root.resolve()),
        "data_augment": False,
        "repeat_time": 1,
        "max_dynamic_patch": int(max_dynamic_patch),
    }
    return {
        "Bench2drive_Traj": {
            **common,
            "annotation": str(traj.resolve()),
            "length": int(traj_length),
        },
        "Bench2drive_QA": {
            **common,
            "annotation": str(qa.resolve()),
            "length": int(qa_length),
        },
    }


def prepare_official_stage1_data(args: argparse.Namespace) -> Dict[str, Any]:
    traj = args.traj_jsonl.expanduser().resolve()
    qa = args.qa_jsonl.expanduser().resolve()
    raw_root = args.raw_data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not traj.is_file() or not qa.is_file():
        raise OfficialStage1DataError(f"official JSONL missing: traj={traj}, qa={qa}")
    if not raw_root.is_dir():
        raise OfficialStage1DataError(f"raw Bench2Drive root missing: {raw_root}")
    if args.smoke_records_per_dataset < 0:
        raise OfficialStage1DataError("--smoke-records-per-dataset must be non-negative")
    if args.max_dynamic_patch < 6:
        raise OfficialStage1DataError("--max-dynamic-patch must be at least six for six-view data")

    output.mkdir(parents=True, exist_ok=True)
    view_root = output / "dataset_view"
    link = _ensure_dataset_symlink(view_root, raw_root)

    formal_meta = _meta(view_root, traj, qa, 196761, 49942, args.max_dynamic_patch)
    formal_meta_path = output / "official_meta.json"
    formal_meta_path.write_text(json.dumps(formal_meta, indent=2) + "\n", encoding="utf-8")

    smoke_meta_path: Optional[Path] = None
    smoke_summary: Dict[str, Any] = {}
    count = int(args.smoke_records_per_dataset)
    if count:
        smoke_dir = output / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_traj = smoke_dir / "traj.jsonl"
        smoke_qa = smoke_dir / "qa.jsonl"
        traj_rows = _select_smoke_rows(traj, count, include_longest_conversation=False)
        qa_rows = _select_smoke_rows(qa, count, include_longest_conversation=True)
        smoke_traj.write_text("".join(traj_rows), encoding="utf-8")
        smoke_qa.write_text("".join(qa_rows), encoding="utf-8")
        smoke_meta = _meta(
            view_root,
            smoke_traj,
            smoke_qa,
            len(traj_rows),
            len(qa_rows),
            args.max_dynamic_patch,
        )
        smoke_meta_path = output / "smoke_meta.json"
        smoke_meta_path.write_text(json.dumps(smoke_meta, indent=2) + "\n", encoding="utf-8")
        qa_records = [json.loads(line) for line in qa_rows]
        smoke_summary = {
            "records_per_dataset": count,
            "traj_jsonl": str(smoke_traj),
            "qa_jsonl": str(smoke_qa),
            "qa_conversation_lengths": [len(row["conversations"]) for row in qa_records],
            "max_qa_conversation_length": max(len(row["conversations"]) for row in qa_records),
        }

    summary = {
        "classification": "official unmodified JSONL with a read-only path view",
        "raw_data_root": str(raw_root),
        "dataset_view_symlink": str(link),
        "dataset_view_target": str(link.resolve()),
        "formal_meta": str(formal_meta_path),
        "smoke_meta": str(smoke_meta_path) if smoke_meta_path else None,
        "natural_record_mixing": {
            "trajectory_records": 196761,
            "qa_records": 49942,
            "trajectory_to_qa_ratio": 196761 / 49942,
            "repeat_time": {"Bench2drive_Traj": 1, "Bench2drive_QA": 1},
            "use_data_resampling": False,
        },
        "source_hashes": {
            "traj_jsonl": sha256_file(traj),
            "qa_jsonl": sha256_file(qa),
        },
        "meta_hashes": {
            "formal": sha256_file(formal_meta_path),
            "smoke": sha256_file(smoke_meta_path) if smoke_meta_path else None,
        },
        "smoke": smoke_summary,
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary = prepare_official_stage1_data(args)
    except (OfficialStage1DataError, OSError, ValueError) as exc:
        print(f"official Stage1 data preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
