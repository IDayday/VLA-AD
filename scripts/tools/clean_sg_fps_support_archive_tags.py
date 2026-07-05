#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import lzma
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ANCHOR_SOURCES = {"gt", "il", "recogdrive_stage3"}


def _iter_paths(root: Path):
    if root.is_file():
        yield root
        return
    yield from sorted(root.glob("*.pkl.xz"))


def _load(path: Path) -> dict[str, Any]:
    with lzma.open(path, "rb") as f:
        record = pickle.load(f)
    if not isinstance(record, dict):
        raise TypeError(f"record must be dict, got {type(record).__name__}: {path}")
    return record


def _save_atomic(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with lzma.open(tmp, "wb") as f:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _is_anchor_source(source: str) -> bool:
    lower = str(source).lower()
    return any(anchor in lower for anchor in ANCHOR_SOURCES)


def clean_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    stats = Counter()
    rewards = np.asarray(record.get("rewards", []), dtype=np.float32)
    valid_mask = np.asarray(record.get("valid_mask", np.zeros_like(rewards, dtype=np.bool_)), dtype=np.bool_)
    sources = [str(source) for source in record.get("sources", [])]
    support_tags = [str(tag) for tag in record.get("support_tags", [""] * len(sources))]
    if rewards.shape != valid_mask.shape or rewards.shape != (len(sources),) or len(support_tags) != len(sources):
        stats["shape_error"] += 1
        return record, dict(stats)

    changed = False
    for idx, tag in enumerate(support_tags):
        if not tag:
            continue
        stats["tagged_before"] += 1
        if bool(valid_mask[idx]) or _is_anchor_source(sources[idx]):
            stats["tagged_kept"] += 1
            continue
        support_tags[idx] = ""
        changed = True
        stats["invalid_non_anchor_tags_cleared"] += 1

    if not changed:
        return record, dict(stats)

    selected_mask = np.asarray([bool(tag) for tag in support_tags], dtype=np.bool_)
    valid_indices = np.flatnonzero(valid_mask)
    raw_idx = int(np.argmax(rewards)) if rewards.size else 0
    valid_idx = int(valid_indices[np.argmax(rewards[valid_indices])]) if valid_indices.size else raw_idx
    selected_indices = np.flatnonzero(selected_mask)
    selected_idx = int(selected_indices[np.argmax(rewards[selected_indices])]) if selected_indices.size else valid_idx

    out = dict(record)
    out["support_tags"] = support_tags
    out["best_valid_reward"] = float(rewards[valid_idx]) if rewards.size else 0.0
    out["best_valid_source"] = str(sources[valid_idx]) if sources else ""
    out["best_selected_reward"] = float(rewards[selected_idx]) if rewards.size else 0.0
    out["best_selected_source"] = str(sources[selected_idx]) if sources else ""
    out["has_valid_candidate"] = bool(valid_mask.any())
    stats["records_changed"] += 1
    return out, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clear SG-FPS v3 support tags from invalid non-anchor candidates after archive build."
    )
    parser.add_argument("--support-archive-path", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.support_archive_path)
    totals = Counter()
    examples: list[dict[str, Any]] = []
    for path in _iter_paths(root):
        totals["records_seen"] += 1
        try:
            record = _load(path)
            cleaned, stats = clean_record(record)
        except Exception as exc:
            totals["load_or_clean_errors"] += 1
            if len(examples) < 16:
                examples.append({"path": str(path), "error": str(exc)})
            continue
        totals.update(stats)
        if cleaned is not record and not args.dry_run:
            _save_atomic(path, cleaned)

    summary = {
        "support_archive_path": str(root),
        "dry_run": bool(args.dry_run),
        "stats": dict(totals),
        "examples": examples,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
