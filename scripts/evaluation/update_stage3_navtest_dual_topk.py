#!/usr/bin/env python3
"""Build per-protocol NAVTEST rankings and maintain hard-linked top-k checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROTOCOLS = {
    "navsim_v1_pdms": "PDMS",
    "navsim_v2_epdms": "EPDMS",
}


def _atomic_tsv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_rows(eval_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(eval_root.glob("epoch_*/epoch_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metrics_path"] = str(path)
        rows.append(payload)
    rows.sort(key=lambda row: (int(row["epoch"]), int(row.get("step", -1))))
    return rows


def _score(row: Dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _link_or_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(destination)


def _update_protocol(
    eval_root: Path,
    rows: List[Dict[str, Any]],
    protocol: str,
    score_key: str,
    top_k: int,
    backup_mode: str,
) -> None:
    ranked = []
    for row in rows:
        score = _score(row, score_key)
        checkpoint = Path(str(row.get("checkpoint", "")))
        if score is None or not checkpoint.is_file():
            continue
        ranked.append((score, row, checkpoint))
    ranked.sort(key=lambda item: (-item[0], int(item[1]["epoch"]), int(item[1].get("step", -1))))
    top = ranked[:top_k]

    top_dir = eval_root / f"top{top_k}_ckpts" / protocol
    top_dir.mkdir(parents=True, exist_ok=True)
    active = set()
    output_rows = []
    for rank, (score, row, checkpoint) in enumerate(top, 1):
        destination = top_dir / f"rank{rank}_{checkpoint.name}"
        active.add(destination.name)
        if backup_mode == "hardlink_or_copy" and not destination.exists():
            _link_or_copy(checkpoint, destination)
        output_rows.append(
            {
                "rank": rank,
                "epoch": row["epoch"],
                "step": row.get("step"),
                "score": score,
                "checkpoint": str(checkpoint),
                "backup_path": str(destination) if backup_mode == "hardlink_or_copy" else "",
                "v1_eval_dir": row.get("v1_eval_dir", ""),
                "v2_eval_dir": row.get("v2_eval_dir", ""),
            }
        )
    for old_path in top_dir.glob("rank*_*.ckpt"):
        if old_path.name not in active:
            old_path.unlink()

    _atomic_tsv(
        eval_root / f"{protocol}_top{top_k}.tsv",
        ["rank", "epoch", "step", "score", "checkpoint", "backup_path", "v1_eval_dir", "v2_eval_dir"],
        output_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backup-mode", choices=("hardlink_or_copy", "record_only"), default="hardlink_or_copy")
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    eval_root = args.eval_root.expanduser().resolve()
    eval_root.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(eval_root)
    all_fields = [
        "epoch",
        "step",
        "checkpoint",
        "PDMS",
        "EPDMS",
        "v1_num_valid",
        "v2_num_valid",
        "v1_eval_dir",
        "v2_eval_dir",
        "updated_at",
    ]
    _atomic_tsv(eval_root / "all_epoch_navtest_metrics.tsv", all_fields, rows)
    for protocol, score_key in PROTOCOLS.items():
        _update_protocol(eval_root, rows, protocol, score_key, args.top_k, args.backup_mode)


if __name__ == "__main__":
    main()
