#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch


RISK_KEYS = (
    "generic_risk_labels",
    "drivable_risk_labels",
    "ttc_risk_labels",
    "comfort_risk_labels",
    "risk_labels",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject RISK-VLA label tensors into chunk-cache .pt samples.")
    parser.add_argument("--chunk-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite-risk-keys", action="store_true")
    parser.add_argument("--output-report-csv", type=Path, default=None)
    parser.add_argument("--output-report-md", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def labels_for_merge(row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    labels = {}
    for key in RISK_KEYS:
        if key in row:
            labels[key] = torch.tensor(row[key], dtype=torch.float32)
    return labels


def load_labels(path: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    labels: Dict[str, Dict[str, torch.Tensor]] = {}
    for row in read_jsonl(path):
        token = row.get("sample_token") or row.get("token") or row.get("scene_token")
        if token is None:
            continue
        tensors = labels_for_merge(row)
        if tensors:
            labels[str(token)] = tensors
    return labels


def sample_token(sample: Dict[str, Any], path: Path) -> str:
    return str(sample.get("sample_token") or sample.get("token") or path.stem)


def find_sample_files(chunk_cache_dir: Path) -> List[Path]:
    return sorted(path for path in chunk_cache_dir.rglob("*.pt") if path.is_file())


def torch_load_sample(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def merge_labels(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if args.dry_run and args.write:
        raise ValueError("--dry-run and --write are mutually exclusive")
    labels = load_labels(args.labels_jsonl)
    rows: List[Dict[str, Any]] = []
    counts = {
        "matched_samples": 0,
        "missing_labels": 0,
        "would_write_count": 0,
        "written_count": 0,
        "skipped_count": 0,
    }
    should_write = bool(args.write)
    for path in find_sample_files(args.chunk_cache_dir):
        sample = torch_load_sample(path)
        if not isinstance(sample, dict):
            rows.append({"path": str(path), "token": path.stem, "status": "skipped_non_dict"})
            counts["skipped_count"] += 1
            continue
        token = sample_token(sample, path)
        if token not in labels:
            rows.append({"path": str(path), "token": token, "status": "missing_label"})
            counts["missing_labels"] += 1
            continue
        existing = [key for key in labels[token] if key in sample]
        if existing and not args.overwrite_risk_keys:
            rows.append({"path": str(path), "token": token, "status": "skipped_existing_keys", "existing_keys": ",".join(existing)})
            counts["skipped_count"] += 1
            continue
        counts["matched_samples"] += 1
        counts["would_write_count"] += 1
        if should_write:
            sample.update(labels[token])
            torch.save(sample, path)
            counts["written_count"] += 1
            status = "written"
        else:
            status = "dry_run_would_write"
        rows.append({"path": str(path), "token": token, "status": status, "existing_keys": ",".join(existing)})
    return pd.DataFrame(rows), counts


def write_markdown(path: Path, counts: Dict[str, int]) -> None:
    lines = [
        "# RISK-VLA Chunk Label Merge Report",
        "",
        "| Count | Value |",
        "| --- | ---: |",
    ]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report, counts = merge_labels(args)
    if args.output_report_csv is not None:
        args.output_report_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output_report_csv, index=False)
    if args.output_report_md is not None:
        args.output_report_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_report_md, counts)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
