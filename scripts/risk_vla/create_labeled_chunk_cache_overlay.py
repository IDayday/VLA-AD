#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


RISK_CLASSES = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
MVP_KEYS = ("generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def token_from_row(row: Dict[str, Any]) -> Optional[str]:
    token = row.get("token") or row.get("sample_token") or row.get("scene_token")
    return str(token) if token is not None else None


def labels_by_token(path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        token = token_from_row(row)
        if token:
            labels[token] = row
    return labels


def load_sample(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a dict sample, got {type(obj).__name__}.")
    return obj


def sample_token(sample: Dict[str, Any], path: Path) -> str:
    for key in ("token", "sample_token", "scene_token"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return path.stem


def iter_samples(source_cache_dir: Path) -> Iterable[Tuple[Path, Path, Optional[Dict[str, Any]], str]]:
    for path in sorted(source_cache_dir.glob("**/*.pt")):
        rel = path.relative_to(source_cache_dir)
        sample: Optional[Dict[str, Any]] = None
        token = path.stem
        try:
            sample = load_sample(path)
            token = sample_token(sample, path)
        except Exception:
            sample = None
        yield path, rel, sample, token


def tensor_from_label(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def inject_risk_labels(sample: Dict[str, Any], label: Dict[str, Any], *, overwrite: bool) -> Dict[str, Any]:
    output = dict(sample)
    if "risk_labels" in label:
        if "risk_labels" in output and not overwrite:
            raise KeyError("risk_labels already exists; pass --overwrite to replace risk keys.")
        output["risk_labels"] = tensor_from_label(label["risk_labels"])
    for key in MVP_KEYS:
        if key in label:
            if key in output and not overwrite:
                raise KeyError(f"{key} already exists; pass --overwrite to replace risk keys.")
            output[key] = tensor_from_label(label[key])
    for name in RISK_CLASSES:
        label_key = f"label_{name}"
        if label_key in label:
            if label_key in output and not overwrite:
                raise KeyError(f"{label_key} already exists; pass --overwrite to replace risk keys.")
            output[label_key] = tensor_from_label(label[label_key])
    return output


def plan_overlay(
    source_cache_dir: Path,
    labels_jsonl: Path,
    *,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    label_map = labels_by_token(labels_jsonl)
    rows: List[Dict[str, Any]] = []
    for source_path, rel_path, sample, token in iter_samples(source_cache_dir):
        if token not in label_map:
            continue
        rows.append(
            {
                "token": token,
                "source_path": str(source_path),
                "relative_path": str(rel_path),
                "has_loaded_sample": sample is not None,
                "has_label": True,
            }
        )
        if max_samples is not None and len(rows) >= int(max_samples):
            break
    return rows


def ensure_output_ready(output_cache_dir: Path, *, overwrite: bool, write: bool) -> None:
    if output_cache_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output cache already exists: {output_cache_dir}. Pass --overwrite to replace it.")
        if write:
            shutil.rmtree(output_cache_dir)


def write_reports(rows: List[Dict[str, Any]], output_cache_dir: Path) -> None:
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_cache_dir / "overlay_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["token", "source_path", "relative_path", "has_loaded_sample", "has_label", "operation", "output_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    md_lines = [
        "# RISK-VLA Labeled Chunk Cache Overlay",
        "",
        f"Rows: `{len(rows)}`",
        "",
        "| operation | count |",
        "| --- | ---: |",
    ]
    counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row.get("operation", "planned"))] = counts.get(str(row.get("operation", "planned")), 0) + 1
    for operation, count in sorted(counts.items()):
        md_lines.append(f"| {operation} | {count} |")
    md_lines.append("")
    (output_cache_dir / "overlay_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def create_overlay(
    source_cache_dir: Path,
    output_cache_dir: Path,
    labels_jsonl: Path,
    *,
    mode: str,
    max_samples: Optional[int] = None,
    dry_run: bool = True,
    write: bool = False,
    overwrite: bool = False,
    overwrite_risk_keys: bool = False,
) -> List[Dict[str, Any]]:
    if write and dry_run:
        raise ValueError("--write and --dry-run are mutually exclusive.")
    rows = plan_overlay(source_cache_dir, labels_jsonl, max_samples=max_samples)
    label_map = labels_by_token(labels_jsonl)
    if write:
        ensure_output_ready(output_cache_dir, overwrite=overwrite, write=True)
        output_cache_dir.mkdir(parents=True, exist_ok=True)
    elif output_cache_dir.exists() and not overwrite:
        raise FileExistsError(f"Output cache already exists: {output_cache_dir}. Pass --overwrite for dry-run planning.")

    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        source_path = Path(row["source_path"])
        rel_path = Path(row["relative_path"])
        output_path = output_cache_dir / rel_path
        row = dict(row)
        row["output_path"] = str(output_path)
        row["operation"] = "would_symlink" if mode == "symlink" else "would_copy_inject"
        if write:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                os.symlink(source_path, output_path)
                row["operation"] = "symlinked"
            else:
                sample = load_sample(source_path)
                sample = inject_risk_labels(sample, label_map[row["token"]], overwrite=overwrite_risk_keys)
                torch.save(sample, output_path)
                row["operation"] = "copied_injected"
        output_rows.append(row)
    if write:
        index_path = output_cache_dir / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps({"sample_token": row["token"], "path": row["relative_path"]}, sort_keys=True))
                handle.write("\n")
    if write or dry_run:
        report_dir = output_cache_dir if write else output_cache_dir.parent / f"{output_cache_dir.name}_dry_run_report"
        write_reports(output_rows, report_dir)
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a non-mutating labeled chunk-cache overlay for RISK-VLA.")
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-risk-keys", action="store_true")
    args = parser.parse_args()

    dry_run = bool(args.dry_run or not args.write)
    rows = create_overlay(
        args.source_cache_dir,
        args.output_cache_dir,
        args.labels_jsonl,
        mode=args.mode,
        max_samples=args.max_samples,
        dry_run=dry_run,
        write=bool(args.write),
        overwrite=bool(args.overwrite),
        overwrite_risk_keys=bool(args.overwrite_risk_keys),
    )
    print(json.dumps({"planned_or_written": len(rows), "mode": args.mode, "write": bool(args.write)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
