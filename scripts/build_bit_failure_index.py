#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


FAILURE_WEIGHTS = {
    "zero_score": 4.0,
    "dac_zero": 3.0,
    "nc_zero": 3.0,
    "ttc_zero": 2.0,
    "low_pdms_below_0_5": 2.0,
    "normal": 1.0,
}


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f) if row.get("sample_token") != "average"]


def discover_metrics_file(eval_dir: Path) -> Path:
    candidates = [
        eval_dir / "per_sample_metrics.jsonl",
        eval_dir / "pdm_per_sample.jsonl",
        eval_dir / "pdm_results.csv",
        eval_dir / "predictions.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No per-sample metric file found under {eval_dir}")


def metadata_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".metadata.json")


def parse_tags(value: str | None) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def tags_allowed(tags: Sequence[str], include_tags: Optional[Sequence[str]], exclude_tags: Optional[Sequence[str]]) -> bool:
    tag_set = {str(tag) for tag in tags}
    if include_tags and tag_set.isdisjoint({str(tag) for tag in include_tags}):
        return False
    if exclude_tags and not tag_set.isdisjoint({str(tag) for tag in exclude_tags}):
        return False
    return True


def infer_split(args: argparse.Namespace, metrics_file: Path) -> str:
    if args.split:
        return str(args.split)
    if args.eval_dir is not None:
        aggregate_path = args.eval_dir / "aggregate_metrics.json"
        if aggregate_path.is_file():
            try:
                aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
                if aggregate.get("split"):
                    return str(aggregate["split"])
            except json.JSONDecodeError:
                pass
    joined = " ".join(str(item).lower() for item in (args.eval_dir, metrics_file))
    if "navtest" in joined:
        return "navtest"
    if "test" in joined:
        return "test"
    if "train" in joined:
        return "train"
    return "unknown"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        return read_csv(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("samples", "predictions", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
    raise ValueError(f"Unsupported metrics file format: {path}")


def normalize_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    pdm = row.get("pdm") if isinstance(row.get("pdm"), dict) else {}
    def metric(*names: str) -> float | None:
        for name in names:
            if name in row:
                value = as_float(row.get(name))
                if value is not None:
                    return value
            if name in pdm:
                value = as_float(pdm.get(name))
                if value is not None:
                    return value
        return None

    score = metric("pdm_score", "score", "PDMS")
    nc = metric("no_at_fault_collisions", "NC")
    dac = metric("drivable_area_compliance", "DAC")
    ttc = metric("time_to_collision_within_bound", "TTC")
    ego_progress = metric("ego_progress", "EP")
    comfort = metric("comfort")
    tags = []
    eps = 1e-9
    if score is not None and score <= eps:
        tags.append("zero_score")
    if dac is not None and dac <= eps:
        tags.append("dac_zero")
    if nc is not None and nc <= eps:
        tags.append("nc_zero")
    if ttc is not None and ttc <= eps:
        tags.append("ttc_zero")
    if score is not None and score < 0.5:
        tags.append("low_pdms_below_0_5")
    if not tags:
        tags.append("normal")
    sample_weight = max(FAILURE_WEIGHTS.get(tag, FAILURE_WEIGHTS["normal"]) for tag in tags)
    return {
        "scene_token": row.get("scene_token"),
        "sample_token": row.get("sample_token"),
        "pdm_score": score,
        "no_at_fault_collisions": nc,
        "drivable_area_compliance": dac,
        "time_to_collision_within_bound": ttc,
        "ego_progress": ego_progress,
        "comfort": comfort,
        "failure_tags": tags,
        "sample_weight": sample_weight,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    fields = [
        "scene_token",
        "sample_token",
        "pdm_score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
        "failure_tags",
        "sample_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["failure_tags"] = ";".join(item.get("failure_tags", []))
            writer.writerow({key: item.get(key) for key in fields})


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    counts: Dict[str, int] = {}
    for row in rows:
        for tag in row.get("failure_tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    return {"num_samples": total, "tag_counts": dict(sorted(counts.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a BiT-Drive left-tail failure index from baseline per-sample metrics.")
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/project/VLA-AD/experiments/bit_drive/failure_index"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--include-tags", default=None)
    parser.add_argument("--exclude-tags", default=None)
    parser.add_argument("--allowed-for-training", action="store_true", default=None)
    parser.add_argument("--disallow-training", action="store_false", dest="allowed_for_training")
    args = parser.parse_args()
    metrics_file = args.metrics_file or discover_metrics_file(args.eval_dir)
    include_tags = parse_tags(args.include_tags)
    exclude_tags = parse_tags(args.exclude_tags)
    split = infer_split(args, metrics_file)
    allowed_for_training = args.allowed_for_training
    if allowed_for_training is None:
        allowed_for_training = split.lower() in {"train", "navtrain", "training"}
    rows = []
    for row in load_rows(metrics_file):
        normalized = normalize_metric_row(row)
        if tags_allowed(normalized.get("failure_tags", []), include_tags, exclude_tags):
            normalized["split"] = split
            normalized["allowed_for_training"] = bool(allowed_for_training)
            rows.append(normalized)
    if not rows:
        raise RuntimeError(f"No metric rows found in {metrics_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "baseline_failure_index.jsonl"
    metadata = {
        "split": split,
        "source_eval_dir": str(args.eval_dir) if args.eval_dir else None,
        "source_metrics_file": str(metrics_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allowed_for_training": bool(allowed_for_training),
        "include_tags": include_tags,
        "exclude_tags": exclude_tags,
    }
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    metadata_path(jsonl_path).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "failure_summary.csv", rows)
    summary = summarize(rows)
    lines = [
        "# BiT-Drive Failure Index Summary",
        "",
        f"Source metrics: `{metrics_file}`",
        f"Split: `{split}`",
        f"Allowed for training: `{bool(allowed_for_training)}`",
        f"Samples: {summary['num_samples']}",
        "",
        "| Tag | Count |",
        "| --- | ---: |",
    ]
    for tag, count in summary["tag_counts"].items():
        lines.append(f"| {tag} | {count} |")
    (args.output_dir / "failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"failure_index": str(jsonl_path), "metadata": metadata, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
