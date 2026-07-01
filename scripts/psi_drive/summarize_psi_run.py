#!/usr/bin/env python3
"""Summarize a PSI-Drive Stage2/Stage3 run without mutating it."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def metric_stats(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    vals = [value for value in (finite_float(item) for item in values) if value is not None]
    if not vals:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def inventory_summary(run_root: Path) -> Dict[str, Any]:
    rows = read_tsv(run_root / "checkpoint_store" / "inventory.tsv")
    unique_sha = {row.get("sha256", "") for row in rows if row.get("sha256")}
    latest = None
    if rows:
        latest = max(rows, key=lambda row: (parse_int(row.get("epoch")), parse_int(row.get("train_step")), row.get("archived_at", "")))
    state_counts: Dict[str, Dict[str, int]] = {}
    for split in ("val6000", "navtest"):
        counts = Counter(row.get(f"{split}_state", "") or "missing" for row in rows)
        state_counts[split] = dict(sorted(counts.items()))
    return {
        "rows": len(rows),
        "unique_sha256": len(unique_sha),
        "latest_checkpoint_id": latest.get("checkpoint_id") if latest else None,
        "latest_sha256": latest.get("sha256") if latest else None,
        "latest_epoch": latest.get("epoch") if latest else None,
        "latest_train_step": latest.get("train_step") if latest else None,
        "state_counts": state_counts,
    }


def raw_checkpoint_summary(run_root: Path) -> Dict[str, Any]:
    raw_dir = run_root / "checkpoints" / "raw"
    paths = sorted(raw_dir.glob("*.ckpt")) if raw_dir.is_dir() else []
    latest = max(paths, key=lambda path: path.stat().st_mtime, default=None)
    return {
        "count": len(paths),
        "latest_path": str(latest) if latest else None,
        "latest_name": latest.name if latest else None,
        "latest_mtime": datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z") if latest else None,
    }


def eval_status_summary(path: Path) -> Dict[str, Any]:
    rows = read_tsv(path)
    latest_by_sha: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row.get("sha256") or row.get("checkpoint_id") or row.get("checkpoint") or ""
        if not key:
            continue
        latest_by_sha[str(key)] = row
    final_counts = Counter(row.get("state", "") or "missing" for row in latest_by_sha.values())
    append_counts = Counter(row.get("state", "") or "missing" for row in rows)
    return {
        "path": str(path),
        "rows": len(rows),
        "unique_checkpoints": len(latest_by_sha),
        "latest_state_counts": dict(sorted(final_counts.items())),
        "appended_state_counts": dict(sorted(append_counts.items())),
    }


def top5_summary(path: Path) -> Dict[str, Any]:
    payload = read_json(path)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        entries = []
    stats = metric_stats(entry.get("pdms") for entry in entries if isinstance(entry, dict))
    best = entries[0] if entries and isinstance(entries[0], dict) else {}
    return {
        "path": str(path),
        "exists": path.is_file(),
        "complete": bool(payload.get("complete", False)) if payload else False,
        "num_ranked": payload.get("num_ranked") if payload else None,
        "entries": len(entries),
        "pdms": stats,
        "best_checkpoint_id": best.get("checkpoint_id"),
        "best_sha256": best.get("sha256"),
        "best_pdms": finite_float(best.get("pdms")),
        "best_object_path": best.get("object_path"),
    }


def summarize(run_root: Path, stage: str) -> Dict[str, Any]:
    data_report = read_json(run_root / "data_report.json")
    audit = read_json(run_root / "checkpoint_store" / "audit_latest.json")
    return {
        "created_at": utc_now(),
        "run_root": str(run_root),
        "stage": stage,
        "resolved_command": str(run_root / "resolved_command.sh") if (run_root / "resolved_command.sh").is_file() else None,
        "data_report": data_report,
        "raw_checkpoints": raw_checkpoint_summary(run_root),
        "inventory": inventory_summary(run_root),
        "checkpoint_store_audit": audit,
        "eval_status": {
            "val6000": eval_status_summary(run_root / "eval" / "val6000" / "checkpoint_eval_status.tsv"),
            "navtest": eval_status_summary(run_root / "eval" / "navtest" / "checkpoint_eval_status.tsv"),
        },
        "top5": {
            "val6000": top5_summary(run_root / "rankings" / "val6000" / "current_top5.json"),
            "navtest": top5_summary(run_root / "rankings" / "navtest" / "current_top5.json"),
        },
    }


def fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_md(summary: Dict[str, Any]) -> str:
    data = summary.get("data_report", {})
    train = data.get("train", {}) if isinstance(data, dict) else {}
    val = data.get("val", {}) if isinstance(data, dict) else {}
    raw = summary["raw_checkpoints"]
    inv = summary["inventory"]
    audit = summary.get("checkpoint_store_audit", {})
    lines = [
        f"# PSI-Drive {summary['stage']} Run Summary",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- run_root: `{summary['run_root']}`",
        f"- resolved_command: `{summary.get('resolved_command') or 'n/a'}`",
        "",
        "## Data",
        "",
        f"- loader_mode: `{data.get('loader_mode', 'n/a') if isinstance(data, dict) else 'n/a'}`",
        f"- cache_train_all_records: `{data.get('cache_train_all_records', 'n/a') if isinstance(data, dict) else 'n/a'}`",
        f"- train records: `{train.get('num_records', 'n/a')}`",
        f"- val records: `{val.get('num_records', 'n/a')}`",
        f"- train/val overlap count: `{data.get('train_val_overlap_count', 'n/a') if isinstance(data, dict) else 'n/a'}`",
        f"- stage2 target source: `{train.get('stage2_target_source', 'n/a')}`",
        f"- support index records: `{train.get('stage2_pareto_support_index_records', 'n/a')}`",
        "",
        "## Checkpoints",
        "",
        f"- raw checkpoint count: `{raw['count']}`",
        f"- latest raw: `{raw.get('latest_name') or 'n/a'}`",
        f"- latest raw mtime: `{raw.get('latest_mtime') or 'n/a'}`",
        f"- inventory rows: `{inv['rows']}`",
        f"- unique archived objects: `{inv['unique_sha256']}`",
        f"- latest archived checkpoint: `{inv.get('latest_checkpoint_id') or 'n/a'}`",
        f"- latest archived sha256: `{inv.get('latest_sha256') or 'n/a'}`",
        f"- checkpoint store audit passed: `{audit.get('passed', 'n/a')}`",
        f"- checkpoint store audit failures: `{len(audit.get('failures', [])) if isinstance(audit, dict) else 'n/a'}`",
        "",
        "## Evaluation Status",
        "",
    ]
    for split in ("val6000", "navtest"):
        status = summary["eval_status"][split]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- status rows: `{status['rows']}`",
                f"- unique checkpoints in status: `{status['unique_checkpoints']}`",
                f"- latest state counts: `{json.dumps(status['latest_state_counts'], sort_keys=True)}`",
                f"- appended state counts: `{json.dumps(status['appended_state_counts'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(["## Top-5", ""])
    for split in ("val6000", "navtest"):
        top = summary["top5"][split]
        pdms = top["pdms"]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- manifest exists: `{top['exists']}`",
                f"- complete: `{top['complete']}`",
                f"- entries: `{top['entries']}`",
                f"- num_ranked: `{fmt_value(top.get('num_ranked'))}`",
                f"- best checkpoint: `{fmt_value(top.get('best_checkpoint_id'))}`",
                f"- best sha256: `{fmt_value(top.get('best_sha256'))}`",
                f"- best PDMS: `{fmt_value(top.get('best_pdms'))}`",
                f"- Top-5 PDMS mean/std: `{fmt_value(pdms.get('mean'))}` / `{fmt_value(pdms.get('std'))}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Remaining Gates",
            "",
            "- Stage2/Stage3 completion is not implied by this summary.",
            "- val6000 Top-5 is complete only when the val6000 manifest exists and has five unique completed entries.",
            "- navtest Top-5 is complete only when the navtest manifest exists and has five unique completed full-navtest entries.",
            "- Final SOTA claims require full navtest metrics and verified immutable checkpoint objects.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.run_root, str(args.stage))
    write_json(args.output_json, summary)
    write_text(args.output_md, render_md(summary))
    print(json.dumps({"run_root": str(args.run_root), "output_json": str(args.output_json), "output_md": str(args.output_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
