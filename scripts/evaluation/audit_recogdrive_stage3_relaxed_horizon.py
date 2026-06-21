#!/usr/bin/env python3
"""Build a ReCogDrive Stage3 evaluation audit under a relaxed early horizon.

This script is read-only: it only scans existing navtest evaluation summaries
and per-scene metric CSVs. It does not run inference and does not recompute PDM
scores.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "pdms": ("pdms_mean", "pdms", "score", "pdm_score"),
    "nc": ("nc_mean", "nc", "no_at_fault_collisions"),
    "dac": ("dac_mean", "dac", "drivable_area_compliance"),
    "ttc": ("ttc_mean", "ttc", "time_to_collision_within_bound"),
    "ep": ("ep_mean", "ep", "ego_progress", "progress"),
    "comfort": ("comfort_mean", "comfort", "history_comfort"),
    "ddc": ("ddc_mean", "ddc", "driving_direction_compliance"),
    "tlc": ("tlc_mean", "tlc", "traffic_light_compliance"),
}
METRICS = tuple(METRIC_ALIASES)


def _float_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _checkpoint_sort(checkpoint_id: str) -> tuple[int, int]:
    text = str(checkpoint_id)
    step_match = re.search(r"step[-_=](\d+)", text)
    epoch_match = re.search(r"epoch[-_=](\d+)", text)
    step = int(step_match.group(1)) if step_match else -1
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    return step, epoch


def _run_name(outputs_root: Path, path: Path) -> str:
    try:
        return path.relative_to(outputs_root).parts[0]
    except Exception:
        return path.parts[-1]


def _read_text_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def _first_int(text: str, patterns: Iterable[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


@lru_cache(maxsize=None)
def _run_training_shape(outputs_root_text: str, run: str) -> dict[str, int | str]:
    run_root = Path(outputs_root_text) / run
    text_parts = []
    for rel in (
        "strict_gspo_launch_config.txt",
        "early_gate_config.txt",
        "train/resolved_command.txt",
        "resolved_command.txt",
        "jobs.tsv",
    ):
        text_parts.append(_read_text_if_exists(run_root / rel))
    for status_path in sorted((run_root / "status").glob("*.json")):
        text_parts.append(_read_text_if_exists(status_path))
    text = "\n".join(part for part in text_parts if part)

    devices = _first_int(
        text,
        (
            r"(?:^|\s)(?:GPUS_PER_NODE|gpus_per_node|devices|trainer\.params\.devices)=(\d+)(?:\s|$)",
            r"--nproc_per_node[=\s](\d+)",
        ),
    )
    batch_size = _first_int(
        text,
        (
            r"(?:^|\s)(?:STAGE3_BATCH_SIZE|stage3_batch_size|batch_size|dataloader\.params\.batch_size)=(\d+)(?:\s|$)",
        ),
    )
    accumulate = _first_int(
        text,
        (
            r"(?:^|\s)(?:STAGE3_ACCUMULATE_GRAD_BATCHES|stage3_accumulate_grad_batches|accumulate_grad_batches|trainer\.params\.accumulate_grad_batches)=(\d+)(?:\s|$)",
        ),
    )
    effective = (
        devices * batch_size * accumulate
        if devices is not None and batch_size is not None and accumulate is not None
        else None
    )
    return {
        "devices": devices if devices is not None else "",
        "batch_size": batch_size if batch_size is not None else "",
        "accumulate_grad_batches": accumulate if accumulate is not None else "",
        "effective_batch_size": effective if effective is not None else "",
    }


def _attach_training_shape(outputs_root: Path, row: dict[str, object]) -> None:
    shape = _run_training_shape(str(outputs_root), str(row["run"]))
    row.update(shape)
    step = int(row.get("step", -1))
    effective = shape.get("effective_batch_size")
    row["seen_train_scenes"] = step * int(effective) if step >= 0 and effective != "" else ""


def _checkpoint_from_eval_csv(path: Path) -> str | None:
    for parent in path.parents:
        name = parent.name
        if name.startswith("eval_"):
            return name[len("eval_") :]
    return None


def _first_metric(row: dict[str, str], metric: str) -> float:
    for key in METRIC_ALIASES[metric]:
        value = _float_or_nan(row.get(key))
        if not math.isnan(value):
            return value
    return math.nan


def _read_tsv_records(outputs_root: Path, path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            checkpoint_id = str(row.get("checkpoint_id") or "").strip()
            if not checkpoint_id:
                continue
            pdms = _first_metric(row, "pdms")
            if math.isnan(pdms):
                continue
            step, epoch = _checkpoint_sort(checkpoint_id)
            record: dict[str, object] = {
                "run": _run_name(outputs_root, path),
                "checkpoint_id": checkpoint_id,
                "step": step,
                "epoch": epoch,
                "source": path.name,
                "record_key": str(row.get("csv_path") or f"{path}:{checkpoint_id}"),
                "path": str(path),
                "num_rows": _float_or_nan(row.get("num_rows")),
                "num_valid_rows": _float_or_nan(row.get("num_valid_rows")),
            }
            for metric in METRICS:
                record[metric] = _first_metric(row, metric)
            records.append(record)
    return records


def _read_metric_csv_record(outputs_root: Path, path: Path) -> dict[str, object] | None:
    checkpoint_id = _checkpoint_from_eval_csv(path)
    if not checkpoint_id:
        return None

    sums = {metric: 0.0 for metric in METRICS}
    counts = {metric: 0 for metric in METRICS}
    total_rows = 0
    valid_rows = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "token" not in reader.fieldnames:
            return None
        valid_field = "valid" if "valid" in reader.fieldnames else None
        for row in reader:
            total_rows += 1
            if valid_field is not None and not _truthy(row.get(valid_field)):
                continue
            valid_rows += 1
            for metric in METRICS:
                value = _first_metric(row, metric)
                if not math.isnan(value):
                    sums[metric] += value
                    counts[metric] += 1
    if counts["pdms"] == 0:
        return None

    step, epoch = _checkpoint_sort(checkpoint_id)
    record: dict[str, object] = {
        "run": _run_name(outputs_root, path),
        "checkpoint_id": checkpoint_id,
        "step": step,
        "epoch": epoch,
        "source": "metric_csv",
        "record_key": str(path),
        "path": str(path),
        "num_rows": total_rows,
        "num_valid_rows": valid_rows,
    }
    for metric in METRICS:
        record[metric] = sums[metric] / counts[metric] if counts[metric] else math.nan
    return record


def _scan_records(outputs_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(outputs_root.rglob("navtest_pdms_analysis.tsv")):
        records.extend(_read_tsv_records(outputs_root, path))
    for path in sorted(outputs_root.rglob("checkpoint_eval_submetrics.tsv")):
        records.extend(_read_tsv_records(outputs_root, path))
    for path in sorted(outputs_root.rglob("*.csv")):
        if not any(parent.name.startswith("eval_") for parent in path.parents):
            continue
        record = _read_metric_csv_record(outputs_root, path)
        if record is not None:
            records.append(record)
    return records


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _group_records(outputs_root: Path, records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["run"]), str(record["checkpoint_id"]))].append(record)

    rows: list[dict[str, object]] = []
    for (run, checkpoint_id), group in grouped.items():
        non_navtest = [item for item in group if item.get("source") != "navtest_pdms_analysis.tsv"]
        if non_navtest:
            group = non_navtest
        unique_by_record: dict[str, dict[str, object]] = {}
        for item in group:
            key = str(item.get("record_key") or item.get("path") or id(item))
            current = unique_by_record.get(key)
            if current is None or str(item.get("source")) == "checkpoint_eval_submetrics.tsv":
                unique_by_record[key] = item
        group = list(unique_by_record.values())

        step, epoch = _checkpoint_sort(checkpoint_id)
        row: dict[str, object] = {
            "run": run,
            "checkpoint_id": checkpoint_id,
            "step": step,
            "epoch": epoch,
            "eval_records": len(group),
            "sources": ",".join(sorted({str(item["source"]) for item in group})),
            "paths": "|".join(sorted({str(item["path"]) for item in group})),
            "num_valid_rows": max(
                (_float_or_nan(item.get("num_valid_rows")) for item in group),
                default=math.nan,
            ),
        }
        for metric in METRICS:
            row[metric] = _mean(_float_or_nan(item.get(metric)) for item in group)
        _attach_training_shape(outputs_root, row)
        rows.append(row)
    return sorted(rows, key=lambda r: (-_float_or_nan(r.get("pdms")), str(r["run"]), str(r["checkpoint_id"])))


def _relaxed_rows(rows: list[dict[str, object]], *, max_step: int, max_epoch: int) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (int(row.get("step", -1)) >= 0 and int(row["step"]) <= max_step)
        or (int(row.get("epoch", -1)) >= 0 and int(row["epoch"]) <= max_epoch)
    ]


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "checkpoint_id",
        "step",
        "epoch",
        "devices",
        "batch_size",
        "accumulate_grad_batches",
        "effective_batch_size",
        "seen_train_scenes",
        "pdms",
        "nc",
        "dac",
        "ttc",
        "ep",
        "comfort",
        "ddc",
        "tlc",
        "eval_records",
        "num_valid_rows",
        "sources",
        "paths",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_best_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    best_by_run: dict[str, dict[str, object]] = {}
    for row in rows:
        run = str(row["run"])
        if run not in best_by_run or _float_or_nan(row.get("pdms")) > _float_or_nan(best_by_run[run].get("pdms")):
            best_by_run[run] = row
    _write_tsv(path, sorted(best_by_run.values(), key=lambda r: -_float_or_nan(r.get("pdms"))))


def _progress_key(row: dict[str, object]) -> tuple[int, int, str]:
    step = int(row.get("step", -1))
    epoch = int(row.get("epoch", -1))
    if step >= 0:
        progress = step
    elif epoch >= 0:
        progress = epoch
    else:
        progress = -1
    return progress, epoch, str(row.get("checkpoint_id", ""))


def _write_run_summary_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["run"])].append(row)

    fieldnames = [
        "run",
        "n_checkpoints",
        "first_checkpoint",
        "first_step",
        "first_epoch",
        "first_pdms",
        "best_checkpoint",
        "best_step",
        "best_epoch",
        "best_seen_train_scenes",
        "best_pdms",
        "best_nc",
        "best_dac",
        "best_ttc",
        "best_ep",
        "best_comfort",
        "best_ddc",
        "best_tlc",
        "best_minus_first",
        "latest_checkpoint",
        "latest_step",
        "latest_epoch",
        "latest_seen_train_scenes",
        "latest_pdms",
        "latest_minus_best",
    ]
    summary: list[dict[str, object]] = []
    for run, group in by_run.items():
        first = min(group, key=_progress_key)
        latest = max(group, key=_progress_key)
        best = max(group, key=lambda row: _float_or_nan(row.get("pdms")))
        best_pdms = _float_or_nan(best.get("pdms"))
        first_pdms = _float_or_nan(first.get("pdms"))
        latest_pdms = _float_or_nan(latest.get("pdms"))
        row: dict[str, object] = {
            "run": run,
            "n_checkpoints": len(group),
            "first_checkpoint": first.get("checkpoint_id"),
            "first_step": first.get("step"),
            "first_epoch": first.get("epoch"),
            "first_pdms": first_pdms,
            "best_checkpoint": best.get("checkpoint_id"),
            "best_step": best.get("step"),
            "best_epoch": best.get("epoch"),
            "best_seen_train_scenes": best.get("seen_train_scenes"),
            "best_pdms": best_pdms,
            "best_minus_first": best_pdms - first_pdms,
            "latest_checkpoint": latest.get("checkpoint_id"),
            "latest_step": latest.get("step"),
            "latest_epoch": latest.get("epoch"),
            "latest_seen_train_scenes": latest.get("seen_train_scenes"),
            "latest_pdms": latest_pdms,
            "latest_minus_best": latest_pdms - best_pdms,
        }
        for metric in ("nc", "dac", "ttc", "ep", "comfort", "ddc", "tlc"):
            row[f"best_{metric}"] = best.get(metric)
        summary.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(summary, key=lambda r: -_float_or_nan(r.get("best_pdms"))):
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("/mnt/project/VLA-AD/outputs"))
    parser.add_argument("--max-step", type=int, default=5500)
    parser.add_argument("--max-epoch", type=int, default=4)
    parser.add_argument(
        "--output-all",
        type=Path,
        default=Path("/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_fullscan.tsv"),
    )
    parser.add_argument(
        "--output-best",
        type=Path,
        default=Path("/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_fullscan_best_by_run.tsv"),
    )
    parser.add_argument(
        "--output-run-summary",
        type=Path,
        default=Path("/mnt/project/VLA-AD/outputs/stage3_epoch4_step5500_eval_audit_run_summary.tsv"),
    )
    args = parser.parse_args()

    records = _scan_records(args.outputs_root)
    grouped = _group_records(args.outputs_root, records)
    relaxed = _relaxed_rows(grouped, max_step=args.max_step, max_epoch=args.max_epoch)
    _write_tsv(args.output_all, relaxed)
    _write_best_tsv(args.output_best, relaxed)
    _write_run_summary_tsv(args.output_run_summary, relaxed)

    print(f"raw_records={len(records)} grouped={len(grouped)} relaxed={len(relaxed)}")
    print(f"wrote {args.output_all}")
    print(f"wrote {args.output_best}")
    print(f"wrote {args.output_run_summary}")
    for row in relaxed[:20]:
        print(
            f"{_float_or_nan(row.get('pdms')):.6f}\t{row['run']}\t{row['checkpoint_id']}"
            f"\tstep={row['step']}\tepoch={row['epoch']}"
        )


if __name__ == "__main__":
    main()
