#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EPS = 1e-9


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rows(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = eval_dir / "per_sample_metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No per_sample_metrics.jsonl in {eval_dir}")
    rows = {}
    for row in read_jsonl(path):
        key = str(row.get("sample_token") or row.get("scene_token") or "")
        if not key:
            continue
        rows[key] = {
            "sample_token": row.get("sample_token"),
            "scene_token": row.get("scene_token"),
            "score": as_float(row.get("score")),
            "dac": as_float(row.get("drivable_area_compliance")),
            "nc": as_float(row.get("no_at_fault_collisions")),
            "ttc": as_float(row.get("time_to_collision_within_bound")),
            "ego": as_float(row.get("ego_progress")),
            "comfort": as_float(row.get("comfort")),
        }
    return rows


def mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def zero_count(values: List[float]) -> int:
    return int(sum(1 for value in values if value <= EPS))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    dac = [float(row["dac"]) for row in rows if row.get("dac") is not None]
    nc = [float(row["nc"]) for row in rows if row.get("nc") is not None]
    ttc = [float(row["ttc"]) for row in rows if row.get("ttc") is not None]
    ego = [float(row["ego"]) for row in rows if row.get("ego") is not None]
    comfort = [float(row["comfort"]) for row in rows if row.get("comfort") is not None]
    return {
        "num_samples": len(rows),
        "mean_pdms": mean(scores),
        "median_pdms": median(scores),
        "p10_pdms": percentile(scores, 10),
        "zero_score_count": zero_count(scores),
        "drivable_area_compliance_zero_count": zero_count(dac),
        "no_at_fault_collision_zero_count": zero_count(nc),
        "time_to_collision_zero_count": zero_count(ttc),
        "ego_progress_mean": mean(ego),
        "comfort_mean": mean(comfort),
    }


def safety_allows_bit(a0: Dict[str, Any], bit: Dict[str, Any]) -> bool:
    for metric in ("nc", "ttc"):
        base = a0.get(metric)
        cur = bit.get(metric)
        if base is not None and cur is not None and base > EPS and cur <= EPS:
            return False
    return True


def choose_rows(a0_rows: Dict[str, Dict[str, Any]], bit_rows: Dict[str, Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    out = []
    for key in sorted(set(a0_rows) & set(bit_rows)):
        a0 = a0_rows[key]
        bit = bit_rows[key]
        if mode == "a0":
            chosen = dict(a0)
            source = "a0"
        elif mode == "bit":
            chosen = dict(bit)
            source = "bit"
        elif mode == "oracle_best":
            use_bit = (bit.get("score") or -1.0) >= (a0.get("score") or -1.0)
            chosen = dict(bit if use_bit else a0)
            source = "bit" if use_bit else "a0"
        elif mode == "safety_oracle":
            use_bit = safety_allows_bit(a0, bit)
            chosen = dict(bit if use_bit else a0)
            source = "bit" if use_bit else "a0"
        else:
            raise ValueError(mode)
        chosen["selection_source"] = source
        chosen["key"] = key
        out.append(chosen)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["method", "num_samples", "mean_pdms", "median_pdms", "p10_pdms", "zero_score_count", "drivable_area_compliance_zero_count", "no_at_fault_collision_zero_count", "time_to_collision_zero_count", "ego_progress_mean", "comfort_mean"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_md(path: Path, summaries: List[Dict[str, Any]], bit_name: str) -> None:
    lines = [
        "# BiT Oracle Fallback Analysis",
        "",
        f"Bit run: `{bit_name}`",
        "",
        "| Method | Mean | Median | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row.get('mean_pdms')} | {row.get('median_pdms')} | {row.get('p10_pdms')} | "
            f"{row.get('zero_score_count')} | {row.get('drivable_area_compliance_zero_count')} | "
            f"{row.get('no_at_fault_collision_zero_count')} | {row.get('time_to_collision_zero_count')} | "
            f"{row.get('ego_progress_mean')} |"
        )
    lines.extend([
        "",
        "SafetyConstrainedOracle is analysis only. It uses per-sample ground-truth metrics and must not be treated as deployable.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute oracle and safety-constrained fallback upper bounds.")
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--bit-dir", type=Path, required=True)
    parser.add_argument("--b6-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def analyze_one(a0_rows: Dict[str, Dict[str, Any]], bit_rows: Dict[str, Dict[str, Any]], bit_name: str, output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    selections = {}
    for mode in ("a0", "bit", "oracle_best", "safety_oracle"):
        rows = choose_rows(a0_rows, bit_rows, mode)
        selections[mode] = rows
        summary = summarize(rows)
        summary["method"] = f"{bit_name}_{mode}" if mode not in {"a0"} else "a0"
        summaries.append(summary)
        write_jsonl(output_dir / f"{bit_name}_{mode}_selection.jsonl", rows)
    write_csv(output_dir / f"{bit_name}_oracle_fallback_summary.csv", summaries)
    write_md(output_dir / f"{bit_name}_oracle_fallback_report.md", summaries, bit_name)
    return summaries


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a0 = load_rows(args.a0_dir)
    bit = load_rows(args.bit_dir)
    summaries = analyze_one(a0, bit, "b5", args.output_dir)
    if args.b6_dir:
        summaries.extend(analyze_one(a0, load_rows(args.b6_dir), "b6", args.output_dir))
    (args.output_dir / "oracle_fallback_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(summaries)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
