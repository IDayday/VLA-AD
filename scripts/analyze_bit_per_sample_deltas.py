#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EPS = 1e-9


RUNS = (
    ("step1", "Step1"),
    ("step2", "Step2"),
    ("step3", "Step3"),
)


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def discover_per_sample_file(eval_dir: Path) -> Path:
    candidates = [
        eval_dir / "per_sample_metrics.jsonl",
        eval_dir / "per_sample_metrics.csv",
        eval_dir / "pdm_per_sample.jsonl",
        eval_dir / "pdm_results.csv",
        eval_dir / "predictions.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No per-sample metrics file found in {eval_dir}")


def load_rows(eval_dir: Path) -> List[Dict[str, Any]]:
    path = discover_per_sample_file(eval_dir)
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix == ".csv":
        rows = read_csv(path)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        raise ValueError(f"Unsupported per-sample file: {path}")
    return [normalize_row(row, source_file=path) for row in rows]


def metric_from_row(row: Dict[str, Any], *names: str) -> Optional[float]:
    pdm = row.get("pdm") if isinstance(row.get("pdm"), dict) else {}
    for name in names:
        value = as_float(row.get(name))
        if value is not None:
            return value
        value = as_float(pdm.get(name))
        if value is not None:
            return value
    return None


def normalize_row(row: Dict[str, Any], *, source_file: Path) -> Dict[str, Any]:
    score = metric_from_row(row, "score", "pdm_score", "PDMS")
    valid_raw = row.get("valid", True)
    if isinstance(valid_raw, str):
        valid = valid_raw.lower() not in {"false", "0", "no", "nan"}
    else:
        valid = bool(valid_raw)
    return {
        "key": str(row.get("sample_token") or row.get("scene_token") or ""),
        "sample_token": row.get("sample_token"),
        "scene_token": row.get("scene_token"),
        "chunk": row.get("chunk"),
        "valid": valid,
        "score": score,
        "drivable_area_compliance": metric_from_row(row, "drivable_area_compliance", "DAC"),
        "no_at_fault_collisions": metric_from_row(row, "no_at_fault_collisions", "NC"),
        "time_to_collision_within_bound": metric_from_row(row, "time_to_collision_within_bound", "TTC"),
        "ego_progress": metric_from_row(row, "ego_progress", "EP"),
        "comfort": metric_from_row(row, "comfort"),
        "trajectory_l1": metric_from_row(row, "trajectory_l1"),
        "source_file": str(source_file),
    }


def index_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("key")
        if key:
            indexed[str(key)] = row
    return indexed


def is_zero(row: Dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    return value is not None and float(value) <= EPS


def score_zero(row: Dict[str, Any]) -> bool:
    return is_zero(row, "score")


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def transition_counts(a0_rows: Dict[str, Dict[str, Any]], run_rows: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "a0_dac0_to_run_dac_nonzero": 0,
        "a0_dac_nonzero_to_run_dac0": 0,
        "a0_nc0_to_run_nc_nonzero": 0,
        "a0_nc_nonzero_to_run_nc0": 0,
        "a0_ttc0_to_run_ttc_nonzero": 0,
        "a0_ttc_nonzero_to_run_ttc0": 0,
        "a0_zero_to_run_nonzero": 0,
        "a0_nonzero_to_run_zero": 0,
    }
    for key in sorted(set(a0_rows) & set(run_rows)):
        a0 = a0_rows[key]
        run = run_rows[key]
        if not a0.get("valid") or not run.get("valid"):
            continue
        a0_dac0 = is_zero(a0, "drivable_area_compliance")
        run_dac0 = is_zero(run, "drivable_area_compliance")
        a0_nc0 = is_zero(a0, "no_at_fault_collisions")
        run_nc0 = is_zero(run, "no_at_fault_collisions")
        a0_ttc0 = is_zero(a0, "time_to_collision_within_bound")
        run_ttc0 = is_zero(run, "time_to_collision_within_bound")
        a0_z = score_zero(a0)
        run_z = score_zero(run)
        counts["a0_dac0_to_run_dac_nonzero"] += int(a0_dac0 and not run_dac0)
        counts["a0_dac_nonzero_to_run_dac0"] += int(not a0_dac0 and run_dac0)
        counts["a0_nc0_to_run_nc_nonzero"] += int(a0_nc0 and not run_nc0)
        counts["a0_nc_nonzero_to_run_nc0"] += int(not a0_nc0 and run_nc0)
        counts["a0_ttc0_to_run_ttc_nonzero"] += int(a0_ttc0 and not run_ttc0)
        counts["a0_ttc_nonzero_to_run_ttc0"] += int(not a0_ttc0 and run_ttc0)
        counts["a0_zero_to_run_nonzero"] += int(a0_z and not run_z)
        counts["a0_nonzero_to_run_zero"] += int(not a0_z and run_z)
    return counts


def paired_deltas(a0_rows: Dict[str, Dict[str, Any]], run_rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    deltas: List[Dict[str, Any]] = []
    for key in sorted(set(a0_rows) & set(run_rows)):
        a0 = a0_rows[key]
        run = run_rows[key]
        if not a0.get("valid") or not run.get("valid"):
            continue
        if a0.get("score") is None or run.get("score") is None:
            continue
        delta = float(run["score"]) - float(a0["score"])
        deltas.append({
            "key": key,
            "sample_token": run.get("sample_token") or a0.get("sample_token"),
            "scene_token": run.get("scene_token") or a0.get("scene_token"),
            "a0_score": a0.get("score"),
            "run_score": run.get("score"),
            "delta_pdms": delta,
            "a0_dac": a0.get("drivable_area_compliance"),
            "run_dac": run.get("drivable_area_compliance"),
            "a0_nc": a0.get("no_at_fault_collisions"),
            "run_nc": run.get("no_at_fault_collisions"),
            "a0_ttc": a0.get("time_to_collision_within_bound"),
            "run_ttc": run.get("time_to_collision_within_bound"),
            "a0_ego": a0.get("ego_progress"),
            "run_ego": run.get("ego_progress"),
        })
    return deltas


def case_row(a0: Dict[str, Any], run: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    return {
        "run": run_name,
        "sample_token": run.get("sample_token") or a0.get("sample_token"),
        "scene_token": run.get("scene_token") or a0.get("scene_token"),
        "a0_score": a0.get("score"),
        "run_score": run.get("score"),
        "delta_pdms": None
        if a0.get("score") is None or run.get("score") is None
        else float(run["score"]) - float(a0["score"]),
        "a0_dac": a0.get("drivable_area_compliance"),
        "run_dac": run.get("drivable_area_compliance"),
        "a0_nc": a0.get("no_at_fault_collisions"),
        "run_nc": run.get("no_at_fault_collisions"),
        "a0_ttc": a0.get("time_to_collision_within_bound"),
        "run_ttc": run.get("time_to_collision_within_bound"),
        "a0_ego": a0.get("ego_progress"),
        "run_ego": run.get("ego_progress"),
    }


def collect_case_lists(
    a0_rows: Dict[str, Dict[str, Any]],
    run_rows: Dict[str, Dict[str, Any]],
    run_name: str,
) -> Dict[str, List[Dict[str, Any]]]:
    cases = {
        "dac_fixed": [],
        "dac_regressed": [],
        "nc_fixed": [],
        "nc_newly_broken": [],
        "ttc_fixed": [],
        "ttc_newly_broken": [],
        "zero_fixed": [],
        "zero_newly_broken": [],
    }
    for key in sorted(set(a0_rows) & set(run_rows)):
        a0 = a0_rows[key]
        run = run_rows[key]
        if not a0.get("valid") or not run.get("valid"):
            continue
        row = case_row(a0, run, run_name)
        a0_dac0 = is_zero(a0, "drivable_area_compliance")
        run_dac0 = is_zero(run, "drivable_area_compliance")
        a0_nc0 = is_zero(a0, "no_at_fault_collisions")
        run_nc0 = is_zero(run, "no_at_fault_collisions")
        a0_ttc0 = is_zero(a0, "time_to_collision_within_bound")
        run_ttc0 = is_zero(run, "time_to_collision_within_bound")
        a0_z = score_zero(a0)
        run_z = score_zero(run)
        if a0_dac0 and not run_dac0:
            cases["dac_fixed"].append(row)
        if not a0_dac0 and run_dac0:
            cases["dac_regressed"].append(row)
        if a0_nc0 and not run_nc0:
            cases["nc_fixed"].append(row)
        if not a0_nc0 and run_nc0:
            cases["nc_newly_broken"].append(row)
        if a0_ttc0 and not run_ttc0:
            cases["ttc_fixed"].append(row)
        if not a0_ttc0 and run_ttc0:
            cases["ttc_newly_broken"].append(row)
        if a0_z and not run_z:
            cases["zero_fixed"].append(row)
        if not a0_z and run_z:
            cases["zero_newly_broken"].append(row)
    for values in cases.values():
        values.sort(key=lambda item: item.get("delta_pdms") or 0.0)
    return cases


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "run",
        "joined_samples",
        "mean_delta_pdms",
        "median_delta_pdms",
        "p10_delta_pdms",
        "improved_samples",
        "regressed_samples",
        "unchanged_samples",
        "a0_dac0_to_run_dac_nonzero",
        "a0_dac_nonzero_to_run_dac0",
        "a0_nc0_to_run_nc_nonzero",
        "a0_nc_nonzero_to_run_nc0",
        "a0_ttc0_to_run_ttc_nonzero",
        "a0_ttc_nonzero_to_run_ttc0",
        "a0_zero_to_run_nonzero",
        "a0_nonzero_to_run_zero",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def make_summary(
    run_key: str,
    run_label: str,
    a0_rows: Dict[str, Dict[str, Any]],
    run_rows: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    deltas = paired_deltas(a0_rows, run_rows)
    delta_values = [float(item["delta_pdms"]) for item in deltas]
    a0_scores = [float(a0_rows[item["key"]]["score"]) for item in deltas]
    run_scores = [float(run_rows[item["key"]]["score"]) for item in deltas]
    transitions = transition_counts(a0_rows, run_rows)
    top_improved = sorted(deltas, key=lambda item: item["delta_pdms"], reverse=True)[:20]
    top_regressed = sorted(deltas, key=lambda item: item["delta_pdms"])[:20]
    summary = {
        "run": run_key,
        "run_label": run_label,
        "joined_samples": len(deltas),
        "mean_delta_pdms": mean(delta_values),
        "median_delta_pdms": median(delta_values),
        "p10_delta_pdms": None
        if percentile(a0_scores, 10) is None or percentile(run_scores, 10) is None
        else percentile(run_scores, 10) - percentile(a0_scores, 10),
        "improved_samples": sum(1 for value in delta_values if value > EPS),
        "regressed_samples": sum(1 for value in delta_values if value < -EPS),
        "unchanged_samples": sum(1 for value in delta_values if abs(value) <= EPS),
        **transitions,
    }
    details = {
        **summary,
        "top20_improved_scenes": top_improved,
        "top20_regressed_scenes": top_regressed,
    }
    return summary, details


def write_transition_md(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# BiT Per-Sample Transition Matrix",
        "",
        "| Run | DAC fixed | DAC regressed | NC fixed | NC newly broken | TTC fixed | TTC newly broken | Zero fixed | Zero newly broken |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['a0_dac0_to_run_dac_nonzero']} | "
            f"{row['a0_dac_nonzero_to_run_dac0']} | {row['a0_nc0_to_run_nc_nonzero']} | "
            f"{row['a0_nc_nonzero_to_run_nc0']} | {row['a0_ttc0_to_run_ttc_nonzero']} | "
            f"{row['a0_ttc_nonzero_to_run_ttc0']} | {row['a0_zero_to_run_nonzero']} | "
            f"{row['a0_nonzero_to_run_zero']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summaries: List[Dict[str, Any]], details: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# BiT v1 Per-Sample Diagnostic Report",
        "",
        "This report compares Step1/Step2/Step3 against A0 on the joined per-sample NAVSIM subset.",
        "",
        "## Delta Summary",
        "",
        "| Run | Joined | Mean Delta | Median Delta | P10 Delta | Improved | Regressed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['joined_samples']} | {fmt(row['mean_delta_pdms'])} | "
            f"{fmt(row['median_delta_pdms'])} | {fmt(row['p10_delta_pdms'])} | "
            f"{row['improved_samples']} | {row['regressed_samples']} |"
        )
    lines.extend([
        "",
        "## Safety Transitions",
        "",
        "| Run | DAC fixed | DAC regressed | NC fixed | NC newly broken | TTC fixed | TTC newly broken | Zero fixed | Zero newly broken |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| {row['run']} | {row['a0_dac0_to_run_dac_nonzero']} | "
            f"{row['a0_dac_nonzero_to_run_dac0']} | {row['a0_nc0_to_run_nc_nonzero']} | "
            f"{row['a0_nc_nonzero_to_run_nc0']} | {row['a0_ttc0_to_run_ttc_nonzero']} | "
            f"{row['a0_ttc_nonzero_to_run_ttc0']} | {row['a0_zero_to_run_nonzero']} | "
            f"{row['a0_nonzero_to_run_zero']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A useful v2 candidate must keep DAC/zero fixes without creating material NC/TTC regressions. "
        "These per-sample transitions should be used before selecting any larger eval or Step4 work.",
        "",
        "## Top Regressions",
    ])
    for run_key, run_details in details.items():
        lines.extend(["", f"### {run_key}", ""])
        for item in run_details["top20_regressed_scenes"][:10]:
            lines.append(
                f"- sample={item.get('sample_token')} scene={item.get('scene_token')} "
                f"delta={fmt(item.get('delta_pdms'))} a0={fmt(item.get('a0_score'))} run={fmt(item.get('run_score'))}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-sample BiT deltas and left-tail transitions.")
    parser.add_argument("--a0-dir", type=Path, required=True)
    parser.add_argument("--step1-dir", type=Path, required=True)
    parser.add_argument("--step2-dir", type=Path, required=True)
    parser.add_argument("--step3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.output_dir / "case_lists"
    case_dir.mkdir(parents=True, exist_ok=True)

    a0 = index_rows(load_rows(args.a0_dir))
    run_dirs = {
        "step1": args.step1_dir,
        "step2": args.step2_dir,
        "step3": args.step3_dir,
    }
    summaries: List[Dict[str, Any]] = []
    detail_json: Dict[str, Dict[str, Any]] = {}
    for run_key, run_label in RUNS:
        run_rows = index_rows(load_rows(run_dirs[run_key]))
        summary, details = make_summary(run_key, run_label, a0, run_rows)
        summaries.append(summary)
        detail_json[run_key] = details
        cases = collect_case_lists(a0, run_rows, run_key)
        for name, rows in cases.items():
            write_jsonl(case_dir / f"{run_key}_{name}.jsonl", rows)

    write_transition_md(args.output_dir / "transition_matrix.md", summaries)
    write_summary_csv(args.output_dir / "bit_delta_summary.csv", summaries)
    (args.output_dir / "bit_delta_summary.json").write_text(
        json.dumps({"summaries": summaries, "details": detail_json}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "BIT_DIAGNOSTIC_REPORT.md", summaries, detail_json)
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(summaries)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
