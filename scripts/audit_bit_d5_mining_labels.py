#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


METRIC_KEYS = ("pdm", "dac", "nc", "ttc", "ego", "comfort")
HIST_BINS = (-1.0, -0.5, -0.2, -0.05, -1e-9, 1e-9, 0.05, 0.2, 0.5, 1.0)
SHARED_EXPERIMENT_ROOT = Path("/mnt/project/VLA-AD/experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit D5 counterfactual mining labels before D5 training.")
    parser.add_argument("--counterfactual-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assert_not_shared_experiment_output(path: Path) -> None:
    resolved = path.resolve()
    shared = SHARED_EXPERIMENT_ROOT.resolve()
    if resolved == shared or shared in resolved.parents:
        raise RuntimeError(f"Refusing to write audit outputs under shared experiment root: {shared}")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def metric_value(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(metrics, dict) or metrics.get(key) is None:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def metric_delta(row: Dict[str, Any], key: str) -> Optional[float]:
    for container in ("delta_metrics", "delta"):
        value = row.get(container)
        if isinstance(value, dict) and value.get(key) is not None:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    base = metric_value(row.get("base_metrics"), key)
    bit = metric_value(row.get("bit_metrics"), key)
    if base is None or bit is None:
        return None
    return bit - base


def hist_label(value: float) -> str:
    lower = "-inf"
    for edge in HIST_BINS:
        if value <= edge:
            return f"({lower},{edge}]"
        lower = str(edge)
    return f"({lower},inf)"


def delta_histograms(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    histograms: Dict[str, Counter[str]] = {key: Counter() for key in METRIC_KEYS}
    missing: Counter[str] = Counter()
    for row in rows:
        for key in METRIC_KEYS:
            delta = metric_delta(row, key)
            if delta is None:
                missing[key] += 1
            else:
                histograms[key][hist_label(delta)] += 1
    return {
        key: {"missing": missing[key], **dict(counter)}
        for key, counter in histograms.items()
    }


def tags_for(row: Dict[str, Any]) -> List[str]:
    tags = row.get("tags") or row.get("bit_counterfactual_tags") or []
    return sorted(str(tag) for tag in tags) or ["neutral"]


def is_hard_nc(row: Dict[str, Any]) -> bool:
    tags = set(tags_for(row))
    if "hard_nc_regression" in tags or "bit_nc_regression" in tags:
        return True
    base = metric_value(row.get("base_metrics"), "nc")
    bit = metric_value(row.get("bit_metrics"), "nc")
    return base is not None and bit is not None and base > 1e-9 and bit <= 1e-9


def is_hard_ttc(row: Dict[str, Any]) -> bool:
    tags = set(tags_for(row))
    if "hard_ttc_regression" in tags or "bit_ttc_regression" in tags:
        return True
    base = metric_value(row.get("base_metrics"), "ttc")
    bit = metric_value(row.get("bit_metrics"), "ttc")
    return base is not None and bit is not None and base > 1e-9 and bit <= 1e-9


def is_soft_safety(row: Dict[str, Any]) -> bool:
    tags = set(tags_for(row))
    return bool(tags & {"soft_safety_regression", "soft_ttc_regression"}) or is_hard_nc(row) or is_hard_ttc(row)


def is_dac_fix(row: Dict[str, Any]) -> bool:
    tags = set(tags_for(row))
    if "dac_fix" in tags or "bit_dac_fix" in tags:
        return True
    base = metric_value(row.get("base_metrics"), "dac")
    bit = metric_value(row.get("bit_metrics"), "dac")
    return base is not None and bit is not None and ((base <= 1e-9 and bit > 1e-9) or bit > base + 0.2)


def is_zero_fix(row: Dict[str, Any]) -> bool:
    tags = set(tags_for(row))
    if "zero_fix" in tags or "bit_zero_fix" in tags:
        return True
    base = metric_value(row.get("base_metrics"), "pdm")
    bit = metric_value(row.get("bit_metrics"), "pdm")
    return base is not None and bit is not None and base <= 1e-9 and bit > 1e-9


def gate_status(summary: Dict[str, Any]) -> Dict[str, Any]:
    hard_nc = int(summary["hard_nc_regressions"])
    hard_ttc = int(summary["hard_ttc_regressions"])
    soft = int(summary["soft_safety_regressions"])
    dac = int(summary["dac_fixes"])
    ideal = hard_nc >= 50 and hard_ttc >= 150 and dac >= 200
    acceptable = hard_nc >= 30 and hard_ttc >= 100 and soft >= 500 and dac >= 200
    return {
        "ideal_gate_passed": ideal,
        "acceptable_gate_passed": acceptable,
        "gate_passed": ideal or acceptable,
        "too_sparse_for_d5": not (ideal or acceptable),
    }


def load_metadata(jsonl_path: Path) -> Dict[str, Any]:
    metadata_path = jsonl_path.parent / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def build_audit(rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    split_counts = Counter(str(row.get("split") or "missing") for row in rows)
    tag_counts = Counter(tag for row in rows for tag in tags_for(row))
    tag_combinations = Counter("+".join(tags_for(row)) for row in rows)
    metric_availability = {
        "base_metrics": sum(1 for row in rows if isinstance(row.get("base_metrics"), dict) and row.get("base_metrics")),
        "bit_metrics": sum(1 for row in rows if isinstance(row.get("bit_metrics"), dict) and row.get("bit_metrics")),
    }
    suspicious: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        issues: List[str] = []
        if not row.get("sample_token") and not row.get("scene_token"):
            issues.append("missing_sample_and_scene_token")
        if "test" in str(row.get("split", "")).lower():
            issues.append("test_split_row")
        if not isinstance(row.get("base_metrics"), dict) or not row.get("base_metrics"):
            issues.append("missing_base_metrics")
        if not isinstance(row.get("bit_metrics"), dict) or not row.get("bit_metrics"):
            issues.append("missing_bit_metrics")
        if row.get("base_pred_traj") is None:
            issues.append("missing_base_pred_traj")
        if row.get("bit_pred_traj") is None:
            issues.append("missing_bit_pred_traj")
        if issues:
            suspicious.append(
                {
                    "row_index": index,
                    "sample_token": row.get("sample_token"),
                    "scene_token": row.get("scene_token"),
                    "split": row.get("split"),
                    "issues": issues,
                }
            )
    summary = {
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(suspicious),
        "split_counts": dict(split_counts),
        "metadata_training_allowed": metadata.get("training_allowed"),
        "training_allowed": bool(metadata.get("training_allowed") is True and not any("test" in split.lower() for split in split_counts)),
        "navtest_present": any("test" in split.lower() for split in split_counts),
        "metric_availability": metric_availability,
        "hard_nc_regressions": sum(1 for row in rows if is_hard_nc(row)),
        "hard_ttc_regressions": sum(1 for row in rows if is_hard_ttc(row)),
        "soft_safety_regressions": sum(1 for row in rows if is_soft_safety(row)),
        "dac_fixes": sum(1 for row in rows if is_dac_fix(row)),
        "zero_fixes": sum(1 for row in rows if is_zero_fix(row)),
        "tag_counts": dict(tag_counts),
        "tag_combinations": dict(tag_combinations),
        "delta_histograms": delta_histograms(rows),
        "suspicious_row_count": len(suspicious),
        "metadata": metadata,
    }
    summary.update(gate_status(summary))
    summary["suspicious_rows"] = suspicious[:200]
    return summary


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    gate = "PASS" if summary["gate_passed"] else "BLOCK"
    lines = [
        "# BiT D5 Mining Label Audit",
        "",
        f"Total rows: `{summary['total_rows']}`",
        f"Valid rows: `{summary['valid_rows']}`",
        f"Training allowed: `{summary['training_allowed']}`",
        f"Navtest present: `{summary['navtest_present']}`",
        f"D5 mining gate: `{gate}`",
        "",
        "## Counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
        f"| hard NC regressions | {summary['hard_nc_regressions']} |",
        f"| hard TTC regressions | {summary['hard_ttc_regressions']} |",
        f"| soft safety regressions | {summary['soft_safety_regressions']} |",
        f"| DAC fixes | {summary['dac_fixes']} |",
        f"| zero fixes | {summary['zero_fixes']} |",
        "",
        "## Split Distribution",
        "",
        "```json",
        json.dumps(summary["split_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Metric Availability",
        "",
        "```json",
        json.dumps(summary["metric_availability"], indent=2, sort_keys=True),
        "```",
        "",
        "## Tag Combinations",
        "",
        "```json",
        json.dumps(summary["tag_combinations"], indent=2, sort_keys=True),
        "```",
        "",
        "## Delta Histograms",
        "",
        "```json",
        json.dumps(summary["delta_histograms"], indent=2, sort_keys=True),
        "```",
        "",
        "## Suspicious Rows",
        "",
        f"Suspicious rows: `{summary['suspicious_row_count']}`. Full list is in `suspicious_rows.jsonl`.",
        "",
        "## Conclusion",
        "",
        "Labels are too sparse for D5 training." if summary["too_sparse_for_d5"] else "Labels pass the D5 mining gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assert_not_shared_experiment_output(args.output_dir)
    rows = list(read_jsonl(args.counterfactual_jsonl))
    metadata = load_metadata(args.counterfactual_jsonl)
    summary = build_audit(rows, metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "delta_histograms.json").write_text(json.dumps(summary["delta_histograms"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "suspicious_rows.jsonl", summary["suspicious_rows"])
    write_report(args.output_dir / "audit_report.md", summary)
    print(json.dumps({key: summary[key] for key in (
        "total_rows",
        "valid_rows",
        "training_allowed",
        "navtest_present",
        "hard_nc_regressions",
        "hard_ttc_regressions",
        "soft_safety_regressions",
        "dac_fixes",
        "too_sparse_for_d5",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
