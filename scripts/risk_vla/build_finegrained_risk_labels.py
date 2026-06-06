from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from scripts.risk_vla.build_strategy_utility_labels import (
    COMFORT_KEYS,
    DAC_KEYS,
    NC_KEYS,
    PROGRESS_KEYS,
    SCORE_KEYS,
    TOKEN_KEYS,
    TTC_KEYS,
    _as_float,
)


RISK_ORDER = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
TIME_BINS = ["none", "early", "mid", "late"]


def _row_value(row: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in row:
            value = _as_float(row.get(key), default=None)
            if value is not None:
                return value
    return default


def _token(row: Dict[str, Any]) -> str:
    for key in TOKEN_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Could not resolve token from row keys={sorted(row.keys())}")


def risk_scene_from_row(row: Dict[str, Any], low_score_threshold: float, progress_threshold: float, comfort_threshold: float, eps: float) -> list[float]:
    score = _row_value(row, SCORE_KEYS, 1.0)
    dac = _row_value(row, DAC_KEYS, 1.0)
    nc = _row_value(row, NC_KEYS, 1.0)
    ttc = _row_value(row, TTC_KEYS, 1.0)
    progress = _row_value(row, PROGRESS_KEYS, 1.0)
    comfort = _row_value(row, COMFORT_KEYS, 1.0)
    return [
        float(score <= low_score_threshold),
        float(dac <= eps),
        float(nc <= eps),
        float(ttc <= eps),
        float(progress < progress_threshold),
        float(comfort < comfort_threshold),
    ]


def event_time_bin(scene: Sequence[float]) -> str:
    if scene[2] or scene[3]:
        return "early"
    if scene[1]:
        return "mid"
    if scene[4] or scene[5] or scene[0]:
        return "late"
    return "none"


def horizon_labels(scene: Sequence[float], horizon: int) -> list[list[float]]:
    labels = [[0.0 for _ in RISK_ORDER] for _ in range(horizon)]
    split1 = max(1, horizon // 3)
    split2 = max(split1 + 1, 2 * horizon // 3)
    for idx, active in enumerate(scene):
        if not active:
            continue
        if idx in (2, 3):
            steps = range(0, split1)
        elif idx == 1:
            steps = range(split1, split2)
        else:
            steps = range(split2, horizon)
        for step in steps:
            labels[step][idx] = 1.0
    return labels


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def build_finegrained_labels(
    pdm_table: Path,
    split: str,
    purpose: str,
    horizon: int = 8,
    low_score_threshold: float = 0.5,
    progress_threshold: float = 0.5,
    comfort_threshold: float = 0.5,
    eps: float = 1e-9,
    max_rows: Optional[int] = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    split_l = split.lower()
    if purpose == "training" and ("test" in split_l or "navtest" in split_l):
        raise RuntimeError(f"Refusing to build training fine-grained labels from split={split!r}")
    rows = []
    counts = {name: 0 for name in RISK_ORDER}
    time_counts = {name: 0 for name in TIME_BINS}
    for idx, row in enumerate(iter_rows(pdm_table)):
        if max_rows is not None and idx >= max_rows:
            break
        scene = risk_scene_from_row(row, low_score_threshold, progress_threshold, comfort_threshold, eps)
        h_labels = horizon_labels(scene, horizon)
        time_bin = event_time_bin(scene)
        safety_horizon = [[step[2], step[3]] for step in h_labels]
        for risk_idx, value in enumerate(scene):
            counts[RISK_ORDER[risk_idx]] += int(bool(value))
        time_counts[time_bin] += 1
        rows.append(
            {
                "sample_token": _token(row),
                "split": split,
                "purpose": purpose,
                "risk_class_order": RISK_ORDER,
                "risk_labels_scene": scene,
                "risk_labels_horizon": h_labels,
                "risk_event_time_bin": time_bin,
                "safety_risk_horizon": safety_horizon,
                "actor_proxy_features": None,
            }
        )
    summary = {
        "pdm_table": str(pdm_table),
        "split": split,
        "purpose": purpose,
        "horizon": horizon,
        "num_rows": len(rows),
        "risk_positive_counts": counts,
        "risk_event_time_bin_counts": time_counts,
    }
    return rows, summary


def write_outputs(rows: list[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "finegrained_risk_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    summary["finegrained_risk_labels_jsonl"] = str(labels_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Fine-Grained Risk Label Report",
        "",
        f"Split: `{summary['split']}`",
        f"Purpose: `{summary['purpose']}`",
        f"Rows: `{summary['num_rows']}`",
        f"Horizon: `{summary['horizon']}`",
        "",
        "| Risk | Positives |",
        "| --- | ---: |",
    ]
    for key, value in summary["risk_positive_counts"].items():
        lines.append(f"| {key} | {value} |")
    (output_dir / "finegrained_risk_label_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build RISK-VLA v2 fine-grained temporal risk labels.")
    parser.add_argument("--pdm-table", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows, summary = build_finegrained_labels(args.pdm_table, args.split, args.purpose, args.horizon, max_rows=args.max_rows)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
