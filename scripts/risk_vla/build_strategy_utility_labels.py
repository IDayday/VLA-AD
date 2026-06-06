from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCORE_KEYS = ("score", "pdm_score", "pdms", "mean_pdms")
DAC_KEYS = ("drivable_area_compliance", "dac", "pdm_dac")
NC_KEYS = ("no_at_fault_collisions", "nc", "pdm_nc")
TTC_KEYS = ("time_to_collision_within_bound", "ttc", "pdm_ttc")
PROGRESS_KEYS = ("ego_progress", "progress")
COMFORT_KEYS = ("comfort",)
TOKEN_KEYS = ("sample_token", "token", "scene_token")
METRIC_NAMES = ("score", "dac", "nc", "ttc", "progress", "comfort")


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(row: Dict[str, Any], keys: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in row:
            value = _as_float(row.get(key), default=None)
            if value is not None:
                return value
    return default


def token_from_row(row: Dict[str, Any]) -> str:
    for key in TOKEN_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Could not find token in row keys={sorted(row.keys())}")


def metrics_from_row(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "score": _first(row, SCORE_KEYS, 0.0),
        "dac": _first(row, DAC_KEYS, 1.0),
        "nc": _first(row, NC_KEYS, 1.0),
        "ttc": _first(row, TTC_KEYS, 1.0),
        "progress": _first(row, PROGRESS_KEYS, 1.0),
        "comfort": _first(row, COMFORT_KEYS, 1.0),
    }


@dataclass(frozen=True)
class CandidateInput:
    name: str
    path: Path
    strategy_name: str
    trajectory_source_path: Optional[str] = None
    source_method: Optional[str] = None
    source_checkpoint: Optional[str] = None


def parse_candidate(value: str) -> CandidateInput:
    parts = value.split(",")
    fields: Dict[str, str] = {}
    if len(parts) == 1 and "=" not in parts[0]:
        path = Path(parts[0])
        return CandidateInput(path.stem, path, path.stem)
    for part in parts:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"Candidate spec must be name=...,path=..., got {value!r}")
        key, item = part.split("=", 1)
        fields[key.strip()] = item.strip()
    if "path" not in fields:
        raise argparse.ArgumentTypeError(f"Candidate spec missing path=...: {value!r}")
    name = fields.get("name") or Path(fields["path"]).stem
    return CandidateInput(
        name=name,
        path=Path(fields["path"]),
        strategy_name=fields.get("strategy") or name,
        trajectory_source_path=fields.get("trajectory_source_path"),
        source_method=fields.get("source_method") or fields.get("method"),
        source_checkpoint=fields.get("source_checkpoint") or fields.get("checkpoint"),
    )


def read_table(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out[token_from_row(row)] = metrics_from_row(row)
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[token_from_row(row)] = metrics_from_row(row)
    return out


def fail_if_training_leakage(path: Optional[Path], split: str, purpose: str) -> None:
    if purpose != "training":
        return
    split_l = split.lower()
    if "test" in split_l or "navtest" in split_l:
        raise RuntimeError(f"Refusing to build training utility labels from split={split!r}")
    if path is None:
        return
    metadata_path = path.parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta_split = str(metadata.get("split", "")).lower()
        if "test" in meta_split or "navtest" in meta_split or bool(metadata.get("analysis_only", False)):
            raise RuntimeError(f"Refusing to use analysis/test labels for training: {metadata_path}")


def is_zero(value: Optional[float], eps: float) -> bool:
    return value is not None and value <= eps


def tail_risk(metrics: Dict[str, Optional[float]], low_score_threshold: float, progress_threshold: float, comfort_threshold: float, eps: float) -> bool:
    return bool(
        (metrics.get("score") is not None and metrics["score"] <= low_score_threshold)
        or is_zero(metrics.get("dac"), eps)
        or is_zero(metrics.get("nc"), eps)
        or is_zero(metrics.get("ttc"), eps)
        or (metrics.get("progress") is not None and metrics["progress"] < progress_threshold)
        or (metrics.get("comfort") is not None and metrics["comfort"] < comfort_threshold)
    )


def delta(candidate: Dict[str, Optional[float]], baseline: Dict[str, Optional[float]], key: str) -> Optional[float]:
    c = candidate.get(key)
    b = baseline.get(key)
    if c is None or b is None:
        return None
    return c - b


def utility(candidate: Dict[str, Any], nc_penalty: float, ttc_penalty: float) -> float:
    """Default v3 candidate utility: reward repairs/recovery and heavily penalize safety regressions."""

    del nc_penalty, ttc_penalty  # Kept in the API for older callers/tests.
    value = float(candidate.get("delta_score_vs_a0") or candidate.get("delta_score") or 0.0)
    value += 0.5 * float(bool(candidate.get("path_repair") or candidate.get("repairs_path")))
    value += 0.4 * float(bool(candidate.get("zero_repair")))
    value += 0.4 * float(bool(candidate.get("ttc_repair") or candidate.get("repairs_ttc")))
    value += 0.2 * float(bool(candidate.get("progress_recovery")))
    value += 0.1 * float(bool(candidate.get("comfort_recovery")))
    value -= 2.5 * float(bool(candidate.get("nc_regression") or candidate.get("regresses_nc")))
    value -= 2.0 * float(bool(candidate.get("ttc_regression") or candidate.get("regresses_ttc")))
    value -= 1.0 * float(bool(candidate.get("path_regression") or candidate.get("regresses_path")))
    value -= 1.0 * float(bool(candidate.get("zero_regression")))
    value -= 0.5 * float(bool(candidate.get("comfort_regression")))
    return value


def choose_anchors(rows: List[Dict[str, Any]], nc_penalty: float, ttc_penalty: float) -> Tuple[str, str, bool]:
    safe_rows = [row for row in rows if not row["regresses_nc"] and not row["regresses_ttc"]]
    pool = safe_rows if safe_rows else rows
    positive = max(pool, key=lambda row: utility(row, nc_penalty, ttc_penalty))
    risky = [row for row in rows if row["regresses_nc"] or row["regresses_ttc"] or row["tail_risk_label"]]
    negative_pool = risky if risky else rows
    negative = min(negative_pool, key=lambda row: utility(row, nc_penalty, ttc_penalty))
    return str(positive["candidate_id"]), str(negative["candidate_id"]), not bool(safe_rows)


def build_labels(
    baseline: CandidateInput,
    candidates: Sequence[CandidateInput],
    split: str,
    purpose: str,
    low_score_threshold: float = 0.5,
    progress_threshold: float = 0.5,
    comfort_threshold: float = 0.5,
    zero_eps: float = 1e-9,
    nc_penalty: float = 2.0,
    ttc_penalty: float = 1.0,
    max_tokens: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    fail_if_training_leakage(baseline.path, split, purpose)
    baseline_rows = read_table(baseline.path)
    candidate_tables = [(candidate, read_table(candidate.path)) for candidate in candidates]
    tokens = sorted(set(baseline_rows).intersection(*(set(table) for _, table in candidate_tables)))
    if max_tokens is not None:
        tokens = tokens[: int(max_tokens)]
    output_rows: List[Dict[str, Any]] = []
    for token in tokens:
        base_metrics = baseline_rows[token]
        token_rows: List[Dict[str, Any]] = []
        for candidate_id, (candidate, table) in enumerate(candidate_tables):
            metrics = table[token]
            row: Dict[str, Any] = {
                "sample_token": token,
                "token": token,
                "token_id": token,
                "split": split,
                "purpose": purpose,
                "candidate_id": candidate_id,
                "strategy_name": candidate.strategy_name,
                "trajectory_source_path": candidate.trajectory_source_path,
                "source_method": candidate.source_method or candidate.name,
                "source_checkpoint": candidate.source_checkpoint,
                "baseline_candidate": baseline.name,
                "candidate_name": candidate.name,
                "score": metrics["score"],
                "pdm_score": metrics["score"],
                "dac": metrics["dac"],
                "drivable_area_compliance": metrics["dac"],
                "nc": metrics["nc"],
                "no_at_fault_collisions": metrics["nc"],
                "ttc": metrics["ttc"],
                "time_to_collision_within_bound": metrics["ttc"],
                "prog": metrics["progress"],
                "progress": metrics["progress"],
                "ego_progress": metrics["progress"],
                "comfort": metrics["comfort"],
            }
            for metric_name in METRIC_NAMES:
                row[f"delta_{metric_name}"] = delta(metrics, base_metrics, metric_name)
                row[f"delta_{metric_name}_vs_a0"] = row[f"delta_{metric_name}"]
            row["delta_dac_vs_a0"] = row["delta_dac"]
            row["delta_nc_vs_a0"] = row["delta_nc"]
            row["delta_ttc_vs_a0"] = row["delta_ttc"]
            row.update(
                {
                    "zero_repair": is_zero(base_metrics.get("score"), zero_eps) and not is_zero(metrics.get("score"), zero_eps),
                    "zero_regression": not is_zero(base_metrics.get("score"), zero_eps) and is_zero(metrics.get("score"), zero_eps),
                    "repairs_path": is_zero(base_metrics.get("dac"), zero_eps) and not is_zero(metrics.get("dac"), zero_eps),
                    "regresses_path": not is_zero(base_metrics.get("dac"), zero_eps) and is_zero(metrics.get("dac"), zero_eps),
                    "repairs_nc": is_zero(base_metrics.get("nc"), zero_eps) and not is_zero(metrics.get("nc"), zero_eps),
                    "regresses_nc": not is_zero(base_metrics.get("nc"), zero_eps) and is_zero(metrics.get("nc"), zero_eps),
                    "repairs_ttc": is_zero(base_metrics.get("ttc"), zero_eps) and not is_zero(metrics.get("ttc"), zero_eps),
                    "regresses_ttc": not is_zero(base_metrics.get("ttc"), zero_eps) and is_zero(metrics.get("ttc"), zero_eps),
                    "tail_risk_label": tail_risk(metrics, low_score_threshold, progress_threshold, comfort_threshold, zero_eps),
                }
            )
            row["path_repair"] = row["repairs_path"]
            row["path_regression"] = row["regresses_path"]
            row["nc_repair"] = row["repairs_nc"]
            row["nc_regression"] = row["regresses_nc"]
            row["ttc_repair"] = row["repairs_ttc"]
            row["ttc_regression"] = row["regresses_ttc"]
            row["progress_recovery"] = bool((row["delta_progress"] or 0.0) > 0.0 and (base_metrics.get("progress") or 0.0) < progress_threshold)
            row["comfort_recovery"] = bool((row["delta_comfort"] or 0.0) > 0.0 and (base_metrics.get("comfort") or 0.0) < comfort_threshold)
            row["comfort_regression"] = bool(row["delta_comfort"] is not None and row["delta_comfort"] < 0.0)
            row["unsafe_path"] = bool(row["regresses_path"] or is_zero(metrics.get("dac"), zero_eps))
            row["unsafe_nc"] = bool(row["regresses_nc"] or is_zero(metrics.get("nc"), zero_eps))
            row["unsafe_ttc"] = bool(row["regresses_ttc"] or is_zero(metrics.get("ttc"), zero_eps))
            row["unsafe_any"] = bool(row["unsafe_path"] or row["unsafe_nc"] or row["unsafe_ttc"] or row["tail_risk_label"])
            row["unsafe_regression"] = bool(row["regresses_path"] or row["regresses_nc"] or row["regresses_ttc"] or row["zero_regression"])
            row["safe_path_repair"] = bool(row["repairs_path"] and not row["regresses_nc"] and not row["regresses_ttc"])
            row["utility_score"] = utility(row, nc_penalty, ttc_penalty)
            token_rows.append(row)
        positive, negative, no_safe_candidate = choose_anchors(token_rows, nc_penalty, ttc_penalty)
        constrained = positive
        for row in token_rows:
            row["constrained_best_strategy"] = constrained
            row["constrained_best_candidate"] = constrained
            row["no_safe_candidate"] = no_safe_candidate
            row["positive_anchor"] = positive
            row["negative_anchor"] = negative
            if str(row["candidate_id"]) == positive and str(row["candidate_id"]) == negative:
                row["pair_type"] = "positive_negative"
            elif str(row["candidate_id"]) == positive:
                row["pair_type"] = "positive"
            elif str(row["candidate_id"]) == negative:
                row["pair_type"] = "negative"
            else:
                row["pair_type"] = "neutral"
            output_rows.append(row)
    summary = {
        "baseline": baseline.name,
        "split": split,
        "purpose": purpose,
        "num_tokens": len(tokens),
        "num_rows": len(output_rows),
        "candidate_names": [candidate.name for candidate in candidates],
        "positive_anchor_counts": {},
        "negative_anchor_counts": {},
    }
    for row in output_rows:
        if row["candidate_id"] == row["positive_anchor"]:
            summary["positive_anchor_counts"][row["candidate_name"]] = summary["positive_anchor_counts"].get(row["candidate_name"], 0) + 1
        if row["candidate_id"] == row["negative_anchor"]:
            summary["negative_anchor_counts"][row["candidate_name"]] = summary["negative_anchor_counts"].get(row["candidate_name"], 0) + 1
    return output_rows, summary


def write_outputs(rows: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "strategy_utility_labels.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    csv_path = output_dir / "strategy_utility_labels.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_name"]), []).append(row)
    candidate_summary = []
    for name, group in sorted(by_candidate.items()):
        candidate_summary.append(
            {
                "candidate_name": name,
                "rows": len(group),
                "mean_pdm_score": mean([float(row["pdm_score"] or 0.0) for row in group]) if group else None,
                "regresses_nc": sum(bool(row["regresses_nc"]) for row in group),
                "regresses_ttc": sum(bool(row["regresses_ttc"]) for row in group),
                "repairs_path": sum(bool(row["repairs_path"]) for row in group),
                "tail_risk": sum(bool(row["tail_risk_label"]) for row in group),
            }
        )
    summary["candidate_summary"] = candidate_summary
    summary["strategy_utility_labels_jsonl"] = str(jsonl_path)
    summary["strategy_utility_labels_csv"] = str(csv_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Strategy Utility Label Report",
        "",
        f"Split: `{summary['split']}`",
        f"Purpose: `{summary['purpose']}`",
        f"Tokens: `{summary['num_tokens']}`",
        f"Rows: `{summary['num_rows']}`",
        "",
        "| Candidate | Rows | Mean PDMS | Path Repairs | NC Regressions | TTC Regressions | Tail Risk Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in candidate_summary:
        lines.append(
            f"| {item['candidate_name']} | {item['rows']} | {item['mean_pdm_score']} | "
            f"{item['repairs_path']} | {item['regresses_nc']} | {item['regresses_ttc']} | {item['tail_risk']} |"
        )
    lines.extend(
        [
            "",
            "Leakage guard: training purpose rejects `test`/`navtest` split names and analysis-only metadata.",
        ]
    )
    (output_dir / "strategy_utility_label_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build RISK-VLA v2 strategy utility labels from matched PDM tables.")
    parser.add_argument("--baseline", type=parse_candidate, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows, summary = build_labels(args.baseline, args.candidate, args.split, args.purpose, max_tokens=args.max_tokens)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.min_real_samples and summary["num_tokens"] < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(
            f"Refusing formal utility-label construction with {summary['num_tokens']} matched tokens; "
            f"minimum is {args.min_real_samples}. Use --debug-allow-small only for debug-only runs."
        )
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
