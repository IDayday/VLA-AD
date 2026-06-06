from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METRIC_NAMES = ("score", "dac", "nc", "ttc", "progress", "comfort")
METRIC_ALIASES = {
    "score": ("score", "pdm", "pdms", "pdm_score"),
    "dac": ("dac", "drivable_area_compliance"),
    "nc": ("nc", "no_at_fault_collisions"),
    "ttc": ("ttc", "time_to_collision_within_bound"),
    "progress": ("progress", "ego", "ego_progress"),
    "comfort": ("comfort",),
}
TRAINING_FORBIDDEN_SPLITS = ("test", "navtest")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(row: Dict[str, Any], name: str) -> float:
    for key in METRIC_ALIASES[name]:
        if key in row:
            return _as_float(row.get(key), 0.0)
    return 1.0 if name in {"dac", "nc", "ttc", "comfort"} else 0.0


def normalize_metrics(row: Dict[str, Any]) -> Dict[str, float]:
    return {name: _metric(row, name) for name in METRIC_NAMES}


def is_zero(value: float, eps: float) -> bool:
    return value <= eps


def split_is_forbidden(split: str) -> bool:
    split_l = split.lower()
    return any(item in split_l for item in TRAINING_FORBIDDEN_SPLITS)


def strategy_name_for_mode(mode: str) -> str:
    mode_l = mode.lower()
    if mode_l.startswith("base"):
        return "base"
    if mode_l.startswith("bit"):
        return "path_intent"
    if "interaction" in mode_l:
        return "interaction"
    if "progress" in mode_l:
        return "progress"
    if "comfort" in mode_l:
        return "comfort"
    if "risk" in mode_l:
        return "risk_vla"
    return mode


def candidate_rank_key(mode: str) -> Tuple[int, str]:
    order = {
        "base_det": 0,
        "bit_det": 1,
        "bit_stochastic_seed0": 2,
        "bit_stochastic_seed1": 3,
        "bit_stochastic_seed2": 4,
    }
    return order.get(mode, 100), mode


def tail_risk(metrics: Dict[str, float], low_score_threshold: float, progress_threshold: float, comfort_threshold: float, eps: float) -> bool:
    return bool(
        metrics["score"] <= low_score_threshold
        or is_zero(metrics["dac"], eps)
        or is_zero(metrics["nc"], eps)
        or is_zero(metrics["ttc"], eps)
        or metrics["progress"] < progress_threshold
        or metrics["comfort"] < comfort_threshold
    )


def row_utility(row: Dict[str, Any], nc_penalty: float, ttc_penalty: float) -> float:
    value = float(row.get("pdm_score") or 0.0)
    if row.get("regresses_nc"):
        value -= nc_penalty
    if row.get("regresses_ttc"):
        value -= ttc_penalty
    return value


def choose_anchors(rows: List[Dict[str, Any]], nc_penalty: float, ttc_penalty: float) -> Tuple[int, int]:
    safe_rows = [row for row in rows if not row["regresses_nc"] and not row["regresses_ttc"]]
    positive_pool = safe_rows if safe_rows else rows
    positive = max(positive_pool, key=lambda row: row_utility(row, nc_penalty, ttc_penalty))
    risky_rows = [row for row in rows if row["regresses_nc"] or row["regresses_ttc"] or row["tail_risk_label"]]
    negative_pool = risky_rows if risky_rows else rows
    negative = min(negative_pool, key=lambda row: row_utility(row, nc_penalty, ttc_penalty))
    return int(positive["candidate_id"]), int(negative["candidate_id"])


def load_counterfactual_rows(path: Path, purpose: str) -> Dict[str, List[Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            token = str(row.get("sample_token") or row.get("token") or row.get("scene_token") or "")
            if not token:
                raise ValueError(f"Missing sample token at {path}:{line_number}")
            split = str(row.get("split") or "")
            if purpose == "training" and split_is_forbidden(split):
                raise RuntimeError(f"Refusing to build training labels from split={split!r} at {path}:{line_number}")
            grouped[token].append(row)
    return grouped


def fit_trajectory(value: Any, horizon: int, action_dim: int) -> np.ndarray:
    arr = np.asarray(value or [], dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((0, action_dim), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, action_dim) if arr.size % action_dim == 0 else arr.reshape(-1, 1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] < horizon:
        pad = np.zeros((horizon - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:horizon]
    if arr.shape[1] < action_dim:
        pad = np.zeros((horizon, action_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :action_dim].astype(np.float32, copy=False)


def build_assets(
    input_jsonl: Path,
    output_dir: Path,
    split: str,
    purpose: str,
    max_tokens: Optional[int],
    horizon: int,
    action_dim: int,
    low_score_threshold: float,
    progress_threshold: float,
    comfort_threshold: float,
    zero_eps: float,
    nc_penalty: float,
    ttc_penalty: float,
) -> Dict[str, Any]:
    grouped = load_counterfactual_rows(input_jsonl, purpose)
    tokens = sorted(grouped)
    if max_tokens is not None:
        tokens = tokens[: int(max_tokens)]

    output_dir.mkdir(parents=True, exist_ok=True)
    label_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    candidate_tensors: List[np.ndarray] = []
    split_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    skipped_tokens: List[str] = []

    for token in tokens:
        raw_rows = sorted(grouped[token], key=lambda item: candidate_rank_key(str(item.get("candidate_mode") or item.get("counterfactual_candidate_mode") or "")))
        if not raw_rows:
            continue
        token_split = str(raw_rows[0].get("split") or split)
        split_counts[token_split] += 1
        base_metrics = normalize_metrics(raw_rows[0].get("base_metrics") or {})
        token_label_rows: List[Dict[str, Any]] = []
        token_trajs: List[np.ndarray] = []
        for candidate_id, raw in enumerate(raw_rows):
            mode = str(raw.get("candidate_mode") or raw.get("counterfactual_candidate_mode") or f"candidate_{candidate_id}")
            mode_counts[mode] += 1
            metrics = normalize_metrics(raw.get("candidate_metrics") or {})
            row: Dict[str, Any] = {
                "sample_token": token,
                "scene_token": raw.get("scene_token"),
                "split": token_split,
                "purpose": purpose,
                "candidate_id": candidate_id,
                "candidate_name": mode,
                "strategy_name": strategy_name_for_mode(mode),
                "trajectory_source_path": str(input_jsonl),
                "baseline_candidate": "base_det",
                "pdm_score": metrics["score"],
                "dac": metrics["dac"],
                "nc": metrics["nc"],
                "ttc": metrics["ttc"],
                "progress": metrics["progress"],
                "comfort": metrics["comfort"],
                "sampling_bucket": raw.get("sampling_bucket"),
                "sampling_stratum": raw.get("sampling_stratum"),
                "tags": raw.get("tags") or raw.get("bit_counterfactual_tags") or [],
            }
            for metric_name in METRIC_NAMES:
                row[f"delta_{metric_name}"] = metrics[metric_name] - base_metrics[metric_name]
            row.update(
                {
                    "repairs_path": is_zero(base_metrics["dac"], zero_eps) and not is_zero(metrics["dac"], zero_eps),
                    "regresses_path": not is_zero(base_metrics["dac"], zero_eps) and is_zero(metrics["dac"], zero_eps),
                    "repairs_nc": is_zero(base_metrics["nc"], zero_eps) and not is_zero(metrics["nc"], zero_eps),
                    "regresses_nc": not is_zero(base_metrics["nc"], zero_eps) and is_zero(metrics["nc"], zero_eps),
                    "repairs_ttc": is_zero(base_metrics["ttc"], zero_eps) and not is_zero(metrics["ttc"], zero_eps),
                    "regresses_ttc": not is_zero(base_metrics["ttc"], zero_eps) and is_zero(metrics["ttc"], zero_eps),
                    "tail_risk_label": tail_risk(metrics, low_score_threshold, progress_threshold, comfort_threshold, zero_eps),
                }
            )
            token_label_rows.append(row)
            token_trajs.append(fit_trajectory(raw.get("candidate_pred_traj"), horizon, action_dim))
            metadata_rows.append(
                {
                    "sample_token": token,
                    "candidate_id": candidate_id,
                    "candidate_name": mode,
                    "strategy_name": row["strategy_name"],
                    "source": "counterfactual_candidates",
                    "split": token_split,
                }
            )
        if not token_label_rows:
            skipped_tokens.append(token)
            continue
        positive_id, negative_id = choose_anchors(token_label_rows, nc_penalty, ttc_penalty)
        for row in token_label_rows:
            row["constrained_best_strategy"] = positive_id
            row["positive_anchor"] = positive_id
            row["negative_anchor"] = negative_id
            label_rows.append(row)
        positive = token_label_rows[positive_id]
        negative = token_label_rows[negative_id]
        if positive_id != negative_id:
            pair_rows.append(
                {
                    "sample_token": token,
                    "positive_candidate_id": positive_id,
                    "negative_candidate_id": negative_id,
                    "positive_strategy_name": positive.get("strategy_name"),
                    "negative_strategy_name": negative.get("strategy_name"),
                    "positive_score": positive.get("pdm_score"),
                    "negative_score": negative.get("pdm_score"),
                    "negative_regresses_nc": negative.get("regresses_nc"),
                    "negative_regresses_ttc": negative.get("regresses_ttc"),
                    "positive_repairs_path": positive.get("repairs_path"),
                    "tail_risk_label": bool(positive.get("tail_risk_label") or negative.get("tail_risk_label")),
                }
            )
        candidate_tensors.append(np.stack(token_trajs, axis=0))

    labels_jsonl = output_dir / "strategy_utility_labels.jsonl"
    with labels_jsonl.open("w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    labels_csv = output_dir / "strategy_utility_labels.csv"
    if label_rows:
        with labels_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_rows[0].keys()))
            writer.writeheader()
            writer.writerows(label_rows)

    metadata_jsonl = output_dir / "candidate_metadata.jsonl"
    with metadata_jsonl.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    pairs_jsonl = output_dir / "safe_alignment_pairs.jsonl"
    with pairs_jsonl.open("w", encoding="utf-8") as handle:
        for row in pair_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    token_array = np.asarray(tokens[: len(candidate_tensors)], dtype="U64")
    trajectories = np.stack(candidate_tensors, axis=0) if candidate_tensors else np.zeros((0, 0, horizon, action_dim), dtype=np.float32)
    np.savez_compressed(output_dir / "candidate_trajectories.npz", sample_tokens=token_array, trajectories=trajectories)

    by_candidate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        by_candidate[str(row["candidate_name"])].append(row)
    candidate_summary = []
    for name, rows in sorted(by_candidate.items(), key=lambda item: candidate_rank_key(item[0])):
        candidate_summary.append(
            {
                "candidate_name": name,
                "rows": len(rows),
                "mean_pdms": mean(float(row["pdm_score"]) for row in rows) if rows else None,
                "zero_score": sum(float(row["pdm_score"]) <= zero_eps for row in rows),
                "dac0": sum(is_zero(float(row["dac"]), zero_eps) for row in rows),
                "nc0": sum(is_zero(float(row["nc"]), zero_eps) for row in rows),
                "ttc0": sum(is_zero(float(row["ttc"]), zero_eps) for row in rows),
                "repairs_path": sum(bool(row["repairs_path"]) for row in rows),
                "regresses_path": sum(bool(row["regresses_path"]) for row in rows),
                "repairs_nc": sum(bool(row["repairs_nc"]) for row in rows),
                "regresses_nc": sum(bool(row["regresses_nc"]) for row in rows),
                "repairs_ttc": sum(bool(row["repairs_ttc"]) for row in rows),
                "regresses_ttc": sum(bool(row["regresses_ttc"]) for row in rows),
                "tail_risk": sum(bool(row["tail_risk_label"]) for row in rows),
            }
        )
    positive_counts = Counter(str(row["candidate_name"]) for row in label_rows if int(row["candidate_id"]) == int(row["positive_anchor"]))
    negative_counts = Counter(str(row["candidate_name"]) for row in label_rows if int(row["candidate_id"]) == int(row["negative_anchor"]))
    summary = {
        "input_jsonl": str(input_jsonl),
        "split": split,
        "purpose": purpose,
        "num_tokens": len(candidate_tensors),
        "num_label_rows": len(label_rows),
        "num_pairs": len(pair_rows),
        "candidate_tensor_shape": list(trajectories.shape),
        "split_counts": dict(split_counts),
        "mode_counts": dict(mode_counts),
        "positive_anchor_counts": dict(positive_counts),
        "negative_anchor_counts": dict(negative_counts),
        "skipped_tokens": skipped_tokens[:50],
        "strategy_utility_labels_jsonl": str(labels_jsonl),
        "strategy_utility_labels_csv": str(labels_csv),
        "candidate_trajectories_npz": str(output_dir / "candidate_trajectories.npz"),
        "candidate_metadata_jsonl": str(metadata_jsonl),
        "safe_alignment_pairs_jsonl": str(pairs_jsonl),
        "candidate_summary": candidate_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output_dir / "strategy_utility_label_report.md", summary)
    return summary


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Counterfactual Candidate Asset Report",
        "",
        f"Input: `{summary['input_jsonl']}`",
        f"Split: `{summary['split']}`",
        f"Purpose: `{summary['purpose']}`",
        f"Tokens: `{summary['num_tokens']}`",
        f"Utility label rows: `{summary['num_label_rows']}`",
        f"Safe alignment pairs: `{summary['num_pairs']}`",
        f"Candidate tensor shape: `{summary['candidate_tensor_shape']}`",
        "",
        "| Candidate | Rows | Mean PDMS | Zero | DAC0 | NC0 | TTC0 | Path Repair | Path Regression | NC Regression | TTC Regression | Tail Risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["candidate_summary"]:
        report_item = dict(item)
        report_item["mean_pdms"] = float(item["mean_pdms"] or 0.0)
        lines.append(
            "| {candidate_name} | {rows} | {mean_pdms:.6f} | {zero_score} | {dac0} | {nc0} | {ttc0} | "
            "{repairs_path} | {regresses_path} | {regresses_nc} | {regresses_ttc} | {tail_risk} |".format(
                **report_item,
            )
        )
    lines.extend(
        [
            "",
            "Positive anchor counts:",
            json.dumps(summary["positive_anchor_counts"], sort_keys=True),
            "",
            "Negative anchor counts:",
            json.dumps(summary["negative_anchor_counts"], sort_keys=True),
            "",
            "Leakage guard: training mode rejects rows whose split name contains `test` or `navtest`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build RISK-VLA v2 training assets from train-side counterfactual candidates.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--low-score-threshold", type=float, default=0.5)
    parser.add_argument("--progress-threshold", type=float, default=0.5)
    parser.add_argument("--comfort-threshold", type=float, default=0.5)
    parser.add_argument("--zero-eps", type=float, default=1e-9)
    parser.add_argument("--nc-penalty", type=float, default=2.0)
    parser.add_argument("--ttc-penalty", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    grouped = load_counterfactual_rows(args.input_jsonl, args.purpose)
    if args.dry_run:
        per_token = Counter(len(rows) for rows in grouped.values())
        print(
            json.dumps(
                {
                    "input_jsonl": str(args.input_jsonl),
                    "tokens": len(grouped),
                    "rows": sum(len(rows) for rows in grouped.values()),
                    "candidates_per_token": dict(per_token),
                    "would_write": str(args.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0

    summary = build_assets(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        split=args.split,
        purpose=args.purpose,
        max_tokens=args.max_tokens,
        horizon=args.horizon,
        action_dim=args.action_dim,
        low_score_threshold=args.low_score_threshold,
        progress_threshold=args.progress_threshold,
        comfort_threshold=args.comfort_threshold,
        zero_eps=args.zero_eps,
        nc_penalty=args.nc_penalty,
        ttc_penalty=args.ttc_penalty,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
