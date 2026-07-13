#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_METRICS = (
    "support_precision_auc",
    "density_corrected_support_recall_auc",
    "support_f1_auc",
    "relative_dispersion_ratio",
    "relative_dispersion_calibration",
    "normalized_effective_mode_coverage",
    "kernel_effective_mode_coverage_auc",
    "snsad",
    "raw_pairwise_ade_m",
    "mean_gt_ade_m",
    "best_gt_ade_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and compare support-aligned policy-diversity shards.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="MODEL_LABEL=directory-or-scene_metrics.csv; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260711)
    return parser.parse_args()


def _csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("shard_*/scene_metrics.csv"))
    if not files and (path / "scene_metrics.csv").is_file():
        files = [path / "scene_metrics.csv"]
    if not files:
        raise FileNotFoundError(f"No scene_metrics.csv files under {path}")
    return files


def load_model(spec: str) -> tuple[str, pd.DataFrame, list[str]]:
    if "=" not in spec:
        raise ValueError(f"--model must be LABEL=PATH, got {spec!r}.")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty model label in {spec!r}.")
    paths = _csv_paths(Path(raw_path))
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.empty or frame["token"].duplicated().any():
        raise ValueError(f"Model {label!r} has empty or duplicate scene rows.")
    return label, frame.sort_values("token").reset_index(drop=True), [str(path) for path in paths]


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    draws = rng.choice(finite, size=(samples, finite.size), replace=True).mean(axis=1)
    return {
        "mean": float(finite.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _strata(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", frame
    modes = pd.to_numeric(frame["reference_mode_count"], errors="coerce")
    yield "mode_1", frame[modes <= 1.0]
    yield "mode_2_3", frame[(modes >= 2.0) & (modes <= 3.0)]
    yield "mode_4_plus", frame[modes >= 4.0]


def summarize_model(frame: pd.DataFrame, rng: np.random.Generator, samples: int) -> dict:
    summary: dict[str, object] = {"num_scenes": int(len(frame)), "strata": {}}
    for stratum, subset in _strata(frame):
        metric_summary = {
            metric: _bootstrap_mean_ci(pd.to_numeric(subset[metric], errors="coerce").to_numpy(), rng, samples)
            for metric in PRIMARY_METRICS
            if metric in subset
        }
        summary["strata"][stratum] = {"num_scenes": int(len(subset)), "metrics": metric_summary}
    return summary


def paired_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    rng: np.random.Generator,
    samples: int,
) -> dict:
    merged = baseline.merge(candidate, on="token", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise ValueError("Models do not contain exactly the same scene tokens.")
    metrics: dict[str, object] = {}
    for metric in PRIMARY_METRICS:
        baseline_key = f"{metric}_baseline"
        candidate_key = f"{metric}_candidate"
        if baseline_key not in merged or candidate_key not in merged:
            continue
        delta = pd.to_numeric(merged[candidate_key], errors="coerce").to_numpy() - pd.to_numeric(
            merged[baseline_key], errors="coerce"
        ).to_numpy()
        finite = delta[np.isfinite(delta)]
        ci = _bootstrap_mean_ci(finite, rng, samples)
        ci["candidate_win_rate"] = float(np.mean(finite > 0.0)) if finite.size else float("nan")
        metrics[metric] = ci
    return {"num_aligned_scenes": int(len(merged)), "metrics": metrics}


def oracle_normalized_gain(
    gt_repeat: pd.DataFrame,
    support_oracle: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    min_oracle_gap: float = 0.05,
) -> dict:
    merged = (
        gt_repeat.merge(support_oracle, on="token", suffixes=("_gt", "_oracle"), validate="one_to_one")
        .merge(candidate, on="token", validate="one_to_one")
    )
    metrics: dict[str, object] = {}
    for metric in (
        "density_corrected_support_recall_auc",
        "kernel_effective_mode_coverage_auc",
        "relative_dispersion_calibration",
        "snsad",
    ):
        gap = pd.to_numeric(merged[f"{metric}_oracle"], errors="coerce") - pd.to_numeric(
            merged[f"{metric}_gt"], errors="coerce"
        )
        valid = np.isfinite(gap) & (gap > min_oracle_gap)
        gain = (
            pd.to_numeric(merged.loc[valid, metric], errors="coerce")
            - pd.to_numeric(merged.loc[valid, f"{metric}_gt"], errors="coerce")
        ) / gap[valid]
        gain = gain[np.isfinite(gain)]
        metrics[metric] = {
            "num_scenes": int(gain.size),
            "raw_mean": float(gain.mean()) if gain.size else float("nan"),
            "raw_median": float(gain.median()) if gain.size else float("nan"),
            "clipped_0_1_mean": float(gain.clip(0.0, 1.0).mean()) if gain.size else float("nan"),
            "positive_scene_ratio": float((gain > 0.0).mean()) if gain.size else float("nan"),
        }
    return {"min_oracle_gap": min_oracle_gap, "metrics": metrics}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    models = [load_model(spec) for spec in args.model]
    payload: dict[str, object] = {
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "models": {},
        "comparisons": {},
        "oracle_normalized_gains": {},
    }
    for label, frame, paths in models:
        frame.to_csv(args.output_dir / f"{label}_scene_metrics.csv", index=False)
        payload["models"][label] = {"inputs": paths, **summarize_model(frame, rng, args.bootstrap_samples)}
    baseline_label, baseline_frame, _ = models[0]
    for label, frame, _ in models[1:]:
        payload["comparisons"][f"{label}_minus_{baseline_label}"] = paired_comparison(
            baseline_frame,
            frame,
            rng,
            args.bootstrap_samples,
        )
    model_frames = {label: frame for label, frame, _ in models}
    if "gt_repeat" in model_frames and "support_oracle" in model_frames:
        for label, frame in model_frames.items():
            if label in {"gt_repeat", "support_oracle"}:
                continue
            payload["oracle_normalized_gains"][label] = oracle_normalized_gain(
                model_frames["gt_repeat"],
                model_frames["support_oracle"],
                frame,
            )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
