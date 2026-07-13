#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


REFERENCE_FIELDS = ("scalar", "ep", "ttc", "quality", "ddc")
SUPPORT_FIELDS = (
    "num_supports",
    "reference_mode_count",
    "support_dispersion",
    "support_reward_mean",
    "support_reward_max",
    "policy_dispersion",
    "raw_pairwise_ade_m",
    "support_precision_auc",
    "density_corrected_support_recall_auc",
    "kernel_effective_mode_coverage_auc",
    "snsad",
)


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _finite_spearman(left: pd.Series, right: pd.Series) -> float | None:
    valid = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(right.to_numpy(dtype=float))
    if valid.sum() < 2:
        return None
    left_valid = left[valid]
    right_valid = right[valid]
    if left_valid.nunique() < 2 or right_valid.nunique() < 2:
        return None
    value = left_valid.corr(right_valid, method="spearman")
    return None if not np.isfinite(value) else float(value)


def analyze(
    checkpoint_path: Path,
    reference_cache_path: Path,
    support_scene_metrics_path: Path | None,
) -> Dict[str, Any]:
    checkpoint = _load(checkpoint_path)
    curriculum = checkpoint.get("lfp_frontier_curriculum")
    if not isinstance(curriculum, dict):
        raise KeyError("checkpoint has no lfp_frontier_curriculum state")
    store = curriculum.get("frontier_store", {})
    states = store.get("states") if isinstance(store, dict) else None
    if not isinstance(states, dict) or not states:
        raise ValueError("checkpoint frontier state is empty")

    reference_payload = _load(reference_cache_path)
    records = reference_payload.get("records") if isinstance(reference_payload, dict) else None
    if not isinstance(records, dict):
        raise ValueError("reference cache has no records mapping")

    missing = sorted(set(states) - set(records))
    if missing:
        raise KeyError(f"reference cache is missing {len(missing)} frontier tokens; first={missing[0]}")

    rows = []
    for token, state in states.items():
        selected = records[token]["selected"]
        row: Dict[str, Any] = {
            "token": token,
            "fast_ema": float(state["fast_ema"]),
            "slow_ema": float(state["slow_ema"]),
            "seen_count": int(state["seen_count"]),
            "selected_source_code": int(records[token]["selected_source_code"]),
            "gt_stage2_abs_scalar_gap": abs(
                float(records[token]["gt"]["scalar"])
                - float(records[token]["stage2"]["scalar"])
            ),
        }
        row.update({f"reference_{field}": float(selected[field]) for field in REFERENCE_FIELDS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["learning_progress"] = (frame["fast_ema"] - frame["slow_ema"]).abs()

    matched_support = 0
    if support_scene_metrics_path is not None:
        support = pd.read_csv(support_scene_metrics_path)
        if "token" not in support.columns:
            raise ValueError("support scene metrics must contain a token column")
        available = [field for field in SUPPORT_FIELDS if field in support.columns]
        frame = frame.merge(support[["token", *available]], on="token", how="left", validate="one_to_one")
        matched_support = int(frame[available[0]].notna().sum()) if available else 0

    correlation_fields = [
        *(f"reference_{field}" for field in REFERENCE_FIELDS),
        "gt_stage2_abs_scalar_gap",
        "selected_source_code",
        *(field for field in SUPPORT_FIELDS if field in frame.columns),
    ]
    correlations = {
        field: {
            "spearman_fast_ema": _finite_spearman(frame["fast_ema"], frame[field]),
            "spearman_learning_progress": _finite_spearman(frame["learning_progress"], frame[field]),
            "mean": float(frame[field].mean()),
            "count": int(frame[field].notna().sum()),
        }
        for field in correlation_fields
    }

    ranked = frame["fast_ema"].rank(method="first")
    frame["energy_quartile"] = pd.qcut(ranked, 4, labels=False)
    quartile_fields = [
        "fast_ema",
        "slow_ema",
        "learning_progress",
        "reference_scalar",
        "reference_ep",
        "gt_stage2_abs_scalar_gap",
        *(field for field in SUPPORT_FIELDS if field in frame.columns),
    ]
    quartiles = {
        str(int(quartile)): {
            field: float(value)
            for field, value in group[quartile_fields].mean(numeric_only=True).items()
        }
        for quartile, group in frame.groupby("energy_quartile", sort=True)
    }

    weights = curriculum.get("sampler", {}).get("weights")
    sampler: Dict[str, float] = {}
    if isinstance(weights, torch.Tensor) and weights.numel() > 0:
        probabilities = weights.double() / weights.double().sum()
        sampler = {
            "entropy": float(-(probabilities * probabilities.clamp_min(1e-300).log()).sum()),
            "effective_sample_size": float(1.0 / probabilities.square().sum()),
            "max_to_median": float(weights.max() / weights.median().clamp_min(1e-12)),
        }

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "reference_cache": str(reference_cache_path.resolve()),
        "support_scene_metrics": (
            str(support_scene_metrics_path.resolve()) if support_scene_metrics_path is not None else None
        ),
        "num_frontier_scenes": int(len(frame)),
        "num_support_matched_scenes": matched_support,
        "fast_ema_zero_ratio": float((frame["fast_ema"] == 0.0).mean()),
        "fast_ema_mean": float(frame["fast_ema"].mean()),
        "learning_progress_mean": float(frame["learning_progress"].mean()),
        "correlations": correlations,
        "energy_quartiles": quartiles,
        "sampler": sampler,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--support-scene-metrics", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = analyze(args.checkpoint, args.reference_cache, args.support_scene_metrics)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
