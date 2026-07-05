#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


STAT_KEYS = {
    "candidate_count",
    "selected_count",
    "valid_ratio",
    "selected_valid_ratio",
    "pareto_front_ratio",
    "selected_pareto_ratio",
    "selected_source_diversity",
    "feasibility_cost_mean",
    "best_valid_minus_gt",
    "best_valid_minus_il",
    "best_selected_minus_gt",
    "selected_min_reward",
    "selected_first_xy_error_m",
    "selected_max_xy_turn_rad",
    "selected_early_xy_turn_rad",
    "selected_semantic_final_heading_error_rad",
    "selected_semantic_path_angle_error_rad",
    "selected_semantic_endpoint_lateral_error_m",
}

COUNT_RATIO_KEYS = {
    "has_valid_candidate_ratio": "has_valid_candidate_count",
    "best_valid_above_gt_ratio": "best_valid_above_gt_count",
    "best_valid_above_il_ratio": "best_valid_above_il_count",
    "best_selected_above_gt_ratio": "best_selected_above_gt_count",
    "selected_has_improver_over_gt_ratio": "selected_has_improver_over_gt_count",
    "selected_source_diversity_ge2_ratio": "selected_source_diversity_ge2_count",
    "external_candidate_scene_ratio": "external_candidate_scene_count",
    "external_selected_scene_ratio": "external_selected_scene_count",
    "scenes_with_selected_low_reward_ratio": "scenes_with_selected_low_reward_count",
    "scenes_with_selected_first_point_mismatch_ratio": "scenes_with_selected_first_point_mismatch_scene_count",
    "scenes_with_selected_local_kink_ratio": "scenes_with_selected_local_kink_scene_count",
    "scenes_with_selected_semantic_mismatch_ratio": "scenes_with_selected_semantic_mismatch_scene_count",
    "poor_gt_few_candidates_ratio": "poor_gt_few_candidates_count",
}

COUNTER_KEYS = {
    "version_counts",
    "source_bucket_counts",
    "selected_source_bucket_counts",
    "source_counts_top20",
    "selected_source_counts_top20",
    "support_tag_counts",
}


def _merge_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    parts = [row[key] for row in rows if isinstance(row.get(key), dict) and row[key].get("count", 0) > 0]
    if not parts:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    total = sum(float(part["count"]) for part in parts)
    return {
        "count": int(total),
        "min": float(min(float(part["min"]) for part in parts)),
        "mean": float(sum(float(part["mean"]) * float(part["count"]) for part in parts) / max(total, 1.0)),
        # Exact global quantiles require raw values. This weighted approximation is used only for summary display.
        "p50": float(sum(float(part["p50"]) * float(part["count"]) for part in parts) / max(total, 1.0)),
        "p90": float(sum(float(part["p90"]) * float(part["count"]) for part in parts) / max(total, 1.0)),
        "max": float(max(float(part["max"]) for part in parts)),
    }


def _merge_counter(rows: list[dict[str, Any]], key: str, limit: int | None = None) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update({str(k): int(v) for k, v in dict(row.get(key, {})).items()})
    if limit is not None:
        return dict(counter.most_common(limit))
    return dict(counter)


def merge_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No audit rows to merge.")
    record_count = sum(int(row.get("record_count", 0)) for row in rows)
    candidate_total = sum(sum(int(v) for v in dict(row.get("source_bucket_counts", {})).values()) for row in rows)
    selected_total = sum(sum(int(v) for v in dict(row.get("support_tag_counts", {})).values()) for row in rows)
    out: dict[str, Any] = {
        "support_archive_path": rows[0].get("support_archive_path", ""),
        "record_count": int(record_count),
        "load_errors": int(sum(int(row.get("load_errors", 0)) for row in rows)),
        "shape_errors": int(sum(int(row.get("shape_errors", 0)) for row in rows)),
        "rank_count": len(rows),
    }
    for key in STAT_KEYS:
        out[key] = _merge_stats(rows, key)
    for key, count_key in COUNT_RATIO_KEYS.items():
        count = sum(float(row.get(key, 0.0)) * float(row.get("record_count", 0)) for row in rows)
        out[count_key] = int(round(count))
        out[key] = float(count / max(record_count, 1))
    for key in COUNTER_KEYS:
        limit = 20 if key.endswith("top20") else None
        out[key] = _merge_counter(rows, key, limit=limit)
    out["fallback_tag_ratio"] = float(out["support_tag_counts"].get("fallback_best", 0) / max(selected_total, 1))
    out["external_candidate_ratio"] = float(out["source_bucket_counts"].get("external", 0) / max(candidate_total, 1))
    out["selected_low_reward_count"] = int(sum(int(row.get("selected_low_reward_count", 0)) for row in rows))
    out["selected_first_point_mismatch_count"] = int(
        sum(int(row.get("selected_first_point_mismatch_count", 0)) for row in rows)
    )
    out["selected_local_kink_count"] = int(sum(int(row.get("selected_local_kink_count", 0)) for row in rows))
    out["selected_semantic_mismatch_count"] = int(
        sum(int(row.get("selected_semantic_mismatch_count", 0)) for row in rows)
    )
    examples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for key, value in dict(row.get("examples", {})).items():
            examples.setdefault(key, []).extend(list(value))
            examples[key] = examples[key][:8]
    out["examples"] = examples
    warnings = []
    if out["load_errors"] or out["shape_errors"]:
        warnings.append("archive contains load/shape errors")
    if out["has_valid_candidate_ratio"] < 0.99:
        warnings.append("valid candidate coverage is below 99%")
    if out["selected_valid_ratio"]["mean"] < 0.90:
        warnings.append("selected support contains too many invalid candidates")
    if out["scenes_with_selected_low_reward_ratio"] > 0.0:
        warnings.append("selected support contains non-GT trajectories below the configured PDMS/reward threshold")
    if out["scenes_with_selected_first_point_mismatch_ratio"] > 0.0:
        warnings.append("selected support contains non-GT trajectories whose first point is far from GT")
    if out["scenes_with_selected_local_kink_ratio"] > 0.0:
        warnings.append("selected support contains non-GT locally kinked trajectories")
    if out["scenes_with_selected_semantic_mismatch_ratio"] > 0.0:
        warnings.append("selected support contains non-GT trajectories with turn/heading semantics inconsistent with GT")
    out["warnings"] = warnings
    return out


def write_markdown(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# SG-FPS Parallel Support Archive Audit",
        "",
        f"- archive: `{metrics['support_archive_path']}`",
        f"- records: `{metrics['record_count']}`",
        f"- rank count: `{metrics['rank_count']}`",
        f"- selected valid ratio mean: `{metrics['selected_valid_ratio']['mean']:.6f}`",
        f"- selected non-GT low reward scene ratio: `{metrics['scenes_with_selected_low_reward_ratio']:.6f}`",
        f"- selected non-GT first-point mismatch scene ratio: `{metrics['scenes_with_selected_first_point_mismatch_ratio']:.6f}`",
        f"- selected non-GT local kink scene ratio: `{metrics['scenes_with_selected_local_kink_ratio']:.6f}`",
        f"- selected non-GT semantic mismatch scene ratio: `{metrics['scenes_with_selected_semantic_mismatch_ratio']:.6f}`",
        f"- selected improver over GT ratio: `{metrics['selected_has_improver_over_gt_ratio']:.6f}`",
        f"- fallback tag ratio: `{metrics['fallback_tag_ratio']:.6f}`",
        "",
        "## Support Tags",
        "",
        "```json",
        json.dumps(metrics["support_tag_counts"], indent=2, sort_keys=True),
        "```",
    ]
    if metrics.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in metrics["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded SG-FPS support audit JSON files.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    paths = sorted(Path().glob(args.input_glob) if not Path(args.input_glob).is_absolute() else Path("/").glob(args.input_glob.lstrip("/")))
    rows = [json.load(open(path, "r", encoding="utf-8")) for path in paths]
    merged = merge_audits(rows)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md:
        write_markdown(merged, Path(args.output_md))
    print(json.dumps(merged, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
