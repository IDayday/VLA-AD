#!/usr/bin/env python3
"""Audit whether an SG-FPS archive is a sound Stage2 teacher distribution."""

from __future__ import annotations

import argparse
import json
import lzma
import math
import multiprocessing as mp
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXTERNAL_PREFIXES = ("ddv2", "driveor", "drivor", "diffusiondrivev2")
METRIC_KEYS = {
    "reward": "rewards",
    "nc": "no_at_fault_collisions",
    "dac": "drivable_area_compliance",
    "ttc": "time_to_collision_within_bound",
    "ep": "ego_progress",
    "comfort": "history_comfort",
    "ddc": "driving_direction_compliance",
}


def _iter_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted((*root.rglob("*.pkl.xz"), *root.rglob("*.pkl")))


def _load(path: Path) -> Mapping[str, Any]:
    opener = lzma.open if path.name.endswith(".pkl.xz") else open
    with opener(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping in {path}, got {type(payload).__name__}.")
    return payload


def _source_bucket(source: str) -> str:
    source = str(source).lower()
    if source == "gt" or source.startswith("gt:"):
        return "gt"
    if source.startswith(EXTERNAL_PREFIXES):
        return "external"
    if source.startswith("progress"):
        return "progress"
    if "lateral" in source:
        return "lateral"
    if source.startswith("timing"):
        return "timing"
    return "other"


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _pairwise_distances(trajs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_xy = trajs[:, None, :, :2] - trajs[None, :, :, :2]
    xy_distance = np.linalg.norm(delta_xy, axis=-1)
    heading = np.abs(_wrap_angle(trajs[:, None, :, 2] - trajs[None, :, :, 2]))
    components = np.stack(
        (
            xy_distance[..., -1],
            xy_distance.mean(axis=-1),
            np.abs(delta_xy[..., 0]).mean(axis=-1),
            np.abs(delta_xy[..., 1]).mean(axis=-1),
            heading.mean(axis=-1),
        ),
        axis=-1,
    )
    scales = np.asarray((3.0, 1.5, 1.5, 0.8, 0.35), dtype=np.float64)
    weights = np.asarray((0.30, 0.30, 0.15, 0.15, 0.10), dtype=np.float64)
    snsad = ((components / scales) * weights).sum(axis=-1)
    return xy_distance.mean(axis=-1), xy_distance[..., -1], snsad


def _component_count(distance: np.ndarray, threshold: float) -> int:
    count = int(distance.shape[0])
    unseen = set(range(count))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            row = stack.pop()
            neighbors = {index for index in unseen if float(distance[row, index]) <= threshold}
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def _heading_tangent_error(traj: np.ndarray, min_segment_length: float = 0.2) -> tuple[float, float] | None:
    origin = np.zeros((1, 3), dtype=np.float64)
    poses = np.concatenate((origin, np.asarray(traj, dtype=np.float64)), axis=0)
    segments = poses[1:, :2] - poses[:-1, :2]
    lengths = np.linalg.norm(segments, axis=-1)
    valid = lengths >= float(min_segment_length)
    if not np.any(valid):
        return None
    tangent = np.arctan2(segments[:, 1], segments[:, 0])
    error = np.abs(_wrap_angle(poses[1:, 2] - tangent))[valid]
    return float(error.mean()), float((error > 0.35).mean())


def _metric_array(record: Mapping[str, Any], key: str, count: int) -> np.ndarray:
    if key == "reward":
        value = record.get("rewards")
    else:
        value = record.get("components", {}).get(METRIC_KEYS[key])
    if value is None:
        return np.full((count,), np.nan, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (count,):
        raise ValueError(f"Metric {key} has shape {array.shape}, expected {(count,)}.")
    return array


def analyze_record(record: Mapping[str, Any]) -> dict[str, Any]:
    candidates = np.asarray(record["candidates"], dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[-1] != 3:
        raise ValueError(f"candidates must be [K,H,3], got {candidates.shape}.")
    count = int(candidates.shape[0])
    sources = [str(source) for source in record["sources"]]
    if len(sources) != count:
        raise ValueError(f"sources has {len(sources)} entries for {count} candidates.")
    selected_indices = np.asarray(record["support_indices"], dtype=np.int64)
    if selected_indices.ndim != 1 or selected_indices.size == 0:
        raise ValueError("support_indices must be a non-empty vector.")
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= count:
        raise ValueError("support_indices contain an out-of-range index.")

    selected = candidates[selected_indices]
    selected_sources = [sources[int(index)] for index in selected_indices]
    buckets = [_source_bucket(source) for source in sources]
    selected_buckets = [_source_bucket(source) for source in selected_sources]
    tags = [str(record.get("support_tags", [""] * count)[int(index)]) for index in selected_indices]
    gt_indices = [index for index, bucket in enumerate(buckets) if bucket == "gt"]
    if len(gt_indices) != 1:
        raise ValueError(f"Expected exactly one GT candidate, found {len(gt_indices)}.")
    gt_index = gt_indices[0]
    gt = candidates[gt_index]

    result: dict[str, Any] = {
        "scene_count": 1,
        "candidate_count": count,
        "selected_count": int(selected_indices.size),
        "pre_pareto_non_external_count": sum(bucket != "external" for bucket in buckets),
        "candidate_buckets": Counter(buckets),
        "selected_buckets": Counter(selected_buckets),
        "selected_source_tag": Counter(zip(selected_buckets, tags)),
        "gt_selected": int(gt_index in selected_indices.tolist()),
        "external_candidate_scene": int("external" in buckets),
        "external_selected_scene": int("external" in selected_buckets),
        "metric_values": {},
        "metric_scene_std": {},
        "metric_saturated": {},
        "gt_relative_ade": [],
        "gt_relative_fde": [],
        "gt_relative_far_gt2": 0,
        "gt_relative_count": 0,
        "heading_error": [],
        "heading_error_gt": [],
        "heading_error_non_gt": [],
        "heading_bad_segment_ratio": [],
    }

    for key in METRIC_KEYS:
        values = _metric_array(record, key, count)[selected_indices]
        finite = values[np.isfinite(values)]
        result["metric_values"][key] = finite.tolist()
        result["metric_scene_std"][key] = float(finite.std()) if finite.size else math.nan
        result["metric_saturated"][key] = (int((finite >= 0.999999).sum()), int(finite.size))

    selected_rewards = _metric_array(record, "reward", count)[selected_indices]
    result["all_selected_reward_one"] = int(np.all(selected_rewards >= 0.999999))
    result["zero_reward_spread"] = int(float(np.nanmax(selected_rewards) - np.nanmin(selected_rewards)) <= 1e-8)

    non_gt_mask = np.asarray(selected_buckets) != "gt"
    if np.any(non_gt_mask):
        delta = selected[non_gt_mask, :, :2] - gt[None, :, :2]
        per_step = np.linalg.norm(delta, axis=-1)
        ade = per_step.mean(axis=-1)
        fde = per_step[:, -1]
        result["gt_relative_ade"] = ade.tolist()
        result["gt_relative_fde"] = fde.tolist()
        result["gt_relative_far_gt2"] = int((ade > 2.0).sum())
        result["gt_relative_count"] = int(ade.size)

    for traj, bucket in zip(selected, selected_buckets):
        heading_error = _heading_tangent_error(traj)
        if heading_error is None:
            continue
        mean_error, bad_ratio = heading_error
        result["heading_error"].append(mean_error)
        result["heading_bad_segment_ratio"].append(bad_ratio)
        result["heading_error_gt" if bucket == "gt" else "heading_error_non_gt"].append(mean_error)

    if selected.shape[0] >= 2:
        ade_matrix, fde_matrix, snsad_matrix = _pairwise_distances(selected)
        upper = np.triu_indices(selected.shape[0], k=1)
        result["pairwise_ade"] = ade_matrix[upper].tolist()
        result["pairwise_fde"] = fde_matrix[upper].tolist()
        result["pairwise_snsad"] = snsad_matrix[upper].tolist()
        masked = snsad_matrix + np.eye(selected.shape[0]) * 1e9
        result["nearest_snsad"] = float(masked.min(axis=1).mean())
        result["mode_count_snsad_040"] = _component_count(snsad_matrix, 0.40)
        result["near_duplicate_pair_count"] = int((snsad_matrix[upper] < 0.10).sum())
        result["pair_count"] = int(len(upper[0]))
    else:
        result.update(
            pairwise_ade=[],
            pairwise_fde=[],
            pairwise_snsad=[],
            nearest_snsad=0.0,
            mode_count_snsad_040=1,
            near_duplicate_pair_count=0,
            pair_count=0,
        )
    return result


def _analyze_path(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return analyze_record(_load(path)), None
    except Exception as exc:  # pragma: no cover - exercised by archive runs
        return None, f"{path}: {exc}"


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def audit_archive(root: Path, *, max_records: int = 0, workers: int = 1) -> dict[str, Any]:
    paths = _iter_paths(root)
    if max_records > 0:
        paths = paths[:max_records]
    counters: Counter[str] = Counter()
    candidate_buckets: Counter[str] = Counter()
    selected_buckets: Counter[str] = Counter()
    selected_source_tag: Counter[tuple[str, str]] = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_scene_std: dict[str, list[float]] = defaultdict(list)
    metric_saturated: Counter[str] = Counter()
    metric_total: Counter[str] = Counter()
    errors: list[str] = []

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            rows = pool.imap_unordered(_analyze_path, paths, chunksize=64)
            iterator = rows
            for row, error in iterator:
                if error:
                    if len(errors) < 20:
                        errors.append(error)
                    counters["load_errors"] += 1
                    continue
                _accumulate(
                    row,
                    counters,
                    candidate_buckets,
                    selected_buckets,
                    selected_source_tag,
                    values,
                    metric_values,
                    metric_scene_std,
                    metric_saturated,
                    metric_total,
                )
    else:
        for path in paths:
            row, error = _analyze_path(path)
            if error:
                if len(errors) < 20:
                    errors.append(error)
                counters["load_errors"] += 1
                continue
            _accumulate(
                row,
                counters,
                candidate_buckets,
                selected_buckets,
                selected_source_tag,
                values,
                metric_values,
                metric_scene_std,
                metric_saturated,
                metric_total,
            )

    scenes = max(counters["scene_count"], 1)
    candidates = max(counters["candidate_count"], 1)
    selected = max(counters["selected_count"], 1)
    candidate_external_share = candidate_buckets["external"] / candidates
    selected_external_share = selected_buckets["external"] / selected
    source_selection_rate = {
        bucket: selected_buckets[bucket] / max(candidate_buckets[bucket], 1)
        for bucket in sorted(candidate_buckets)
    }
    return {
        "archive": str(root),
        "records_requested": len(paths),
        "record_count": counters["scene_count"],
        "load_errors": counters["load_errors"],
        "error_examples": errors,
        "candidate_count": counters["candidate_count"],
        "selected_count": counters["selected_count"],
        "candidate_count_per_scene": _stats(values["candidate_count"]),
        "selected_count_per_scene": _stats(values["selected_count"]),
        "pre_pareto_non_external_count": _stats(values["pre_pareto_non_external_count"]),
        "candidate_bucket_counts": dict(candidate_buckets),
        "selected_bucket_counts": dict(selected_buckets),
        "source_selection_rate": source_selection_rate,
        "candidate_external_share": candidate_external_share,
        "selected_external_share": selected_external_share,
        "external_selection_amplification": selected_external_share / max(candidate_external_share, 1e-12),
        "gt_selected_scene_ratio": counters["gt_selected"] / scenes,
        "external_candidate_scene_ratio": counters["external_candidate_scene"] / scenes,
        "external_selected_scene_ratio": counters["external_selected_scene"] / scenes,
        "all_selected_reward_one_scene_ratio": counters["all_selected_reward_one"] / scenes,
        "zero_reward_spread_scene_ratio": counters["zero_reward_spread"] / scenes,
        "metric_values": {key: _stats(value) for key, value in metric_values.items()},
        "metric_scene_std": {key: _stats(value) for key, value in metric_scene_std.items()},
        "metric_saturation_ratio": {
            key: metric_saturated[key] / max(metric_total[key], 1) for key in sorted(metric_total)
        },
        "gt_relative_ade_m": _stats(values["gt_relative_ade"]),
        "gt_relative_fde_m": _stats(values["gt_relative_fde"]),
        "gt_relative_ade_gt2_ratio": counters["gt_relative_far_gt2"] / max(counters["gt_relative_count"], 1),
        "pairwise_ade_m": _stats(values["pairwise_ade"]),
        "pairwise_fde_m": _stats(values["pairwise_fde"]),
        "pairwise_snsad": _stats(values["pairwise_snsad"]),
        "nearest_snsad_per_scene": _stats(values["nearest_snsad"]),
        "mode_count_snsad_040": _stats(values["mode_count_snsad_040"]),
        "near_duplicate_pair_ratio": counters["near_duplicate_pair_count"] / max(counters["pair_count"], 1),
        "heading_tangent_error_rad": _stats(values["heading_error"]),
        "heading_tangent_error_gt_rad": _stats(values["heading_error_gt"]),
        "heading_tangent_error_non_gt_rad": _stats(values["heading_error_non_gt"]),
        "heading_bad_segment_ratio": _stats(values["heading_bad_segment_ratio"]),
        "selected_bucket_tag_counts": {
            f"{bucket}/{tag}": count for (bucket, tag), count in sorted(selected_source_tag.items())
        },
    }


def _accumulate(
    row: Mapping[str, Any],
    counters: Counter[str],
    candidate_buckets: Counter[str],
    selected_buckets: Counter[str],
    selected_source_tag: Counter[tuple[str, str]],
    values: dict[str, list[float]],
    metric_values: dict[str, list[float]],
    metric_scene_std: dict[str, list[float]],
    metric_saturated: Counter[str],
    metric_total: Counter[str],
) -> None:
    for key in (
        "scene_count",
        "candidate_count",
        "selected_count",
        "gt_selected",
        "external_candidate_scene",
        "external_selected_scene",
        "all_selected_reward_one",
        "zero_reward_spread",
        "gt_relative_far_gt2",
        "gt_relative_count",
        "near_duplicate_pair_count",
        "pair_count",
    ):
        counters[key] += int(row[key])
    candidate_buckets.update(row["candidate_buckets"])
    selected_buckets.update(row["selected_buckets"])
    selected_source_tag.update(row["selected_source_tag"])
    for key in (
        "candidate_count",
        "selected_count",
        "pre_pareto_non_external_count",
        "gt_relative_ade",
        "gt_relative_fde",
        "pairwise_ade",
        "pairwise_fde",
        "pairwise_snsad",
        "heading_error",
        "heading_error_gt",
        "heading_error_non_gt",
        "heading_bad_segment_ratio",
    ):
        value = row[key]
        values[key].extend(value if isinstance(value, list) else [value])
    for key in ("nearest_snsad", "mode_count_snsad_040"):
        values[key].append(float(row[key]))
    for key, metric in row["metric_values"].items():
        metric_values[key].extend(metric)
    for key, value in row["metric_scene_std"].items():
        metric_scene_std[key].append(float(value))
    for key, (saturated, total) in row["metric_saturated"].items():
        metric_saturated[key] += int(saturated)
        metric_total[key] += int(total)


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# SG-FPS Stage2 Teacher-Semantics Audit",
            "",
            f"- archive: `{result['archive']}`",
            f"- records: `{result['record_count']}`",
            f"- candidates: `{result['candidate_count']}`",
            f"- selected: `{result['selected_count']}`",
            f"- candidate external share: `{result['candidate_external_share']:.6f}`",
            f"- selected external share: `{result['selected_external_share']:.6f}`",
            f"- external selection amplification: `{result['external_selection_amplification']:.6f}`",
            f"- GT selected scene ratio: `{result['gt_selected_scene_ratio']:.6f}`",
            f"- all selected reward=1 scene ratio: `{result['all_selected_reward_one_scene_ratio']:.6f}`",
            f"- zero reward spread scene ratio: `{result['zero_reward_spread_scene_ratio']:.6f}`",
            f"- non-GT ADE >2m ratio: `{result['gt_relative_ade_gt2_ratio']:.6f}`",
            f"- near-duplicate pair ratio: `{result['near_duplicate_pair_ratio']:.6f}`",
            "",
            "## Metric Saturation",
            "",
            "```json",
            json.dumps(result["metric_saturation_ratio"], indent=2, sort_keys=True),
            "```",
            "",
            "## Source Selection Rate",
            "",
            "```json",
            json.dumps(result["source_selection_rate"], indent=2, sort_keys=True),
            "```",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-archive-path", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    result = audit_archive(
        Path(args.support_archive_path),
        max_records=max(int(args.max_records), 0),
        workers=max(int(args.workers), 1),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_markdown(result), encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
