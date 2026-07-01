#!/usr/bin/env python3
"""Analyze whether low/zero navtest scenes are under-covered by navtrain.

This script is read-only with respect to model outputs. It extracts lightweight
scene descriptors from existing metric caches, joins navtest PDMS rows, and
compares train-nearest-neighbor coverage for score buckets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.planning.metric_caching.fast_metric_cache_loader import load_metric_cache_auto


FEATURE_NAMES: Tuple[str, ...] = (
    "ego_speed",
    "ego_accel",
    "route_lane_count",
    "centerline_length",
    "drivable_token_count",
    "object_count",
    "vehicle_count",
    "pedestrian_count",
    "generic_object_count",
    "traffic_cone_count",
    "object_min_dist",
    "vehicle_min_dist",
    "pedestrian_min_dist",
    "object_count_10m",
    "object_count_20m",
    "object_count_30m",
    "vehicle_count_20m",
    "pedestrian_count_20m",
    "collided_track_count",
    "gt_final_x_4s",
    "gt_final_y_4s",
    "gt_abs_final_y_4s",
    "gt_final_heading_4s",
    "gt_path_len_4s",
    "gt_displacement_4s",
    "gt_progress_ratio_4s",
    "gt_max_abs_y_4s",
    "gt_mean_abs_y_4s",
    "gt_max_abs_heading_4s",
    "gt_turn_amount_4s",
    "gt_curvature_proxy_4s",
    "gt_reverse_frac_4s",
    "gt_final_x_5s",
    "gt_final_y_5s",
    "gt_abs_final_y_5s",
    "gt_final_heading_5s",
    "gt_path_len_5s",
    "gt_displacement_5s",
    "gt_progress_ratio_5s",
    "gt_max_abs_y_5s",
    "gt_mean_abs_y_5s",
    "gt_max_abs_heading_5s",
    "gt_turn_amount_5s",
    "gt_curvature_proxy_5s",
    "gt_reverse_frac_5s",
)


def _wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _token_log_from_path(path: Path, cache_root: Path) -> Tuple[str, str]:
    token = path.parent.name
    try:
        rel = path.relative_to(cache_root)
        log_name = rel.parts[0] if rel.parts else ""
    except ValueError:
        log_name = path.parents[2].name if len(path.parents) >= 3 else ""
    return token, log_name


def _list_metric_cache_paths(cache_root: Path) -> List[Path]:
    return sorted(Path(cache_root).rglob("metric_cache.pkl"))


def _agent_type_name(obj: Any) -> str:
    try:
        return str(obj.tracked_object_type).upper()
    except Exception:
        return ""


def _object_center_xy(obj: Any) -> Tuple[float, float] | None:
    try:
        center = obj.center
        return float(center.x), float(center.y)
    except Exception:
        try:
            center = obj.box.center
            return float(center.x), float(center.y)
        except Exception:
            return None


def _ego_speed_accel(metric_cache: Any) -> Tuple[float, float]:
    dyn = metric_cache.ego_state.dynamic_car_state
    speed = 0.0
    accel = 0.0
    for name, sink in (("rear_axle_velocity_2d", "speed"), ("rear_axle_acceleration_2d", "accel")):
        try:
            vec = getattr(dyn, name)
            val = math.hypot(float(vec.x), float(vec.y))
        except Exception:
            val = 0.0
        if sink == "speed":
            speed = val
        else:
            accel = val
    return speed, accel


def _relative_trajectory(metric_cache: Any) -> np.ndarray:
    origin = metric_cache.ego_state.rear_axle
    ox, oy, oh = float(origin.x), float(origin.y), float(origin.heading)
    c, s = math.cos(-oh), math.sin(-oh)
    states = metric_cache.trajectory.get_sampled_trajectory()
    rows: List[Tuple[float, float, float]] = []
    for state in states:
        pose = state.rear_axle
        dx, dy = float(pose.x) - ox, float(pose.y) - oy
        x = c * dx - s * dy
        y = s * dx + c * dy
        h = _safe_float(_wrap_angle(float(pose.heading) - oh))
        rows.append((x, y, h))
    if not rows:
        return np.zeros((1, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _trajectory_features(rel: np.ndarray, horizon_s: float) -> Dict[str, float]:
    # Metric cache trajectories are sampled at 10Hz in current data.
    idx = min(len(rel) - 1, int(round(horizon_s / 0.1)))
    sub = rel[: idx + 1]
    xy = sub[:, :2].astype(np.float64)
    heading = np.unwrap(sub[:, 2].astype(np.float64))
    deltas = np.diff(xy, axis=0)
    seg = np.linalg.norm(deltas, axis=1)
    path_len = float(seg.sum())
    final = xy[-1]
    displacement = float(np.linalg.norm(final - xy[0]))
    heading_delta = float(_wrap_angle(heading[-1] - heading[0]))
    heading_steps = np.diff(heading)
    turn_amount = float(np.abs(heading_steps).sum())
    progress_ratio = float(final[0] / max(path_len, 1e-6))
    reverse_frac = float((np.diff(xy[:, 0]) < -1e-3).mean()) if len(xy) > 1 else 0.0
    suffix = "4s" if horizon_s <= 4.0 else "5s"
    return {
        f"gt_final_x_{suffix}": float(final[0]),
        f"gt_final_y_{suffix}": float(final[1]),
        f"gt_abs_final_y_{suffix}": float(abs(final[1])),
        f"gt_final_heading_{suffix}": heading_delta,
        f"gt_path_len_{suffix}": path_len,
        f"gt_displacement_{suffix}": displacement,
        f"gt_progress_ratio_{suffix}": progress_ratio,
        f"gt_max_abs_y_{suffix}": float(np.max(np.abs(xy[:, 1]))),
        f"gt_mean_abs_y_{suffix}": float(np.mean(np.abs(xy[:, 1]))),
        f"gt_max_abs_heading_{suffix}": float(np.max(np.abs(_wrap_angle(heading - heading[0])))),
        f"gt_turn_amount_{suffix}": turn_amount,
        f"gt_curvature_proxy_{suffix}": float(turn_amount / max(path_len, 1e-6)),
        f"gt_reverse_frac_{suffix}": reverse_frac,
    }


def _extract_one(args: Tuple[str, str, str]) -> Dict[str, Any]:
    split, path_text, cache_root_text = args
    path = Path(path_text)
    cache_root = Path(cache_root_text)
    token, log_name = _token_log_from_path(path, cache_root)
    out: Dict[str, Any] = {
        "split": split,
        "token": token,
        "log_name": log_name,
        "path": str(path),
        "ok": True,
        "error": "",
    }
    try:
        mc = load_metric_cache_auto(path)
        ego_speed, ego_accel = _ego_speed_accel(mc)
        out["ego_speed"] = ego_speed
        out["ego_accel"] = ego_accel
        out["route_lane_count"] = float(len(getattr(mc, "route_lane_ids", []) or []))
        out["centerline_length"] = _safe_float(getattr(mc.centerline, "length", 0.0))
        out["drivable_token_count"] = float(len(getattr(mc.drivable_area_map, "tokens", []) or []))
        objects = getattr(mc.observation, "unique_objects", {}) or {}
        ego_center = mc.ego_state.center
        ex, ey = float(ego_center.x), float(ego_center.y)
        distances: List[float] = []
        vehicle_distances: List[float] = []
        pedestrian_distances: List[float] = []
        counts = {
            "vehicle_count": 0.0,
            "pedestrian_count": 0.0,
            "generic_object_count": 0.0,
            "traffic_cone_count": 0.0,
        }
        for obj in objects.values():
            xy = _object_center_xy(obj)
            if xy is None:
                continue
            dist = math.hypot(xy[0] - ex, xy[1] - ey)
            distances.append(dist)
            type_name = _agent_type_name(obj)
            if "VEHICLE" in type_name:
                counts["vehicle_count"] += 1.0
                vehicle_distances.append(dist)
            elif "PEDESTRIAN" in type_name:
                counts["pedestrian_count"] += 1.0
                pedestrian_distances.append(dist)
            elif "TRAFFIC_CONE" in type_name:
                counts["traffic_cone_count"] += 1.0
            elif "GENERIC_OBJECT" in type_name:
                counts["generic_object_count"] += 1.0
        out["object_count"] = float(len(distances))
        out.update(counts)
        out["object_min_dist"] = float(min(distances)) if distances else 999.0
        out["vehicle_min_dist"] = float(min(vehicle_distances)) if vehicle_distances else 999.0
        out["pedestrian_min_dist"] = float(min(pedestrian_distances)) if pedestrian_distances else 999.0
        out["object_count_10m"] = float(sum(d <= 10.0 for d in distances))
        out["object_count_20m"] = float(sum(d <= 20.0 for d in distances))
        out["object_count_30m"] = float(sum(d <= 30.0 for d in distances))
        out["vehicle_count_20m"] = float(sum(d <= 20.0 for d in vehicle_distances))
        out["pedestrian_count_20m"] = float(sum(d <= 20.0 for d in pedestrian_distances))
        out["collided_track_count"] = float(len(getattr(mc.observation, "collided_track_ids", []) or []))
        rel = _relative_trajectory(mc)
        out.update(_trajectory_features(rel, 4.0))
        out.update(_trajectory_features(rel, 5.0))
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["error"] = repr(exc)
        for name in FEATURE_NAMES:
            out.setdefault(name, np.nan)
    return out


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "token", "log_name", "path", "ok", "error", *FEATURE_NAMES]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_feature_csv(
    *,
    train_cache_dir: Path,
    navtest_cache_dir: Path,
    output_csv: Path,
    workers: int,
    limit_train: int | None,
    limit_navtest: int | None,
) -> pd.DataFrame:
    if output_csv.is_file():
        return pd.read_csv(output_csv)
    train_paths = _list_metric_cache_paths(train_cache_dir)
    navtest_paths = _list_metric_cache_paths(navtest_cache_dir)
    if limit_train:
        train_paths = train_paths[:limit_train]
    if limit_navtest:
        navtest_paths = navtest_paths[:limit_navtest]
    tasks = [
        *[("navtrain", str(path), str(train_cache_dir)) for path in train_paths],
        *[("navtest", str(path), str(navtest_cache_dir)) for path in navtest_paths],
    ]
    rows: List[Dict[str, Any]] = []
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_extract_one, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 5000 == 0 or idx == total:
                print(f"extracted {idx}/{total}", flush=True)
    rows.sort(key=lambda r: (str(r.get("split")), str(r.get("token"))))
    _write_rows_csv(output_csv, rows)
    return pd.DataFrame(rows)


def _read_eval_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "token" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Expected token and score columns in {path}")
    numeric_cols = [
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _standardize(train_x: np.ndarray, other_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train_z = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    other_z = np.nan_to_num((other_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return train_z, other_z, mean, std


def _bucket_name(score: float) -> str:
    if score == 0.0:
        return "zero"
    if score <= 0.5:
        return "low_0_0p5"
    if score <= 0.85:
        return "mid_0p5_0p85"
    if score <= 0.95:
        return "good_0p85_0p95"
    return "high_0p95_1"


def _summarize_group(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"count": int(len(df))}
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        out[f"{col}_mean"] = float(values.mean())
        out[f"{col}_median"] = float(values.median())
    return out


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        value = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{nd}f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_csv = output_dir / "metric_cache_scene_features.csv"
    features_df = build_feature_csv(
        train_cache_dir=Path(args.train_metric_cache_dir),
        navtest_cache_dir=Path(args.navtest_metric_cache_dir),
        output_csv=feature_csv,
        workers=int(args.workers),
        limit_train=args.limit_train,
        limit_navtest=args.limit_navtest,
    )
    features_df = features_df[features_df["ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    train = features_df[features_df["split"] == "navtrain"].copy()
    navtest_features = features_df[features_df["split"] == "navtest"].copy()
    eval_df = _read_eval_csv(Path(args.navtest_eval_csv))
    nav = navtest_features.merge(eval_df, on="token", how="inner", suffixes=("", "_eval"))
    if train.empty or nav.empty:
        raise RuntimeError(f"Insufficient rows after join: train={len(train)}, nav={len(nav)}")

    train_x = train.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    nav_x = nav.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64)
    train_z, nav_z, _, _ = _standardize(train_x, nav_x)
    tree = cKDTree(train_z)
    k = min(int(args.knn_k), len(train))
    dists, indices = tree.query(nav_z, k=k, workers=max(1, int(args.query_workers)))
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]
    nav["nn1_dist"] = dists[:, 0]
    nav["nn5_dist"] = dists[:, min(4, k - 1)]
    nav["nn10_dist"] = dists[:, min(9, k - 1)]
    nav["nn50_dist"] = dists[:, min(49, k - 1)]
    nav["nn10_mean_dist"] = dists[:, : min(10, k)].mean(axis=1)
    nav["nn50_mean_dist"] = dists[:, : min(50, k)].mean(axis=1)
    nav["nearest_train_token"] = train.iloc[indices[:, 0]]["token"].to_numpy()
    nav["nearest_train_log_name"] = train.iloc[indices[:, 0]]["log_name"].to_numpy()

    score = pd.to_numeric(nav["score"], errors="coerce").fillna(0.0)
    nav["score_bucket"] = score.map(_bucket_name)
    nav["pdms_core"] = (
        5 * pd.to_numeric(nav["ego_progress_eval"], errors="coerce")
        + 5 * pd.to_numeric(nav["time_to_collision_within_bound"], errors="coerce")
        + 2 * pd.to_numeric(nav["comfort"], errors="coerce")
    ) / 12.0

    # Feature novelty: per-feature absolute z-score and top contributors.
    abs_z = np.abs(nav_z)
    nav["mean_abs_feature_z"] = abs_z.mean(axis=1)
    nav["max_abs_feature_z"] = abs_z.max(axis=1)
    top_feature_idx = abs_z.argmax(axis=1)
    nav["top_novel_feature"] = [FEATURE_NAMES[int(i)] for i in top_feature_idx]

    nav_out = output_dir / "navtest_score_with_train_knn.csv"
    nav.to_csv(nav_out, index=False)

    group_cols = [
        "score",
        "pdms_core",
        "nn1_dist",
        "nn10_mean_dist",
        "nn50_mean_dist",
        "mean_abs_feature_z",
        "max_abs_feature_z",
        "gt_path_len_4s",
        "gt_final_x_4s",
        "gt_abs_final_y_4s",
        "gt_turn_amount_4s",
        "object_count_20m",
        "vehicle_count_20m",
        "pedestrian_count_20m",
    ]
    group_summary = []
    for bucket, group in nav.groupby("score_bucket", sort=False):
        row = {"bucket": bucket, **_summarize_group(group, group_cols)}
        group_summary.append(row)
    group_summary_df = pd.DataFrame(group_summary).sort_values("score_mean")
    group_summary_path = output_dir / "score_bucket_train_knn_summary.csv"
    group_summary_df.to_csv(group_summary_path, index=False)

    zero = nav[nav["score"] == 0.0].copy()
    nonzero = nav[nav["score"] > 0.0].copy()
    high = nav[nav["score"] > 0.95].copy()
    low_nonzero = nav[(nav["score"] > 0.0) & (nav["score"] <= 0.5)].copy()

    feature_delta_rows = []
    compare_features = [
        "nn1_dist",
        "nn10_mean_dist",
        "nn50_mean_dist",
        "mean_abs_feature_z",
        "gt_path_len_4s",
        "gt_final_x_4s",
        "gt_abs_final_y_4s",
        "gt_turn_amount_4s",
        "gt_curvature_proxy_4s",
        "object_count_20m",
        "vehicle_count_20m",
        "pedestrian_count_20m",
        "ego_speed",
        "route_lane_count",
        "centerline_length",
    ]
    for name in compare_features:
        z_mean = float(pd.to_numeric(zero[name], errors="coerce").mean()) if len(zero) else float("nan")
        h_mean = float(pd.to_numeric(high[name], errors="coerce").mean()) if len(high) else float("nan")
        nz_mean = float(pd.to_numeric(nonzero[name], errors="coerce").mean()) if len(nonzero) else float("nan")
        feature_delta_rows.append(
            {
                "feature": name,
                "zero_mean": z_mean,
                "nonzero_mean": nz_mean,
                "high_mean": h_mean,
                "zero_minus_high": z_mean - h_mean,
                "zero_minus_nonzero": z_mean - nz_mean,
            }
        )
    feature_delta_df = pd.DataFrame(feature_delta_rows).sort_values("zero_minus_high", key=lambda s: s.abs(), ascending=False)
    feature_delta_path = output_dir / "zero_vs_high_feature_deltas.csv"
    feature_delta_df.to_csv(feature_delta_path, index=False)

    zero_top_logs = zero.groupby("log_name").agg(count=("token", "count"), mean_score=("score", "mean")).reset_index()
    zero_top_logs = zero_top_logs.sort_values("count", ascending=False)
    zero_top_logs_path = output_dir / "zero_score_top_logs.csv"
    zero_top_logs.to_csv(zero_top_logs_path, index=False)

    nearest_examples = zero.sort_values(["nn10_mean_dist", "score"], ascending=[False, True]).head(int(args.example_rows))
    nearest_examples_path = output_dir / "zero_score_most_train_novel_examples.csv"
    nearest_examples[
        [
            "token",
            "log_name",
            "score",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress_eval",
            "time_to_collision_within_bound",
            "driving_direction_compliance",
            "nn1_dist",
            "nn10_mean_dist",
            "nearest_train_token",
            "nearest_train_log_name",
            "top_novel_feature",
            "gt_path_len_4s",
            "gt_final_x_4s",
            "gt_abs_final_y_4s",
            "gt_turn_amount_4s",
            "object_count_20m",
        ]
    ].to_csv(nearest_examples_path, index=False)

    summary = {
        "feature_csv": str(feature_csv),
        "navtest_knn_csv": str(nav_out),
        "score_bucket_summary_csv": str(group_summary_path),
        "zero_vs_high_feature_deltas_csv": str(feature_delta_path),
        "zero_score_top_logs_csv": str(zero_top_logs_path),
        "zero_score_most_train_novel_examples_csv": str(nearest_examples_path),
        "num_train": int(len(train)),
        "num_navtest_joined": int(len(nav)),
        "num_zero": int(len(zero)),
        "num_nonzero": int(len(nonzero)),
        "num_low_nonzero_le_0p5": int(len(low_nonzero)),
        "num_high_gt_0p95": int(len(high)),
        "zero_ratio": float(len(zero) / max(len(nav), 1)),
        "zero_nn10_mean_dist": float(zero["nn10_mean_dist"].mean()) if len(zero) else None,
        "nonzero_nn10_mean_dist": float(nonzero["nn10_mean_dist"].mean()) if len(nonzero) else None,
        "high_nn10_mean_dist": float(high["nn10_mean_dist"].mean()) if len(high) else None,
        "zero_mean_abs_feature_z": float(zero["mean_abs_feature_z"].mean()) if len(zero) else None,
        "high_mean_abs_feature_z": float(high["mean_abs_feature_z"].mean()) if len(high) else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_rows = []
    for _, row in group_summary_df.iterrows():
        md_rows.append(
            [
                row["bucket"],
                int(row["count"]),
                _fmt(row["score_mean"], 6),
                _fmt(row["pdms_core_mean"], 4),
                _fmt(row["nn1_dist_mean"], 3),
                _fmt(row["nn10_mean_dist_mean"], 3),
                _fmt(row["nn50_mean_dist_mean"], 3),
                _fmt(row["mean_abs_feature_z_mean"], 3),
                _fmt(row["gt_path_len_4s_mean"], 2),
                _fmt(row["gt_abs_final_y_4s_mean"], 2),
                _fmt(row["object_count_20m_mean"], 2),
            ]
        )
    delta_rows = []
    for _, row in feature_delta_df.head(12).iterrows():
        delta_rows.append(
            [
                row["feature"],
                _fmt(row["zero_mean"], 3),
                _fmt(row["high_mean"], 3),
                _fmt(row["zero_minus_high"], 3),
                _fmt(row["zero_minus_nonzero"], 3),
            ]
        )
    example_rows = []
    for _, row in nearest_examples.head(12).iterrows():
        example_rows.append(
            [
                row["token"],
                row["log_name"],
                _fmt(row["nn10_mean_dist"], 3),
                row["nearest_train_token"],
                row["top_novel_feature"],
                _fmt(row["gt_path_len_4s"], 2),
                _fmt(row["object_count_20m"], 1),
            ]
        )
    md = [
        "# Navtest Zero-Score Train Similarity Analysis",
        "",
        f"- Train metric cache: `{args.train_metric_cache_dir}`",
        f"- Navtest metric cache: `{args.navtest_metric_cache_dir}`",
        f"- Navtest eval CSV: `{args.navtest_eval_csv}`",
        f"- Train scenes analyzed: {len(train)}",
        f"- Navtest scenes joined: {len(nav)}",
        f"- Zero-score navtest scenes: {len(zero)} ({len(zero) / max(len(nav), 1):.4%})",
        "",
        "## Score Buckets",
        "",
        _md_table(
            [
                "bucket",
                "count",
                "score_mean",
                "core_mean",
                "nn1",
                "nn10",
                "nn50",
                "mean_abs_z",
                "gt_path_4s",
                "abs_y_4s",
                "obj20m",
            ],
            md_rows,
        ),
        "",
        "## Largest Zero-vs-High Feature Gaps",
        "",
        _md_table(["feature", "zero_mean", "high_mean", "zero-high", "zero-nonzero"], delta_rows),
        "",
        "## Most Train-Novel Zero-Score Examples",
        "",
        _md_table(["token", "log_name", "nn10", "nearest_train", "top_feature", "gt_path_4s", "obj20m"], example_rows),
        "",
        "## Interpretation Notes",
        "",
        "- `nn1/nn10/nn50` are standardized feature-space distances to navtrain nearest neighbors. Larger means less covered by similar train metric-cache scenes under these descriptors.",
        "- Features include GT future shape, ego speed, nearby object counts, route/centerline size, and map-cache complexity. They do not use navtest rewards for training.",
        "- This analysis can identify data coverage or scene-composition gaps, but it cannot by itself prove the trained policy failed on similar train scenes. For that, run SOTA eval on the nearest train tokens listed in the CSV.",
    ]
    md_path = output_dir / "navtest_zero_score_train_similarity.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(md_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-metric-cache-dir", default="/mnt/project/VLA-AD/cache/metric_cache_train_full")
    parser.add_argument("--navtest-metric-cache-dir", default="/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1")
    parser.add_argument(
        "--navtest-eval-csv",
        default="/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_4to7_sharded_conc4_watch_from5700_v1/eval_step-step_21600/local_sharded_aggregated.csv",
    )
    parser.add_argument("--output-dir", default="reports/recogdrive_stage3/data_similarity/core_pareto_step21600")
    parser.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 8) // 2)))
    parser.add_argument("--query-workers", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--example-rows", type=int, default=50)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-navtest", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
