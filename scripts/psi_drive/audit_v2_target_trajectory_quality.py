#!/usr/bin/env python3
"""Audit trajectory smoothness/backtracking for PSI-Drive v2 and support targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


DEFAULT_CLEAN_SUPPORT = Path("/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt")
DEFAULT_SUPP_SUPPORT = Path(
    "/mnt/project/VLA-AD/outputs/psi_drive/support_index/"
    "stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt"
)
DEFAULT_V2_PRED = Path(
    "reports/psi_drive/stage3_v2_vs_original_recogdrive_stage3_visual_cases_20260701/"
    "four_conclusion_complex_visual_cases_20260702/online_pool_predictions/v2_step21600/predictions.json"
)
DEFAULT_ORIG_PRED = Path(
    "reports/psi_drive/stage3_v2_vs_original_recogdrive_stage3_visual_cases_20260701/"
    "four_conclusion_complex_visual_cases_20260702/online_pool_predictions/"
    "original_epoch9_step13300/predictions.json"
)
DEFAULT_OUT_DIR = Path("reports/psi_drive/v2_target_trajectory_quality_audit_20260702")


METRIC_COLUMNS = [
    "initial_bend0_deg",
    "initial_bend1_deg",
    "initial_bend_max_deg",
    "max_turn_deg",
    "mean_accel_l2",
    "mean_jerk_l2",
    "tail_reverse_m",
    "tail_min_dx",
    "total_reverse_x_m",
    "final_x_backtrack_m",
    "final_x",
    "path_length",
]


def _as_numpy_traj(trajs: Any) -> np.ndarray:
    if isinstance(trajs, torch.Tensor):
        arr = trajs.detach().cpu().numpy()
    else:
        arr = np.asarray(trajs)
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != (8, 3):
        raise ValueError(f"Expected trajectory array with shape [N, 8, 3], got {arr.shape}.")
    return arr


def _turn_angles(seg: np.ndarray) -> np.ndarray:
    v0 = seg[:, :-1, :]
    v1 = seg[:, 1:, :]
    n0 = np.linalg.norm(v0, axis=-1)
    n1 = np.linalg.norm(v1, axis=-1)
    denom = n0 * n1
    valid = denom > 1e-6
    cos = np.zeros_like(denom)
    cos[valid] = np.sum(v0[valid] * v1[valid], axis=-1) / denom[valid]
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    angles[~valid] = np.nan
    return angles


def trajectory_metrics(trajs: Any) -> Dict[str, np.ndarray]:
    arr = _as_numpy_traj(trajs)
    n = arr.shape[0]
    origin = np.zeros((n, 1, 2), dtype=np.float64)
    xy = arr[:, :, :2]
    xy_ext = np.concatenate([origin, xy], axis=1)
    seg = xy_ext[:, 1:, :] - xy_ext[:, :-1, :]
    seg_dx = seg[:, :, 0]
    seg_l2 = np.linalg.norm(seg, axis=-1)
    angles = _turn_angles(seg)
    accel = seg[:, 1:, :] - seg[:, :-1, :]
    jerk = accel[:, 1:, :] - accel[:, :-1, :]
    max_x = np.max(xy_ext[:, :, 0], axis=1)
    final_x = xy[:, -1, 0]
    tail_dx = seg_dx[:, -3:]
    return {
        "initial_bend0_deg": angles[:, 0],
        "initial_bend1_deg": angles[:, 1],
        "initial_bend_max_deg": np.nanmax(angles[:, :2], axis=1),
        "max_turn_deg": np.nanmax(angles, axis=1),
        "mean_accel_l2": np.linalg.norm(accel, axis=-1).mean(axis=1),
        "mean_jerk_l2": np.linalg.norm(jerk, axis=-1).mean(axis=1),
        "tail_reverse_m": np.maximum(0.0, -tail_dx).sum(axis=1),
        "tail_min_dx": np.min(tail_dx, axis=1),
        "total_reverse_x_m": np.maximum(0.0, -seg_dx).sum(axis=1),
        "final_x_backtrack_m": np.maximum(0.0, max_x - final_x),
        "final_x": final_x,
        "path_length": seg_l2.sum(axis=1),
    }


def _finite_mask(metrics: Dict[str, np.ndarray]) -> np.ndarray:
    mask = np.ones_like(next(iter(metrics.values())), dtype=bool)
    for value in metrics.values():
        mask &= np.isfinite(value)
    return mask


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(np.nanmean(values)) if values.size else float("nan")
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _ratio(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return float("nan")
    if weights is None:
        return float(mask.mean())
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        return float("nan")
    return float(np.sum(mask[valid].astype(np.float64) * weights[valid]) / np.sum(weights[valid]))


def summarize_metrics(metrics: Dict[str, np.ndarray], weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    finite = _finite_mask(metrics)
    out: Dict[str, float] = {"n": int(finite.sum())}
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.where(finite, weights, 0.0)
        out["weight_sum"] = float(np.sum(weights))
    else:
        weights = None
    for key in METRIC_COLUMNS:
        values = metrics[key][finite]
        w = weights[finite] if weights is not None else None
        out[f"{key}_mean"] = _weighted_mean(values, w)
        if weights is None:
            out[f"{key}_p50"] = float(np.nanpercentile(values, 50)) if values.size else float("nan")
            out[f"{key}_p90"] = float(np.nanpercentile(values, 90)) if values.size else float("nan")
            out[f"{key}_p99"] = float(np.nanpercentile(values, 99)) if values.size else float("nan")
    bend = metrics["initial_bend_max_deg"][finite]
    tail_rev = metrics["tail_reverse_m"][finite]
    final_back = metrics["final_x_backtrack_m"][finite]
    total_rev = metrics["total_reverse_x_m"][finite]
    w = weights[finite] if weights is not None else None
    out["initial_bend_gt45_ratio"] = _ratio(bend > 45.0, w)
    out["initial_bend_gt60_ratio"] = _ratio(bend > 60.0, w)
    out["initial_bend_gt90_ratio"] = _ratio(bend > 90.0, w)
    out["tail_reverse_gt005_ratio"] = _ratio(tail_rev > 0.05, w)
    out["tail_reverse_gt020_ratio"] = _ratio(tail_rev > 0.20, w)
    out["final_backtrack_gt005_ratio"] = _ratio(final_back > 0.05, w)
    out["total_reverse_gt005_ratio"] = _ratio(total_rev > 0.05, w)
    return out


def subset_metrics(metrics: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: value[mask] for key, value in metrics.items()}


def flatten_sources(sources: List[List[str]], mask: np.ndarray) -> np.ndarray:
    flat = np.empty(mask.shape, dtype=object)
    for i, row in enumerate(sources):
        for j in range(mask.shape[1]):
            flat[i, j] = row[j] if j < len(row) else ""
    return flat.reshape(-1)


def normalized_flat_weights(weights: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    w = weights.detach().cpu().float()
    m = mask.detach().cpu().bool()
    w = torch.where(m, w.clamp(min=0.0), torch.zeros_like(w))
    sums = w.sum(dim=1, keepdim=True)
    counts = m.sum(dim=1, keepdim=True).clamp(min=1)
    uniform = m.float() / counts.float()
    w = torch.where(sums > 0.0, w / sums.clamp(min=1e-8), uniform)
    return w.reshape(-1).numpy()


def load_support(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def support_flat_metrics(payload: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mask = payload["support_mask"].cpu().bool().reshape(-1).numpy()
    traj = payload["support_trajectories"].reshape(-1, 8, 3)
    metrics = trajectory_metrics(traj)
    weights = normalized_flat_weights(payload["support_weights"], payload["support_mask"])
    return metrics, mask, weights


def source_rows(payload: Dict[str, Any], flat_mask: np.ndarray) -> List[Dict[str, Any]]:
    sources = flatten_sources(payload["support_sources"], payload["support_mask"].cpu().bool().numpy())
    scores = payload["support_scores"].reshape(-1).numpy()
    pdms = payload["support_pdms"].reshape(-1).numpy()
    core = payload["support_core"].reshape(-1).numpy()
    tokens = payload["tokens"]
    rows = []
    for flat_idx in np.flatnonzero(flat_mask):
        row = flat_idx // 3
        slot = flat_idx % 3
        rows.append(
            {
                "token": tokens[row],
                "support_idx": int(slot),
                "source": str(sources[flat_idx]),
                "score": float(scores[flat_idx]),
                "pdms": float(pdms[flat_idx]),
                "core": float(core[flat_idx]),
            }
        )
    return rows


def write_support_worst_csv(
    path: Path,
    payload: Dict[str, Any],
    metrics: Dict[str, np.ndarray],
    flat_mask: np.ndarray,
    limit: int = 200,
) -> None:
    rows = source_rows(payload, flat_mask)
    if not rows:
        return
    metric_subset = {key: value[flat_mask] for key, value in metrics.items()}
    for i, row in enumerate(rows):
        for key in METRIC_COLUMNS:
            row[key] = float(metric_subset[key][i])
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["initial_bend_max_deg"] > 60.0,
            r["tail_reverse_m"] > 0.05,
            r["initial_bend_max_deg"],
            r["tail_reverse_m"],
            r["mean_jerk_l2"],
        ),
        reverse=True,
    )[:limit]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sorted_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted_rows)


def summarize_support(clean: Dict[str, Any], supp: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    clean_metrics, clean_flat_mask, clean_flat_weights = support_flat_metrics(clean)
    supp_metrics, supp_flat_mask, supp_flat_weights = support_flat_metrics(supp)

    clean_mask_2d = clean["support_mask"].cpu().bool()
    supp_mask_2d = supp["support_mask"].cpu().bool()
    traj_delta = (clean["support_trajectories"] - supp["support_trajectories"]).abs().reshape(clean_mask_2d.shape[0], 3, -1)
    traj_changed_2d = traj_delta.amax(dim=-1) > 1e-6
    changed_2d = supp_mask_2d & ((~clean_mask_2d) | traj_changed_2d)
    new_valid_2d = supp_mask_2d & (~clean_mask_2d)
    modified_2d = supp_mask_2d & clean_mask_2d & traj_changed_2d

    supp_sources = flatten_sources(supp["support_sources"], supp_mask_2d.numpy())
    clean_sources = flatten_sources(clean["support_sources"], clean_mask_2d.numpy())
    supp_flat_changed = changed_2d.reshape(-1).numpy()
    supp_flat_new = new_valid_2d.reshape(-1).numpy()
    supp_flat_modified = modified_2d.reshape(-1).numpy()
    supp_flat_gt = (supp_sources == "gt") & supp_flat_mask
    supp_flat_non_gt = (supp_sources != "gt") & supp_flat_mask
    clean_flat_gt = (clean_sources == "gt") & clean_flat_mask
    clean_flat_non_gt = (clean_sources != "gt") & clean_flat_mask

    groups: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]] = {
        "clean_all_valid_unweighted": (clean_metrics, clean_flat_mask, None),
        "clean_all_valid_stage2_weighted": (clean_metrics, clean_flat_mask, clean_flat_weights),
        "clean_gt": (clean_metrics, clean_flat_gt, None),
        "clean_non_gt": (clean_metrics, clean_flat_non_gt, None),
        "supp_all_valid_unweighted": (supp_metrics, supp_flat_mask, None),
        "supp_all_valid_stage2_weighted": (supp_metrics, supp_flat_mask, supp_flat_weights),
        "supp_gt": (supp_metrics, supp_flat_gt, None),
        "supp_non_gt": (supp_metrics, supp_flat_non_gt, None),
        "supp_changed_valid": (supp_metrics, supp_flat_changed, None),
        "supp_changed_valid_stage2_weighted": (supp_metrics, supp_flat_changed, supp_flat_weights),
        "supp_new_valid": (supp_metrics, supp_flat_new, None),
        "supp_modified_valid": (supp_metrics, supp_flat_modified, None),
    }
    summary: Dict[str, Any] = {}
    for name, (metrics, mask, weights) in groups.items():
        summary[name] = summarize_metrics(subset_metrics(metrics, mask), None if weights is None else weights[mask])

    source_summary = {}
    for source, count in Counter(supp_sources[supp_flat_mask]).most_common():
        mask = (supp_sources == source) & supp_flat_mask
        if int(mask.sum()) < 50:
            continue
        source_summary[str(source)] = summarize_metrics(subset_metrics(supp_metrics, mask))
        source_summary[str(source)]["count"] = int(mask.sum())
    summary["supp_by_source"] = source_summary

    counts = {
        "tokens_same": bool(clean["tokens"] == supp["tokens"]),
        "clean_valid_candidates": int(clean_flat_mask.sum()),
        "supp_valid_candidates": int(supp_flat_mask.sum()),
        "supp_changed_valid_candidates": int(supp_flat_changed.sum()),
        "supp_new_valid_candidates": int(supp_flat_new.sum()),
        "supp_modified_valid_candidates": int(supp_flat_modified.sum()),
        "rows_count_increased": int((supp_mask_2d.sum(dim=1) > clean_mask_2d.sum(dim=1)).sum().item()),
        "rows_count_decreased": int((supp_mask_2d.sum(dim=1) < clean_mask_2d.sum(dim=1)).sum().item()),
        "rows_traj_changed": int(traj_changed_2d.any(dim=1).sum().item()),
    }
    summary["counts"] = counts

    write_support_worst_csv(out_dir / "supp_changed_worst_candidates.csv", supp, supp_metrics, supp_flat_changed)
    write_support_worst_csv(out_dir / "supp_all_worst_candidates.csv", supp, supp_metrics, supp_flat_mask)
    return summary


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}.")
    return data


def prediction_metric_rows(name: str, records: List[Dict[str, Any]], use_gt: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for rec in records:
        traj_key = "gt_traj" if use_gt else "pred_traj"
        if not rec.get("valid", False) or traj_key not in rec:
            continue
        metrics = trajectory_metrics(rec[traj_key])
        row = {
            "model": name,
            "token": rec.get("token", ""),
            "log_name": rec.get("log_name", ""),
            "map_name": rec.get("map_name", ""),
            "scene_token": rec.get("scene_token", ""),
        }
        pdm = rec.get("pdm") if isinstance(rec.get("pdm"), dict) else {}
        for key in [
            "score",
            "ego_progress",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ]:
            row[f"pdm_{key}"] = pdm.get(key, "")
        for key in METRIC_COLUMNS:
            row[key] = float(metrics[key][0])
        rows.append(row)
    return rows


def summarize_predictions(v2_path: Path, orig_path: Path, out_dir: Path) -> Dict[str, Any]:
    v2 = load_predictions(v2_path)
    orig = load_predictions(orig_path)
    rows = []
    rows.extend(prediction_metric_rows("v2_step21600_pred", v2))
    rows.extend(prediction_metric_rows("v2_records_gt", v2, use_gt=True))
    rows.extend(prediction_metric_rows("original_epoch9_step13300_pred", orig))

    with (out_dir / "prediction_trajectory_metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "model",
            "token",
            "log_name",
            "map_name",
            "scene_token",
            "pdm_score",
            "pdm_ego_progress",
            "pdm_no_at_fault_collisions",
            "pdm_drivable_area_compliance",
            "pdm_time_to_collision_within_bound",
            "pdm_comfort",
            "pdm_driving_direction_compliance",
            *METRIC_COLUMNS,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary: Dict[str, Any] = {}
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, model_rows in by_model.items():
        metrics = {key: np.asarray([row[key] for row in model_rows], dtype=np.float64) for key in METRIC_COLUMNS}
        summary[model] = summarize_metrics(metrics)
        pdm_scores = [float(row["pdm_score"]) for row in model_rows if row["pdm_score"] != ""]
        if pdm_scores:
            summary[model]["pdm_score_mean"] = float(np.mean(pdm_scores))

    orig_by_token = {row["token"]: row for row in by_model.get("original_epoch9_step13300_pred", [])}
    v2_by_token = {row["token"]: row for row in by_model.get("v2_step21600_pred", [])}
    common_tokens = sorted(set(orig_by_token) & set(v2_by_token))
    diffs = []
    for token in common_tokens:
        v = v2_by_token[token]
        o = orig_by_token[token]
        diffs.append(
            {
                "token": token,
                "v2_initial_bend_max_deg": v["initial_bend_max_deg"],
                "orig_initial_bend_max_deg": o["initial_bend_max_deg"],
                "delta_initial_bend_max_deg": v["initial_bend_max_deg"] - o["initial_bend_max_deg"],
                "v2_tail_reverse_m": v["tail_reverse_m"],
                "orig_tail_reverse_m": o["tail_reverse_m"],
                "delta_tail_reverse_m": v["tail_reverse_m"] - o["tail_reverse_m"],
                "v2_pdm_score": v["pdm_score"],
                "orig_pdm_score": o["pdm_score"],
            }
        )
    if diffs:
        with (out_dir / "prediction_v2_vs_original_diffs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
            writer.writeheader()
            writer.writerows(diffs)
        summary["v2_vs_original_common"] = {
            "n": len(diffs),
            "v2_more_initial_bend_by_10deg": int(sum(d["delta_initial_bend_max_deg"] > 10.0 for d in diffs)),
            "v2_less_initial_bend_by_10deg": int(sum(d["delta_initial_bend_max_deg"] < -10.0 for d in diffs)),
            "v2_more_tail_reverse_by_5cm": int(sum(d["delta_tail_reverse_m"] > 0.05 for d in diffs)),
            "v2_less_tail_reverse_by_5cm": int(sum(d["delta_tail_reverse_m"] < -0.05 for d in diffs)),
        }
    return summary


def fmt_pct(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{100.0 * float(value):.2f}%"


def fmt_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def write_report(path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    support = summary["support"]
    preds = summary["predictions"]
    lines = [
        "# V2 轨迹弯折/后退与补充 target 审计",
        "",
        "## 结论摘要",
        "",
        "- 历史 Core-Pareto GRPO v2 训练不使用补充 support target：实际启动参数使用官方 "
        "`ReCogDrive_Diffusion_Planner_2B_IL.ckpt` 作为 `agent.checkpoint_path` 和 "
        "`agent.reference_policy_checkpoint`，没有 `stage2_target_source=pareto_support`，也没有 "
        "`agent.grpo_support_index_path`；最终配置里 `grpo_use_support_relative` 走默认关闭。",
        "- 补充后的 support index 本身确实包含一部分几何质量较差的候选；这会影响后续 APSD/SR-PGRPO，"
        "但不能解释 2026-06-16 这版 v2 的输出弯折，因为该 v2 训练没有读入这些 target。",
        "- v2 输出中的弯折/末端后退更像是 online GRPO 直接优化 PDMS/Core-Pareto 指标后的副作用："
        "奖励显式关注 EP/TTC/comfort/DDC/NC/DAC，未对首段曲率突变和末端倒退做硬约束。",
        "",
        "## 输入",
        "",
        f"- clean support: `{args.clean_support}`",
        f"- supplemented support: `{args.supp_support}`",
        f"- v2 predictions: `{args.v2_predictions}`",
        f"- original predictions: `{args.original_predictions}`",
        "",
        "## 指标口径",
        "",
        "- 首段弯折：在局部坐标系加入原点，计算 `原点->p1` 与 `p1->p2`、`p1->p2` 与 `p2->p3` 的夹角，取最大值。",
        "- 末段后退：最后 3 段的局部 x 方向负位移累计，`>0.05m` 记作明显后退，`>0.20m` 记作严重后退。",
        "- 平滑性：相邻段二阶差分均值为 accel，三阶差分均值为 jerk；这里是几何诊断，不等价于官方 comfort。",
        "",
        "## Support Index 统计",
        "",
        f"- token 集一致：`{support['counts']['tokens_same']}`",
        f"- clean 有效候选：`{support['counts']['clean_valid_candidates']}`",
        f"- supplemented 有效候选：`{support['counts']['supp_valid_candidates']}`",
        f"- supplemented 新增有效候选：`{support['counts']['supp_new_valid_candidates']}`",
        f"- supplemented 改写有效候选：`{support['counts']['supp_modified_valid_candidates']}`",
        f"- support 数增加的 row：`{support['counts']['rows_count_increased']}`；减少的 row：`{support['counts']['rows_count_decreased']}`",
        "",
        "| 组别 | n | 首段弯折>45 | 首段弯折>60 | 末段后退>5cm | 末段后退>20cm | jerk均值 | final_x回退均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    support_rows = [
        ("clean_all_valid_unweighted", "clean 全候选"),
        ("clean_all_valid_stage2_weighted", "clean 按Stage2采样权重"),
        ("supp_all_valid_unweighted", "supp 全候选"),
        ("supp_all_valid_stage2_weighted", "supp 按Stage2采样权重"),
        ("clean_gt", "clean GT"),
        ("supp_gt", "supp GT"),
        ("clean_non_gt", "clean 非GT"),
        ("supp_non_gt", "supp 非GT"),
        ("supp_changed_valid", "supp 改动候选"),
        ("supp_new_valid", "supp 新增候选"),
        ("supp_modified_valid", "supp 改写候选"),
    ]
    for key, label in support_rows:
        item = support[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(item.get("n", 0)),
                    fmt_pct(item["initial_bend_gt45_ratio"]),
                    fmt_pct(item["initial_bend_gt60_ratio"]),
                    fmt_pct(item["tail_reverse_gt005_ratio"]),
                    fmt_pct(item["tail_reverse_gt020_ratio"]),
                    fmt_num(item["mean_jerk_l2_mean"]),
                    fmt_num(item["final_x_backtrack_m_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Supplemented Source 分解",
            "",
            "| source | n | 首段弯折>45 | 末段后退>5cm | jerk均值 | PDMS选择风险说明 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for source, item in sorted(
        support["supp_by_source"].items(),
        key=lambda kv: kv[1].get("tail_reverse_gt005_ratio", 0.0) + kv[1].get("initial_bend_gt45_ratio", 0.0),
        reverse=True,
    ):
        note = "需过滤" if item["initial_bend_gt45_ratio"] > 0.05 or item["tail_reverse_gt005_ratio"] > 0.05 else "整体可接受"
        lines.append(
            f"| {source} | {item['count']} | {fmt_pct(item['initial_bend_gt45_ratio'])} | "
            f"{fmt_pct(item['tail_reverse_gt005_ratio'])} | {fmt_num(item['mean_jerk_l2_mean'])} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Navtest 已保存输出统计",
            "",
            "这部分只覆盖已保存用于可视化的 81 个复杂候选，不是全 navtest 统计；用于验证现象和 target 审计口径一致。",
            "",
            "| 轨迹来源 | n | PDMS均值 | 首段弯折>45 | 首段弯折>60 | 末段后退>5cm | 末段后退>20cm | jerk均值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pred_order = ["v2_step21600_pred", "original_epoch9_step13300_pred", "v2_records_gt"]
    pred_labels = {
        "v2_step21600_pred": "v2 step21600 输出",
        "original_epoch9_step13300_pred": "原版stage3输出",
        "v2_records_gt": "同81样本GT",
    }
    for key in pred_order:
        if key not in preds:
            continue
        item = preds[key]
        lines.append(
            f"| {pred_labels[key]} | {item['n']} | {fmt_num(item.get('pdm_score_mean', float('nan')))} | "
            f"{fmt_pct(item['initial_bend_gt45_ratio'])} | {fmt_pct(item['initial_bend_gt60_ratio'])} | "
            f"{fmt_pct(item['tail_reverse_gt005_ratio'])} | {fmt_pct(item['tail_reverse_gt020_ratio'])} | "
            f"{fmt_num(item['mean_jerk_l2_mean'])} |"
        )
    if "v2_vs_original_common" in preds:
        common = preds["v2_vs_original_common"]
        lines.extend(
            [
                "",
                "同 token 对比：",
                "",
                f"- common tokens: `{common['n']}`",
                f"- v2 首段弯折比原版大 10 度以上：`{common['v2_more_initial_bend_by_10deg']}`",
                f"- v2 首段弯折比原版小 10 度以上：`{common['v2_less_initial_bend_by_10deg']}`",
                f"- v2 末段后退比原版多 5cm 以上：`{common['v2_more_tail_reverse_by_5cm']}`",
                f"- v2 末段后退比原版少 5cm 以上：`{common['v2_less_tail_reverse_by_5cm']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 文件输出",
            "",
            "- `support_summary.json`: 全量统计。",
            "- `supp_changed_worst_candidates.csv`: 补充/改动候选里弯折或后退最明显的样本。",
            "- `supp_all_worst_candidates.csv`: supplemented 全候选里弯折或后退最明显的样本。",
            "- `prediction_trajectory_metrics.csv`: 已保存 v2/original/GT 输出逐 token 几何指标。",
            "- `prediction_v2_vs_original_diffs.csv`: v2 与原版同 token 几何差异。",
            "",
            "## 后续建议",
            "",
            "1. v2/SR-PGRPO 奖励中加入 hard/soft 几何约束：首两段夹角阈值、末三段 x 负位移阈值、final_x_backtrack 阈值。",
            "2. 重新构建 support index 时，先过滤 `tail_reverse_m>0.05`、`initial_bend_max_deg>60`、过大 jerk 的非 GT 候选；"
            "GT 可以保留但降低坏几何样本被采样概率。",
            "3. 对 best-by-PDMS 的候选选择加 secondary sort：PDMS/Core 相近时优先更小 jerk、更小首段弯折、无末端后退。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-support", type=Path, default=DEFAULT_CLEAN_SUPPORT)
    parser.add_argument("--supp-support", type=Path, default=DEFAULT_SUPP_SUPPORT)
    parser.add_argument("--v2-predictions", type=Path, default=DEFAULT_V2_PRED)
    parser.add_argument("--original-predictions", type=Path, default=DEFAULT_ORIG_PRED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean = load_support(args.clean_support)
    supp = load_support(args.supp_support)
    summary = {
        "inputs": {
            "clean_support": str(args.clean_support),
            "supp_support": str(args.supp_support),
            "v2_predictions": str(args.v2_predictions),
            "original_predictions": str(args.original_predictions),
        },
        "support": summarize_support(clean, supp, args.out_dir),
        "predictions": summarize_predictions(args.v2_predictions, args.original_predictions, args.out_dir),
    }
    (args.out_dir / "support_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    write_report(args.out_dir / "README.md", summary, args)
    print(json.dumps(summary["support"]["counts"], ensure_ascii=False, indent=2))
    if "v2_step21600_pred" in summary["predictions"]:
        print(json.dumps(summary["predictions"]["v2_step21600_pred"], ensure_ascii=False, indent=2, allow_nan=True)[:2000])
    print(args.out_dir / "README.md")


if __name__ == "__main__":
    main()
