#!/usr/bin/env python3
"""Compare original ReCogDrive Stage3 RL NAVTEST metrics with Stage2 IL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


METRICS = ["PDMS", "NC", "DAC", "EP", "TTC", "Comfort", "DDC", "SafetyMean"]
SAFETY_METRICS = ["NC", "DAC", "TTC", "Comfort", "DDC"]
STAGE2_COLUMN_MAP = {
    "score": "PDMS",
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "ego_progress": "EP",
    "time_to_collision_within_bound": "TTC",
    "comfort": "Comfort",
    "driving_direction_compliance": "DDC",
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage3-source-csv",
        type=Path,
        default=Path(
            "reports/psi_drive/"
            "stage3_v2_vs_original_recogdrive_stage3_visual_cases_20260701/"
            "v2_vs_original_epoch9_scene_delta.csv"
        ),
        help="CSV containing orig_* metrics from the earlier V2-vs-original-stage3 alignment.",
    )
    parser.add_argument(
        "--stage2-pdm-csv",
        type=Path,
        default=Path(
            "/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/"
            "a0_official_aligned_a0complete/eval/step_00100000/pdm_results.csv"
        ),
        help="Stage2 IL pdm_results.csv.",
    )
    parser.add_argument(
        "--stage2-metrics-json",
        type=Path,
        default=Path(
            "/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/"
            "a0_official_aligned_a0complete/eval/step_00100000/metrics.json"
        ),
        help="Stage2 IL metrics.json for provenance.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/psi_drive/original_stage3_rl_vs_stage2_il_navtest_20260703"),
        help="Output directory.",
    )
    return parser.parse_args()


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def join_list(values: Iterable[str]) -> str:
    return "+".join([str(v) for v in values if str(v)])


def read_stage3(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["token"].astype(str) != "average"].copy()
    needed = ["token", "log_name", "scene_token", "chunk"] + [f"orig_{m}" for m in METRICS if m != "SafetyMean"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing original Stage3 columns: {missing}")
    if "orig_SafetyMean" not in df.columns:
        df["orig_SafetyMean"] = df[[f"orig_{m}" for m in SAFETY_METRICS]].mean(axis=1)
    out = df[["token", "log_name", "scene_token", "chunk"]].copy()
    for metric in METRICS:
        out[f"orig_stage3_{metric}"] = pd.to_numeric(df[f"orig_{metric}"], errors="coerce")
    return out


def read_stage2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)].copy()
    df = df[df["sample_token"].astype(str) != "average"].copy()
    missing = [col for col in STAGE2_COLUMN_MAP if col not in df.columns]
    if missing:
        raise ValueError(f"Missing Stage2 columns: {missing}")
    out = df[["sample_token"]].rename(columns={"sample_token": "token"}).copy()
    for src, metric in STAGE2_COLUMN_MAP.items():
        out[f"stage2_il_{metric}"] = pd.to_numeric(df[src], errors="coerce")
    out["stage2_il_SafetyMean"] = out[[f"stage2_il_{m}" for m in SAFETY_METRICS]].mean(axis=1)
    return out


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for metric in METRICS:
        cand = df[f"orig_stage3_{metric}"]
        base = df[f"stage2_il_{metric}"]
        delta = df[f"delta_{metric}"]
        gt = delta > EPS
        lt = delta < -EPS
        eq = ~(gt | lt)
        base_zero = base.abs() <= EPS
        cand_zero = cand.abs() <= EPS
        base_pos = base > EPS
        cand_pos = cand > EPS
        rows.append(
            {
                "metric": metric,
                "total": total,
                "orig_stage3_mean": float(cand.mean()),
                "stage2_il_mean": float(base.mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "orig_stage3_gt_stage2_il_count": int(gt.sum()),
                "orig_stage3_eq_stage2_il_count": int(eq.sum()),
                "orig_stage3_lt_stage2_il_count": int(lt.sum()),
                "stage2_il_zero_count": int(base_zero.sum()),
                "zero_to_positive_count": int((base_zero & cand_pos).sum()),
                "zero_still_zero_count": int((base_zero & cand_zero).sum()),
                "positive_to_zero_count": int((base_pos & cand_zero).sum()),
                "positive_to_positive_improved_count": int((base_pos & cand_pos & gt).sum()),
                "positive_to_positive_declined_count": int((base_pos & cand_pos & lt).sum()),
                "zero_net_repair_count": int((base_zero & cand_pos).sum() - (base_pos & cand_zero).sum()),
                "delta_gt_0p01_count": int((delta > 0.01 + EPS).sum()),
                "delta_gt_0p05_count": int((delta > 0.05 + EPS).sum()),
                "delta_gt_0p10_count": int((delta > 0.10 + EPS).sum()),
                "delta_lt_minus_0p01_count": int((delta < -0.01 - EPS).sum()),
                "delta_lt_minus_0p05_count": int((delta < -0.05 - EPS).sum()),
                "delta_lt_minus_0p10_count": int((delta < -0.10 - EPS).sum()),
                "orig_stage3_gt_stage2_il_pct": pct(int(gt.sum()), total),
                "orig_stage3_eq_stage2_il_pct": pct(int(eq.sum()), total),
                "orig_stage3_lt_stage2_il_pct": pct(int(lt.sum()), total),
                "stage2_il_zero_pct": pct(int(base_zero.sum()), total),
                "zero_to_positive_pct": pct(int((base_zero & cand_pos).sum()), total),
                "positive_to_zero_pct": pct(int((base_pos & cand_zero).sum()), total),
                "delta_gt_0p05_pct": pct(int((delta > 0.05 + EPS).sum()), total),
                "delta_lt_minus_0p05_pct": pct(int((delta < -0.05 - EPS).sum()), total),
            }
        )
    return pd.DataFrame(rows)


def zero_transition_tokens(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        base = df[f"stage2_il_{metric}"]
        cand = df[f"orig_stage3_{metric}"]
        if direction == "zero_to_positive":
            mask = (base.abs() <= EPS) & (cand > EPS)
        elif direction == "positive_to_zero":
            mask = (base > EPS) & (cand.abs() <= EPS)
        else:
            raise ValueError(direction)
        sub = df.loc[mask].copy()
        sub.insert(0, "metric", metric)
        sub["stage2_il_value"] = sub[f"stage2_il_{metric}"]
        sub["orig_stage3_value"] = sub[f"orig_stage3_{metric}"]
        sub["delta_value"] = sub[f"delta_{metric}"]
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def safety_preserved_ep_summary(df: pd.DataFrame) -> pd.DataFrame:
    conditions = {
        "safety_items_not_decreased": pd.Series(True, index=df.index),
        "SafetyMean_not_decreased": df["delta_SafetyMean"] >= -EPS,
        "orig_stage3_all_safety_full_score": pd.Series(True, index=df.index),
        "both_all_safety_full_score": pd.Series(True, index=df.index),
        "safety_items_not_decreased_and_orig_stage3_all_safety_full_score": pd.Series(True, index=df.index),
    }
    for metric in SAFETY_METRICS:
        conditions["safety_items_not_decreased"] &= df[f"delta_{metric}"] >= -EPS
        conditions["orig_stage3_all_safety_full_score"] &= (df[f"orig_stage3_{metric}"] - 1.0).abs() <= EPS
        conditions["both_all_safety_full_score"] &= (
            ((df[f"orig_stage3_{metric}"] - 1.0).abs() <= EPS)
            & ((df[f"stage2_il_{metric}"] - 1.0).abs() <= EPS)
        )
    conditions["safety_items_not_decreased_and_orig_stage3_all_safety_full_score"] = (
        conditions["safety_items_not_decreased"] & conditions["orig_stage3_all_safety_full_score"]
    )

    rows = []
    for condition, cond_mask in conditions.items():
        for threshold in [0.0, 0.01, 0.05, 0.10]:
            mask = cond_mask & (df["delta_EP"] > threshold + EPS)
            sub = df.loc[mask]
            rows.append(
                {
                    "safety_condition": condition,
                    "ep_delta_threshold": threshold,
                    "count": int(mask.sum()),
                    "pct_total": pct(int(mask.sum()), len(df)),
                    "mean_ep_delta_on_selected": float(sub["delta_EP"].mean()) if len(sub) else None,
                    "mean_pdms_delta_on_selected": float(sub["delta_PDMS"].mean()) if len(sub) else None,
                    "mean_safetymean_delta_on_selected": float(sub["delta_SafetyMean"].mean()) if len(sub) else None,
                }
            )
    return pd.DataFrame(rows)


def ep_preserved_safety_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ep_not_down = df["delta_EP"] >= -EPS
    ep_strict_up = df["delta_EP"] > EPS
    safety_not_down = pd.Series(True, index=df.index)
    any_safety_up = pd.Series(False, index=df.index)
    any_safety_zero_repair = pd.Series(False, index=df.index)
    improved_items = []
    zero_repair_items = []

    for _, row in df.iterrows():
        improved = [m for m in SAFETY_METRICS if row[f"delta_{m}"] > EPS]
        repairs = [m for m in SAFETY_METRICS if abs(row[f"stage2_il_{m}"]) <= EPS and row[f"orig_stage3_{m}"] > EPS]
        improved_items.append(join_list(improved))
        zero_repair_items.append(join_list(repairs))

    for metric in SAFETY_METRICS:
        safety_not_down &= df[f"delta_{metric}"] >= -EPS
        any_safety_up |= df[f"delta_{metric}"] > EPS
        any_safety_zero_repair |= (
            (df[f"stage2_il_{metric}"].abs() <= EPS) & (df[f"orig_stage3_{metric}"] > EPS)
        )

    masks = {
        "EP_not_down": ep_not_down,
        "EP_not_down_and_any_safety_item_up": ep_not_down & any_safety_up,
        "EP_not_down_and_SafetyMean_up": ep_not_down & (df["delta_SafetyMean"] > EPS),
        "EP_not_down_and_safety_items_not_down_and_any_safety_item_up": ep_not_down & safety_not_down & any_safety_up,
        "EP_strict_up_and_safety_items_not_down_and_any_safety_item_up": ep_strict_up & safety_not_down & any_safety_up,
        "EP_not_down_and_safety_items_not_down_and_zero_to_positive_safety_repair": (
            ep_not_down & safety_not_down & any_safety_zero_repair
        ),
    }
    rows = []
    for condition, mask in masks.items():
        sub = df.loc[mask]
        rows.append(
            {
                "condition": condition,
                "count": int(mask.sum()),
                "pct_total": pct(int(mask.sum()), len(df)),
                "mean_delta_PDMS": float(sub["delta_PDMS"].mean()) if len(sub) else None,
                "mean_delta_EP": float(sub["delta_EP"].mean()) if len(sub) else None,
                "mean_delta_SafetyMean": float(sub["delta_SafetyMean"].mean()) if len(sub) else None,
                "pdms_up_count": int((sub["delta_PDMS"] > EPS).sum()),
                "pdms_down_count": int((sub["delta_PDMS"] < -EPS).sum()),
            }
        )

    main_mask = masks["EP_not_down_and_safety_items_not_down_and_any_safety_item_up"]
    token_df = df.loc[main_mask].copy()
    token_df["improved_safety_items"] = [improved_items[i] for i in token_df.index]
    token_df["zero_to_positive_safety_items"] = [zero_repair_items[i] for i in token_df.index]

    breakdown_rows = []
    upper_mask = ep_not_down & safety_not_down
    for metric in SAFETY_METRICS:
        metric_up = upper_mask & (df[f"delta_{metric}"] > EPS)
        sub = df.loc[metric_up]
        zero_rep = metric_up & (df[f"stage2_il_{metric}"].abs() <= EPS) & (df[f"orig_stage3_{metric}"] > EPS)
        breakdown_rows.append(
            {
                "safety_metric": metric,
                "improved_count_under_ep_not_down_and_safety_not_down": int(metric_up.sum()),
                "pct_total": pct(int(metric_up.sum()), len(df)),
                "zero_to_positive_count": int(zero_rep.sum()),
                "mean_delta_metric_on_improved": float(sub[f"delta_{metric}"].mean()) if len(sub) else None,
                "mean_delta_PDMS_on_improved": float(sub["delta_PDMS"].mean()) if len(sub) else None,
                "mean_delta_EP_on_improved": float(sub["delta_EP"].mean()) if len(sub) else None,
            }
        )

    combo = (
        token_df.groupby("improved_safety_items", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "improved_safety_items"], ascending=[False, True])
    )
    combo["pct_of_main"] = combo["count"].map(lambda x: pct(int(x), len(token_df)))
    return pd.DataFrame(rows), pd.DataFrame(breakdown_rows), combo, token_df


def write_markdown(
    out_dir: Path,
    aligned: pd.DataFrame,
    summary: pd.DataFrame,
    safety_ep: pd.DataFrame,
    ep_safety: pd.DataFrame,
    breakdown: pd.DataFrame,
    combo: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metric_name = {
        "PDMS": "总分 PDMS",
        "NC": "无责任碰撞 NC",
        "DAC": "可行驶区域 DAC",
        "EP": "进展 EP",
        "TTC": "时距碰撞 TTC",
        "Comfort": "舒适性 Comfort",
        "DDC": "行驶方向 DDC",
        "SafetyMean": "安全均值 SafetyMean",
    }
    lines = [
        "# 原版 Stage3 RL 相比原版 Stage2 IL 的 navtest 细粒度统计",
        "",
        "日期：2026-07-03",
        "",
        "对比口径：原版 ReCogDrive Stage3 RL local reproduction `epoch9_step13300` "
        "vs A0-official-aligned Stage2 IL `step_00100000`。"
        f"统计基于 full navtest 对齐后的 {len(aligned)} 个真实有效 token，已剔除聚合 `average` 行。",
        "",
        "## 数据来源",
        "",
        f"- 原版 Stage3 RL 逐场景列：`{args.stage3_source_csv}` 中的 `orig_*` 列",
        f"- Stage2 IL 逐场景表：`{args.stage2_pdm_csv}`",
        f"- Stage2 IL 汇总指标：`{args.stage2_metrics_json}`",
        "",
        "## 口径说明",
        "",
        "- `Stage3 > Stage2 IL`：该指标逐 token 分数严格上升。",
        "- `0 -> >0`：Stage2 IL 该指标为 0，Stage3 该指标大于 0。",
        "- `>0 -> 0`：Stage2 IL 该指标大于 0，Stage3 退化为 0。",
        "- `安全子项逐项不下降`：NC、DAC、TTC、Comfort、DDC 五项逐项满足 `Stage3 >= Stage2 IL`。",
        "- `SafetyMean`：NC、DAC、TTC、Comfort、DDC 五个安全相关子项的简单平均，仅用于分析安全整体变化。",
        "",
        "## 指标整体变化",
        "",
        "| 指标 | Stage2 IL均值 | 原版Stage3均值 | 均值变化 | Stage3>Stage2 IL | 持平 | Stage3<Stage2 IL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {metric_name[row.metric]} | {row.stage2_il_mean:.6f} | {row.orig_stage3_mean:.6f} | "
            f"{row.mean_delta:+.6f} | {int(row.orig_stage3_gt_stage2_il_count)} "
            f"({row.orig_stage3_gt_stage2_il_pct:.2f}%) | "
            f"{int(row.orig_stage3_eq_stage2_il_count)} ({row.orig_stage3_eq_stage2_il_pct:.2f}%) | "
            f"{int(row.orig_stage3_lt_stage2_il_count)} ({row.orig_stage3_lt_stage2_il_pct:.2f}%) |"
        )
    pdms_row = summary[summary["metric"] == "PDMS"].iloc[0]
    ep_row = summary[summary["metric"] == "EP"].iloc[0]
    safety_row = summary[summary["metric"] == "SafetyMean"].iloc[0]
    lines += [
        "",
        f"结论：原版 Stage3 相比 Stage2 IL 的 PDMS 均值提升 {pdms_row.mean_delta:+.6f}，"
        f"EP 均值提升 {ep_row.mean_delta:+.6f}，SafetyMean 均值提升 {safety_row.mean_delta:+.6f}。",
        "",
        "## 0 分修复与退化",
        "",
        "| 指标 | Stage2 IL 0分场景数 | 0 -> >0 | 0 -> 仍为0 | >0 -> 0 | 0分净修复数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {metric_name[row.metric]} | {int(row.stage2_il_zero_count)} | "
            f"{int(row.zero_to_positive_count)} ({row.zero_to_positive_pct:.2f}%) | "
            f"{int(row.zero_still_zero_count)} | {int(row.positive_to_zero_count)} "
            f"({row.positive_to_zero_pct:.2f}%) | {int(row.zero_net_repair_count):+d} |"
        )
    lines += ["", "关键读数：", ""]
    for metric in ["PDMS", "EP", "NC", "DAC", "TTC", "DDC"]:
        row = summary[summary["metric"] == metric].iloc[0]
        lines.append(
            f"- {metric_name[metric]}：Stage2 IL 0 分的 {int(row.stage2_il_zero_count)} 个场景中，"
            f"原版 Stage3 有 {int(row.zero_to_positive_count)} 个修复到 >0；同时有 "
            f"{int(row.positive_to_zero_count)} 个从 >0 退化到 0，净修复 "
            f"{int(row.zero_net_repair_count):+d}。"
        )
    lines += [
        "",
        "## 安全不下降前提下 EP 提升",
        "",
        "| 安全条件 | EP提升阈值 | 场景数 | 占全量比例 | 平均EP提升 | 平均PDMS提升 | 平均SafetyMean变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = [
        "safety_items_not_decreased",
        "SafetyMean_not_decreased",
        "orig_stage3_all_safety_full_score",
        "both_all_safety_full_score",
        "safety_items_not_decreased_and_orig_stage3_all_safety_full_score",
    ]
    for condition in preferred:
        row = safety_ep[(safety_ep["safety_condition"] == condition) & (safety_ep["ep_delta_threshold"] == 0.0)].iloc[0]
        lines.append(
            f"| {condition} | >0 | {int(row['count'])} | {row['pct_total']:.2f}% | "
            f"{row['mean_ep_delta_on_selected']:+.6f} | {row['mean_pdms_delta_on_selected']:+.6f} | "
            f"{row['mean_safetymean_delta_on_selected']:+.6f} |"
        )
    main = safety_ep[
        (safety_ep["safety_condition"] == "safety_items_not_decreased")
        & (safety_ep["ep_delta_threshold"] == 0.0)
    ].iloc[0]
    lines += [
        "",
        f"主口径下，安全五项逐项不下降且 EP 严格提升的场景有 {int(main['count'])} 个，"
        f"占全量 {main['pct_total']:.2f}%。这些场景平均 EP 提升 "
        f"{main['mean_ep_delta_on_selected']:+.6f}，平均 PDMS 提升 "
        f"{main['mean_pdms_delta_on_selected']:+.6f}。",
        "",
        "## EP 不下降前提下安全提升",
        "",
        "| 口径 | 场景数 | 占全量比例 | 平均PDMS变化 | 平均EP变化 | 平均SafetyMean变化 | PDMS提升数 | PDMS下降数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ep_safety.iterrows():
        lines.append(
            f"| {row.condition} | {int(row['count'])} | {row.pct_total:.2f}% | "
            f"{row.mean_delta_PDMS:+.6f} | {row.mean_delta_EP:+.6f} | "
            f"{row.mean_delta_SafetyMean:+.6f} | {int(row.pdms_up_count)} | {int(row.pdms_down_count)} |"
        )
    lines += [
        "",
        "分安全项看，在 `EP 不下降 + 安全逐项不下降` 的上层条件下：",
        "",
        "| 安全项 | 提升场景数 | 占全量比例 | 其中0->>0 | 该安全项平均提升 | 平均PDMS变化 | 平均EP变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in breakdown.iterrows():
        metric_delta = "n/a" if pd.isna(row.mean_delta_metric_on_improved) else f"{row.mean_delta_metric_on_improved:+.6f}"
        pdms_delta = "n/a" if pd.isna(row.mean_delta_PDMS_on_improved) else f"{row.mean_delta_PDMS_on_improved:+.6f}"
        ep_delta = "n/a" if pd.isna(row.mean_delta_EP_on_improved) else f"{row.mean_delta_EP_on_improved:+.6f}"
        lines.append(
            f"| {row.safety_metric} | {int(row.improved_count_under_ep_not_down_and_safety_not_down)} | "
            f"{row.pct_total:.2f}% | {int(row.zero_to_positive_count)} | {metric_delta} | {pdms_delta} | {ep_delta} |"
        )
    lines += [
        "",
        "## 安全提升组合",
        "",
        "| 改善安全项组合 | 场景数 | 占主口径比例 |",
        "|---|---:|---:|",
    ]
    for _, row in combo.head(20).iterrows():
        lines.append(f"| {row.improved_safety_items} | {int(row['count'])} | {row.pct_of_main:.2f}% |")
    lines += [
        "",
        "## 输出文件",
        "",
        "- 对齐逐场景表：`original_stage3_rl_vs_stage2_il_scene_delta.csv`",
        "- 指标汇总：`detailed_metric_delta_summary.csv`",
        "- `0 -> >0` token 列表：`zero_to_positive_tokens_by_metric.csv`",
        "- `>0 -> 0` token 列表：`positive_to_zero_tokens_by_metric.csv`",
        "- 安全不下降且 EP 提升 token 列表：`tokens_safety_not_decreased_ep_improved.csv`",
        "- EP 不下降且安全提升 token 列表：`tokens_ep_not_down_safety_improved.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage3 = read_stage3(args.stage3_source_csv)
    stage2 = read_stage2(args.stage2_pdm_csv)
    aligned = stage3.merge(stage2, on="token", how="inner", validate="one_to_one")
    if aligned.empty:
        raise RuntimeError("No common tokens between Stage3 and Stage2 inputs.")
    if len(aligned) != len(stage3) or len(aligned) != len(stage2):
        warning = {
            "stage3_rows": len(stage3),
            "stage2_rows": len(stage2),
            "common_rows": len(aligned),
            "stage3_only": len(set(stage3["token"]) - set(stage2["token"])),
            "stage2_only": len(set(stage2["token"]) - set(stage3["token"])),
        }
        (args.out_dir / "alignment_warning.json").write_text(json.dumps(warning, indent=2), encoding="utf-8")

    for metric in METRICS:
        aligned[f"delta_{metric}"] = aligned[f"orig_stage3_{metric}"] - aligned[f"stage2_il_{metric}"]

    ordered = ["token", "log_name", "scene_token", "chunk"]
    for metric in METRICS:
        ordered += [f"orig_stage3_{metric}", f"stage2_il_{metric}", f"delta_{metric}"]
    aligned = aligned[ordered].sort_values("token").reset_index(drop=True)
    aligned.to_csv(args.out_dir / "original_stage3_rl_vs_stage2_il_scene_delta.csv", index=False)

    summary = metric_summary(aligned)
    summary.to_csv(args.out_dir / "detailed_metric_delta_summary.csv", index=False)
    zero_transition_tokens(aligned, "zero_to_positive").to_csv(
        args.out_dir / "zero_to_positive_tokens_by_metric.csv", index=False
    )
    zero_transition_tokens(aligned, "positive_to_zero").to_csv(
        args.out_dir / "positive_to_zero_tokens_by_metric.csv", index=False
    )

    safety_not_down = pd.Series(True, index=aligned.index)
    for metric in SAFETY_METRICS:
        safety_not_down &= aligned[f"delta_{metric}"] >= -EPS
    aligned.loc[safety_not_down & (aligned["delta_EP"] > EPS)].to_csv(
        args.out_dir / "tokens_safety_not_decreased_ep_improved.csv", index=False
    )

    safety_ep = safety_preserved_ep_summary(aligned)
    safety_ep.to_csv(args.out_dir / "safety_preserved_ep_improvement_summary.csv", index=False)
    ep_safety, breakdown, combo, token_ep_safety = ep_preserved_safety_stats(aligned)
    ep_safety.to_csv(args.out_dir / "ep_preserved_safety_improvement_summary.csv", index=False)
    breakdown.to_csv(args.out_dir / "ep_preserved_safety_improvement_breakdown.csv", index=False)
    combo.to_csv(args.out_dir / "ep_preserved_safety_improvement_combinations.csv", index=False)
    token_ep_safety.to_csv(args.out_dir / "tokens_ep_not_down_safety_improved.csv", index=False)
    aligned.sort_values(["delta_PDMS", "delta_EP", "delta_SafetyMean"], ascending=False).head(200).to_csv(
        args.out_dir / "top_positive_pdms_cases.csv", index=False
    )
    aligned.sort_values(["delta_PDMS", "delta_EP", "delta_SafetyMean"], ascending=True).head(200).to_csv(
        args.out_dir / "top_negative_pdms_cases.csv", index=False
    )

    provenance = {
        "stage3_source_csv": str(args.stage3_source_csv),
        "stage2_pdm_csv": str(args.stage2_pdm_csv),
        "stage2_metrics_json": str(args.stage2_metrics_json) if args.stage2_metrics_json.exists() else None,
        "num_common_tokens": int(len(aligned)),
        "orig_stage3_pdms": float(aligned["orig_stage3_PDMS"].mean()),
        "stage2_il_pdms": float(aligned["stage2_il_PDMS"].mean()),
        "delta_pdms": float(aligned["delta_PDMS"].mean()),
    }
    if args.stage2_metrics_json.exists():
        provenance["stage2_metrics"] = json.loads(args.stage2_metrics_json.read_text(encoding="utf-8"))
    (args.out_dir / "summary.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out_dir, aligned, summary, safety_ep, ep_safety, breakdown, combo, args)
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
