#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


RISK_CLASSES = ["low_score", "path_dac", "interaction_nc", "ttc", "progress", "comfort"]
PRED_PREFIXES = ("risk_vla_prob_", "prob_", "pred_", "")
LABEL_PREFIXES = ("label_", "risk_label_", "")
STRATEGY_CLASSES = ["base", "path_intent", "interaction", "progress", "comfort"]
STRATEGY_PREFIXES = ("strategy_weight_", "risk_vla_weight_")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _scene_label_from_value(value: object, index: int) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return float(arr[index])
    if arr.ndim == 2:
        return float(arr[:, index].mean())
    raise ValueError(f"risk label value must be [K] or [H, K], got shape {arr.shape}.")


def _labels_jsonl_to_df(path: Path) -> pd.DataFrame:
    records: list[dict] = []
    for row in _read_jsonl(path):
        record: dict[str, object] = {}
        token = row.get("token") or row.get("sample_token") or row.get("scene_token")
        if token is not None:
            record["token"] = token
        if "risk_labels" in row:
            for idx, name in enumerate(RISK_CLASSES):
                record[name] = _scene_label_from_value(row["risk_labels"], idx)
        elif all(key in row for key in ("generic_risk_labels", "drivable_risk_labels", "ttc_risk_labels", "comfort_risk_labels")):
            mvp = {
                "low_score": row["generic_risk_labels"],
                "path_dac": row["drivable_risk_labels"],
                "interaction_nc": row["ttc_risk_labels"],
                "ttc": row["ttc_risk_labels"],
                "progress": row["comfort_risk_labels"],
                "comfort": row["comfort_risk_labels"],
            }
            for name in RISK_CLASSES:
                record[name] = float(np.asarray(mvp[name], dtype=float).mean())
        else:
            for name in RISK_CLASSES:
                if name in row:
                    record[name] = float(row[name])
                elif f"label_{name}" in row:
                    record[name] = float(row[f"label_{name}"])
        records.append(record)
    return pd.DataFrame(records)


def _read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return _labels_jsonl_to_df(path)
    df = pd.read_csv(path)
    out = pd.DataFrame()
    if "token" in df.columns:
        out["token"] = df["token"]
    for name in RISK_CLASSES:
        for prefix in LABEL_PREFIXES:
            col = f"{prefix}{name}"
            if col in df.columns:
                out[name] = pd.to_numeric(df[col], errors="coerce")
                break
    return out


def _prediction_column(df: pd.DataFrame, class_name: str) -> Optional[str]:
    for prefix in PRED_PREFIXES:
        col = f"{prefix}{class_name}"
        if col in df.columns:
            return col
    return None


def _strategy_column(df: pd.DataFrame, strategy_name: str) -> Optional[str]:
    for prefix in STRATEGY_PREFIXES:
        col = f"{prefix}{strategy_name}"
        if col in df.columns:
            return col
    return None


def _prepare_matched(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    pred = predictions.copy()
    label = labels.copy()
    if "token" in pred.columns and "token" in label.columns:
        return pred.merge(label, on="token", how="inner", suffixes=("", "_label"))
    count = min(len(pred), len(label))
    merged = pred.iloc[:count].reset_index(drop=True)
    for name in RISK_CLASSES:
        if name in label.columns:
            merged[name] = label[name].iloc[:count].reset_index(drop=True)
    return merged


def prepare_matched_predictions(predictions_csv: Path, labels_path: Path) -> pd.DataFrame:
    return _prepare_matched(pd.read_csv(predictions_csv), _read_labels(labels_path))


def _precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    if y_true.size == 0:
        return float("nan"), float("nan")
    k_eff = min(int(k), int(y_true.size))
    order = np.argsort(-y_score)[:k_eff]
    positives = float(y_true.sum())
    hits = float(y_true[order].sum())
    precision = hits / max(float(k_eff), 1.0)
    recall = hits / positives if positives > 0.0 else float("nan")
    return precision, recall


def _binary_mutual_information(labels: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(labels) & np.isfinite(values)
    labels = labels[valid] >= 0.5
    values = values[valid]
    if labels.size == 0:
        return float("nan")
    threshold = np.nanmedian(values)
    active = values >= threshold
    total = float(labels.size)
    mi = 0.0
    for label_value in (False, True):
        for active_value in (False, True):
            joint = float(np.logical_and(labels == label_value, active == active_value).sum()) / total
            if joint <= 0.0:
                continue
            p_label = float((labels == label_value).sum()) / total
            p_active = float((active == active_value).sum()) / total
            mi += joint * np.log(joint / max(p_label * p_active, 1e-12))
    return float(mi)


def strategy_activation_by_subset(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategy_cols = {name: _strategy_column(data, name) for name in STRATEGY_CLASSES}
    if not any(strategy_cols.values()):
        return pd.DataFrame()
    for risk_name in RISK_CLASSES:
        if risk_name not in data.columns:
            continue
        labels = pd.to_numeric(data[risk_name], errors="coerce").to_numpy(dtype=float)
        positive = labels >= 0.5
        row: dict[str, object] = {
            "risk_subset": risk_name,
            "num_rows": int(np.isfinite(labels).sum()),
            "num_positive": int(np.logical_and(np.isfinite(labels), positive).sum()),
            "positive_rate": float(np.nanmean(positive.astype(float))) if labels.size else float("nan"),
        }
        for strategy_name, col in strategy_cols.items():
            if col is None:
                continue
            values = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            row[f"mean_{strategy_name}_positive"] = float(np.nanmean(values[positive])) if positive.any() else float("nan")
            row[f"mean_{strategy_name}_negative"] = float(np.nanmean(values[~positive])) if (~positive).any() else float("nan")
            row[f"mi_{strategy_name}"] = _binary_mutual_information(labels, values)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(predictions_csv: Path, labels_path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings_list: list[str] = []
    pred = pd.read_csv(predictions_csv)
    labels = _read_labels(labels_path)
    data = _prepare_matched(pred, labels)
    if data.empty:
        raise ValueError("No matched prediction/label rows.")

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        has_sklearn = True
    except Exception as exc:
        average_precision_score = None
        roc_auc_score = None
        has_sklearn = False
        warnings_list.append(f"sklearn unavailable; skipping AUC/AP ({exc}).")

    rows: list[dict[str, object]] = []
    for name in RISK_CLASSES:
        pred_col = _prediction_column(data, name)
        if pred_col is None:
            warnings_list.append(f"Missing prediction column for {name}.")
            continue
        if name not in data.columns:
            warnings_list.append(f"Missing label column for {name}.")
            continue
        y_score = pd.to_numeric(data[pred_col], errors="coerce").to_numpy(dtype=float)
        y_label = pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_score) & np.isfinite(y_label)
        y_score = np.clip(y_score[valid], 0.0, 1.0)
        y_true = (y_label[valid] >= 0.5).astype(float)
        row: dict[str, object] = {
            "risk_class": name,
            "num_rows": int(y_true.size),
            "positive_rate": float(y_true.mean()) if y_true.size else float("nan"),
            "mean_predicted_probability": float(y_score.mean()) if y_score.size else float("nan"),
            "brier_score": float(np.mean((y_score - y_true) ** 2)) if y_true.size else float("nan"),
        }
        if has_sklearn and y_true.size and 0.0 < y_true.sum() < y_true.size:
            row["auc"] = float(roc_auc_score(y_true, y_score))
            row["ap"] = float(average_precision_score(y_true, y_score))
        else:
            row["auc"] = float("nan")
            row["ap"] = float("nan")
        global_rate = max(float(row["positive_rate"]), 1e-12)
        for k in (50, 100, 200):
            precision, recall = _precision_recall_at_k(y_true, y_score, k)
            row[f"precision_at_{k}"] = precision
            row[f"recall_at_{k}"] = recall
            row[f"enrichment_at_{k}"] = precision / global_rate if np.isfinite(precision) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows), warnings_list


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No metrics were produced."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_topk_exports(data: pd.DataFrame, output_dir: Path, *, top_k: int = 200) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in RISK_CLASSES:
        pred_col = _prediction_column(data, name)
        if pred_col is None:
            continue
        cols = [col for col in ["token", "split", "log_name", name, pred_col] if col in data.columns]
        for strategy in STRATEGY_CLASSES:
            col = _strategy_column(data, strategy)
            if col is not None:
                cols.append(col)
        top = data.sort_values(pred_col, ascending=False).head(int(top_k))
        top[cols].to_csv(output_dir / f"topk_risk_tokens_{name}.csv", index=False)


def write_markdown(
    summary: pd.DataFrame,
    warnings_list: Iterable[str],
    path: Path,
    strategy_activation: Optional[pd.DataFrame] = None,
) -> None:
    lines = ["# RISK-VLA Diagnostic Aggregation", ""]
    if warnings_list:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings_list)
        lines.append("")
    lines.extend(["## Risk Prediction Metrics", "", _markdown_table(summary), ""])
    if strategy_activation is not None:
        lines.extend(["## Strategy Activation By Risk Subset", "", _markdown_table(strategy_activation), ""])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_all(predictions_csv: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    summary, warnings_list = aggregate(predictions_csv, labels_path)
    data = prepare_matched_predictions(predictions_csv, labels_path)
    activation = strategy_activation_by_subset(data)
    return summary, activation, data, warnings_list


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISK-VLA risk prediction diagnostics.")
    parser.add_argument("--predictions-csv", required=True)
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument("--labels-jsonl")
    label_group.add_argument("--labels-csv")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()
    if args.output_csv is None and args.output_md is None and args.output_dir is None:
        parser.error("Set --output-csv/--output-md or --output-dir.")

    labels_path = Path(args.labels_jsonl or args.labels_csv)
    summary, activation, matched, warnings_list = aggregate_all(Path(args.predictions_csv), labels_path)
    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)
    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, warnings_list, output_md, activation)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "risk_prediction_metrics.csv", index=False)
        activation.to_csv(output_dir / "strategy_activation_by_subset.csv", index=False)
        write_topk_exports(matched, output_dir, top_k=args.top_k)
        write_markdown(summary, warnings_list, output_dir / "risk_vla_diagnostics.md", activation)
    for warning in warnings_list:
        warnings.warn(warning, RuntimeWarning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
