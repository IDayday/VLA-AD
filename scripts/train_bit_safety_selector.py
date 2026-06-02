#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import (  # noqa: E402
    SELECTOR_FEATURE_NAMES,
    FeatureMLPSelector,
    feature_tensor,
)
from scripts.eval_bit_safety_router import aggregate as aggregate_metric_rows  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_metadata(jsonl_path: Path) -> Dict[str, Any]:
    meta_path = jsonl_path.parent / "metadata.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def require_training_allowed(jsonl_path: Path) -> None:
    metadata = load_metadata(jsonl_path)
    split = str(metadata.get("split", "")).lower() if metadata else ""
    analysis_only = bool(metadata.get("analysis_only", False)) if metadata else False
    if "test" in split or analysis_only:
        raise RuntimeError(
            f"{jsonl_path} is not allowed for selector training. "
            f"metadata.training_allowed={metadata.get('training_allowed')} "
            f"split={metadata.get('split')!r} analysis_only={analysis_only}."
        )


def metric_row(row: Dict[str, Any], source: str, selection_source: str) -> Dict[str, Any]:
    metrics = row.get(f"{source}_metrics") or {}
    return {
        "sample_token": row.get("sample_token"),
        "scene_token": row.get("scene_token"),
        "selection_source": selection_source,
        "score": metrics.get("pdm"),
        "drivable_area_compliance": metrics.get("dac"),
        "no_at_fault_collisions": metrics.get("nc"),
        "time_to_collision_within_bound": metrics.get("ttc"),
        "ego_progress": metrics.get("ego"),
        "comfort": metrics.get("comfort"),
        "valid": bool(metrics),
    }


def aggregate_selection(rows: Sequence[Dict[str, Any]], use_bit: Sequence[bool]) -> Dict[str, Any]:
    metric_rows = [
        metric_row(row, "bit" if bit else "base", "bit" if bit else "base")
        for row, bit in zip(rows, use_bit)
    ]
    return aggregate_metric_rows(metric_rows)


def load_dataset(path: Path, feature_names: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    rows = [row for row in read_jsonl(path) if row.get("label_use_bit") in (0, 1)]
    if not rows:
        raise RuntimeError(f"No labeled rows found in {path}")
    x = feature_tensor((row.get("features", {}) for row in rows), feature_names)
    y = torch.tensor([float(row["label_use_bit"]) for row in rows], dtype=torch.float32)
    return x, y, rows


def threshold_sweep(
    rows: Sequence[Dict[str, Any]],
    probs: torch.Tensor,
    *,
    safety_first: bool,
) -> Tuple[float, List[Dict[str, Any]], bool]:
    thresholds = [round(i / 100.0, 2) for i in range(0, 101)]
    base_agg = aggregate_selection(rows, [False] * len(rows))
    sweep_rows: List[Dict[str, Any]] = []
    feasible: List[Dict[str, Any]] = []
    for threshold in thresholds:
        use_bit = (probs >= threshold).cpu().numpy().astype(bool).tolist()
        agg = aggregate_selection(rows, use_bit)
        ok = (
            int(agg.get("no_at_fault_collision_zero_count") or 0) <= int(base_agg.get("no_at_fault_collision_zero_count") or 0) + 1
            and int(agg.get("time_to_collision_zero_count") or 0) <= int(base_agg.get("time_to_collision_zero_count") or 0) + 1
            and int(agg.get("drivable_area_compliance_zero_count") or 0) <= int(base_agg.get("drivable_area_compliance_zero_count") or 0)
        )
        row = {
            "threshold": threshold,
            "safety_ok": ok,
            **agg,
        }
        sweep_rows.append(row)
        if ok:
            feasible.append(row)
    if feasible:
        best = max(feasible, key=lambda row: (float(row.get("mean_pdms") or -1.0), -int(row.get("zero_score_count") or 10**9)))
        return float(best["threshold"]), sweep_rows, True
    if safety_first:
        best = min(
            sweep_rows,
            key=lambda row: (
                int(row.get("no_at_fault_collision_zero_count") or 10**9),
                int(row.get("time_to_collision_zero_count") or 10**9),
                int(row.get("drivable_area_compliance_zero_count") or 10**9),
                -float(row.get("mean_pdms") or -1.0),
            ),
        )
    else:
        best = max(sweep_rows, key=lambda row: float(row.get("mean_pdms") or -1.0))
    return float(best["threshold"]), sweep_rows, False


def write_sweep(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "threshold",
        "safety_ok",
        "mean_pdms",
        "p10_pdms",
        "zero_score_count",
        "drivable_area_compliance_zero_count",
        "no_at_fault_collision_zero_count",
        "time_to_collision_zero_count",
        "ego_progress_mean",
        "selection_bit_count",
        "selection_base_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight FeatureMLP BiT safety selector.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-names-json", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--threshold-mode", choices=("fixed", "val_sweep"), default="val_sweep")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--safety-first", action="store_true")
    return parser.parse_args()


def load_feature_names(args: argparse.Namespace) -> List[str]:
    if args.feature_names_json is not None:
        return [str(name) for name in json.loads(args.feature_names_json.read_text(encoding="utf-8"))]
    for candidate in (args.train_jsonl.parent / "feature_names.json", args.val_jsonl.parent / "feature_names.json"):
        if candidate.is_file():
            return [str(name) for name in json.loads(candidate.read_text(encoding="utf-8"))]
    return list(SELECTOR_FEATURE_NAMES)


def main() -> int:
    args = parse_args()
    require_training_allowed(args.train_jsonl)
    require_training_allowed(args.val_jsonl)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = load_feature_names(args)
    x_train, y_train, train_rows = load_dataset(args.train_jsonl, feature_names)
    x_val, y_val, val_rows = load_dataset(args.val_jsonl, feature_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureMLPSelector(x_train.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val_device = x_val.to(device)
    if args.positive_weight > 0.0:
        pos_weight = torch.tensor(args.positive_weight, device=device)
    else:
        positives = max(float(y_train.sum().item()), 1.0)
        negatives = max(float((1.0 - y_train).sum().item()), 1.0)
        pos_weight = torch.tensor(negatives / positives, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = x_train.shape[0]
    for epoch in range(args.epochs):
        order = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                val_probs = torch.sigmoid(model(x_val_device)).cpu()
                val_pred = (val_probs >= 0.5).float()
                acc = float((val_pred == y_val).float().mean().item())
            print(f"epoch={epoch + 1} loss={sum(losses) / max(len(losses), 1):.6f} val_acc@0.5={acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val_device)).cpu()
    if args.threshold_mode == "val_sweep":
        threshold, sweep_rows, threshold_satisfies_safety = threshold_sweep(val_rows, val_probs, safety_first=args.safety_first)
    else:
        threshold = args.fixed_threshold
        use_bit = (val_probs >= threshold).numpy().astype(bool).tolist()
        row = {"threshold": threshold, "safety_ok": None, **aggregate_selection(val_rows, use_bit)}
        sweep_rows = [row]
        threshold_satisfies_safety = False
    write_sweep(args.output_dir / "val_threshold_sweep.csv", sweep_rows)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": x_train.shape[1],
        "feature_names": feature_names,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }, args.output_dir / "selector.pt")
    config = {
        "selector_type": "FeatureMLPSelector",
        "feature_names": feature_names,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "threshold_satisfies_safety": threshold_satisfies_safety,
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "positive_weight": float(pos_weight.detach().cpu().item()),
    }
    (args.output_dir / "selector_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chosen = min(sweep_rows, key=lambda row: abs(float(row["threshold"]) - threshold))
    lines = [
        "# BiT FeatureMLP Selector Validation",
        "",
        f"Train samples: {len(train_rows)}",
        f"Val samples: {len(val_rows)}",
        f"Positive rate train/val: {float(y_train.mean().item()):.4f} / {float(y_val.mean().item()):.4f}",
        f"Chosen threshold: {threshold}",
        f"Safety constraints satisfied: `{threshold_satisfies_safety}`",
        "",
        "| Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {chosen.get('mean_pdms')} | {chosen.get('p10_pdms')} | {chosen.get('zero_score_count')} | "
        f"{chosen.get('drivable_area_compliance_zero_count')} | {chosen.get('no_at_fault_collision_zero_count')} | "
        f"{chosen.get('time_to_collision_zero_count')} | {chosen.get('ego_progress_mean')} | {chosen.get('selection_bit_count')} |",
    ]
    (args.output_dir / "val_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
