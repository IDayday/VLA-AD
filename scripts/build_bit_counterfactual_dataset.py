#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.bit_safety_selector import trajectory_delta_features  # noqa: E402
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from scripts.eval_bit_drive_pdm import (  # noqa: E402
    build_metric_cache_loader,
    build_pdm_tools,
    build_planner,
    chunk_dirs,
    dtype_from_precision,
    left_tail_metrics,
    load_checkpoint,
    load_sample,
    load_yaml,
    make_batch,
    sample_paths,
    seed_everything,
)


METRIC_KEYS = {
    "pdm": "score",
    "dac": "drivable_area_compliance",
    "nc": "no_at_fault_collisions",
    "ttc": "time_to_collision_within_bound",
    "ego": "ego_progress",
    "comfort": "comfort",
}


def resolve_checkpoint_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.name in {"best.ckpt", "latest.ckpt"} and path.parent.is_dir():
        for name in ("best.ckpt", "latest.ckpt"):
            candidate = path.parent / name
            if candidate.is_file():
                return candidate
    return path


def split_chunk_pattern(split: str) -> str:
    normalized = split.lower()
    if normalized in {"navtest", "test"}:
        return "navtest_full_chunk_*"
    if normalized in {"navtrain", "train"}:
        return "train_full_chunk_*"
    if normalized in {"navval", "val", "validation"}:
        return "navval_full_chunk_*"
    return f"{split}_full_chunk_*"


def default_metric_cache(split: str) -> Optional[Path]:
    if split.lower() in {"navtest", "test"}:
        return Path("/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1")
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def metrics_from_eval_dir(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = eval_dir / "per_sample_metrics.jsonl"
    return {str(row.get("sample_token") or row.get("scene_token")): row for row in read_jsonl(path)}


def predictions_from_eval_dir(eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = eval_dir / "predictions.jsonl"
    return {str(row.get("sample_token") or row.get("scene_token")): row for row in read_jsonl(path)}


def compact_metrics(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Optional[float]]]:
    if not row or not row.get("valid", True):
        return None
    out: Dict[str, Optional[float]] = {}
    for short, key in METRIC_KEYS.items():
        value = row.get(key)
        out[short] = None if value is None else float(value)
    return out


def score_prediction(
    *,
    sample_token: str,
    pred: torch.Tensor,
    metric_cache_loader: Any,
    pdm_tools: Tuple[Any, Any, Any],
) -> Dict[str, Any]:
    if sample_token not in metric_cache_loader.metric_cache_paths:
        return {"valid": False, "error": "missing_metric_cache"}
    try:
        metric_cache = metric_cache_loader.get_from_token(sample_token)
        future_sampling, simulator, scorer = pdm_tools
        result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=Trajectory(poses=pred.detach().float().cpu().numpy()),
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        return {"valid": True, **asdict(result)}
    except Exception as exc:  # pragma: no cover - depends on NAVSIM metric cache internals
        return {"valid": False, "error": repr(exc)}


def label_row(
    base_metrics: Optional[Dict[str, Optional[float]]],
    bit_metrics: Optional[Dict[str, Optional[float]]],
    *,
    margin_pdm: float,
    require_no_nc_regression: bool,
    require_no_ttc_regression: bool,
    require_dac_nonregression: bool,
) -> Tuple[Optional[int], str, Dict[str, Optional[float]]]:
    delta = {"pdm": None, "dac": None, "nc": None, "ttc": None}
    if base_metrics is None or bit_metrics is None:
        return None, "missing_metrics", delta
    for key in delta:
        if base_metrics.get(key) is not None and bit_metrics.get(key) is not None:
            delta[key] = float(bit_metrics[key]) - float(base_metrics[key])
    if delta["pdm"] is None:
        return None, "missing_pdm", delta
    if delta["pdm"] <= margin_pdm:
        return 0, "pdm_margin_not_met", delta
    if require_no_nc_regression and float(base_metrics.get("nc") or 0.0) > 1e-9 and float(bit_metrics.get("nc") or 0.0) <= 1e-9:
        return 0, "nc_regression", delta
    if require_no_ttc_regression and float(base_metrics.get("ttc") or 0.0) > 1e-9 and float(bit_metrics.get("ttc") or 0.0) <= 1e-9:
        return 0, "ttc_regression", delta
    if require_dac_nonregression and float(bit_metrics.get("dac") or 0.0) + 1e-9 < float(base_metrics.get("dac") or 0.0):
        return 0, "dac_regression", delta
    return 1, "bit_improves_pdm_and_safety_ok", delta


def build_from_eval_dirs(
    *,
    base_eval_dir: Path,
    bit_eval_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    base_metrics = metrics_from_eval_dir(base_eval_dir)
    bit_metrics = metrics_from_eval_dir(bit_eval_dir)
    base_preds = predictions_from_eval_dir(base_eval_dir)
    bit_preds = predictions_from_eval_dir(bit_eval_dir)
    rows = []
    for key in sorted(set(base_preds) & set(bit_preds) & set(base_metrics) & set(bit_metrics)):
        base_pred = base_preds[key]
        bit_pred = bit_preds[key]
        base_traj = base_pred["pred_traj"]
        bit_traj = bit_pred["pred_traj"]
        bm = compact_metrics(base_metrics[key])
        im = compact_metrics(bit_metrics[key])
        label, reason, delta = label_row(
            bm,
            im,
            margin_pdm=args.margin_pdm,
            require_no_nc_regression=args.require_no_nc_regression,
            require_no_ttc_regression=args.require_no_ttc_regression,
            require_dac_nonregression=args.require_dac_nonregression,
        )
        rows.append({
            "scene_token": base_pred.get("scene_token"),
            "sample_token": base_pred.get("sample_token"),
            "base_pred_traj": base_traj,
            "bit_pred_traj": bit_traj,
            "base_metrics": bm,
            "bit_metrics": im,
            "delta": delta,
            "label_use_bit": label,
            "label_reason": reason,
            "features": trajectory_delta_features(base_traj, bit_traj),
        })
    return rows[: args.max_samples] if args.max_samples is not None else rows


def build_by_running_models(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.metric_cache_dir is None:
        if args.analysis_only:
            metric_cache_loader = None
            pdm_tools = None
        else:
            raise RuntimeError(
                "A metric cache is required to build selector training labels. "
                "Use --analysis-only only for non-training analysis outputs."
            )
    else:
        metric_cache_loader = build_metric_cache_loader(args.metric_cache_dir)
        pdm_tools = build_pdm_tools()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision, device)
    base_planner = build_planner(load_yaml(args.base_config)).to(device)
    bit_planner = build_planner(load_yaml(args.bit_config)).to(device)
    if dtype != torch.float32:
        base_planner = base_planner.to(dtype=dtype)
        bit_planner = bit_planner.to(dtype=dtype)
    load_checkpoint(base_planner, resolve_checkpoint_path(args.base_checkpoint))
    load_checkpoint(bit_planner, resolve_checkpoint_path(args.bit_checkpoint))
    base_planner.eval()
    bit_planner.eval()

    chunks = chunk_dirs(args)
    rows = []
    for chunk, path, index_record in sample_paths(chunks, args.max_samples):
        sample = load_sample(path)
        vl_features, action_input = make_batch(sample, device, dtype)
        with torch.no_grad():
            base_output = base_planner.get_action(vl_features, action_input, deterministic=args.deterministic)
            bit_output = bit_planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        base_traj_tensor = base_output["pred_traj"].detach().float().cpu().squeeze(0)
        bit_traj_tensor = bit_output["pred_traj"].detach().float().cpu().squeeze(0)
        if not torch.isfinite(base_traj_tensor).all() or not torch.isfinite(bit_traj_tensor).all():
            raise RuntimeError(f"Non-finite trajectory for {path}")
        sample_token = str(sample.get("sample_token", index_record.get("sample_token", path.stem)))
        scene_token = str(sample.get("scene_token", index_record.get("scene_token", path.stem)))
        if metric_cache_loader is not None and pdm_tools is not None:
            bm_full = score_prediction(sample_token=sample_token, pred=base_traj_tensor, metric_cache_loader=metric_cache_loader, pdm_tools=pdm_tools)
            im_full = score_prediction(sample_token=sample_token, pred=bit_traj_tensor, metric_cache_loader=metric_cache_loader, pdm_tools=pdm_tools)
            bm = compact_metrics(bm_full)
            im = compact_metrics(im_full)
        else:
            bm_full = {"valid": False, "error": "metric_cache_not_provided"}
            im_full = {"valid": False, "error": "metric_cache_not_provided"}
            bm = None
            im = None
        label, reason, delta = label_row(
            bm,
            im,
            margin_pdm=args.margin_pdm,
            require_no_nc_regression=args.require_no_nc_regression,
            require_no_ttc_regression=args.require_no_ttc_regression,
            require_dac_nonregression=args.require_dac_nonregression,
        )
        rows.append({
            "scene_token": scene_token,
            "sample_token": sample_token,
            "base_pred_traj": base_traj_tensor.tolist(),
            "bit_pred_traj": bit_traj_tensor.tolist(),
            "base_metrics": bm,
            "bit_metrics": im,
            "base_metrics_raw": bm_full,
            "bit_metrics_raw": im_full,
            "delta": delta,
            "label_use_bit": label,
            "label_reason": reason,
            "features": trajectory_delta_features(base_traj_tensor, bit_traj_tensor),
        })
    return rows


def label_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reasons: Dict[str, int] = {}
    labels = {0: 0, 1: 0, None: 0}
    for row in rows:
        labels[row.get("label_use_bit")] = labels.get(row.get("label_use_bit"), 0) + 1
        reason = str(row.get("label_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "num_samples": len(rows),
        "label_0": labels.get(0, 0),
        "label_1": labels.get(1, 0),
        "label_missing": labels.get(None, 0),
        "positive_rate": labels.get(1, 0) / len(rows) if rows else None,
        "reasons": reasons,
    }


def metrics_rows(rows: List[Dict[str, Any]], side: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        metrics = row.get(f"{side}_metrics")
        out.append({
            "scene_token": row.get("scene_token"),
            "sample_token": row.get("sample_token"),
            "valid": metrics is not None,
            "score": None if metrics is None else metrics.get("pdm"),
            "drivable_area_compliance": None if metrics is None else metrics.get("dac"),
            "no_at_fault_collisions": None if metrics is None else metrics.get("nc"),
            "time_to_collision_within_bound": None if metrics is None else metrics.get("ttc"),
            "ego_progress": None if metrics is None else metrics.get("ego"),
            "comfort": None if metrics is None else metrics.get("comfort"),
        })
    return out


def write_outputs(rows: List[Dict[str, Any]], args: argparse.Namespace, metadata: Dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", rows)
    write_jsonl(
        args.output_dir / "base_predictions.jsonl",
        (
            {
                "scene_token": row.get("scene_token"),
                "sample_token": row.get("sample_token"),
                "pred_traj": row.get("base_pred_traj"),
                "metrics": row.get("base_metrics"),
            }
            for row in rows
        ),
    )
    write_jsonl(
        args.output_dir / "bit_predictions.jsonl",
        (
            {
                "scene_token": row.get("scene_token"),
                "sample_token": row.get("sample_token"),
                "pred_traj": row.get("bit_pred_traj"),
                "metrics": row.get("bit_metrics"),
            }
            for row in rows
        ),
    )
    distribution = label_distribution(rows)
    (args.output_dir / "label_distribution.json").write_text(json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    base_agg = left_tail_metrics(metrics_rows(rows, "base"))
    bit_agg = left_tail_metrics(metrics_rows(rows, "bit"))
    lines = [
        "# BiT Counterfactual Dataset Summary",
        "",
        f"Split: `{args.split}`",
        f"Samples: {len(rows)}",
        f"Training allowed: `{metadata['training_allowed']}`",
        f"Analysis only: `{args.analysis_only}`",
        "",
        "| Candidate | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Base | {base_agg.get('mean_pdms')} | {base_agg.get('p10_pdms')} | {base_agg.get('zero_score_count')} | {base_agg.get('drivable_area_compliance_zero_count')} | {base_agg.get('no_at_fault_collision_zero_count')} | {base_agg.get('time_to_collision_zero_count')} | {base_agg.get('ego_progress_mean')} |",
        f"| BiT | {bit_agg.get('mean_pdms')} | {bit_agg.get('p10_pdms')} | {bit_agg.get('zero_score_count')} | {bit_agg.get('drivable_area_compliance_zero_count')} | {bit_agg.get('no_at_fault_collision_zero_count')} | {bit_agg.get('time_to_collision_zero_count')} | {bit_agg.get('ego_progress_mean')} |",
        "",
        "## Label Distribution",
        "",
        "```json",
        json.dumps(distribution, indent=2, sort_keys=True),
        "```",
    ]
    (args.output_dir / "counterfactual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Base-vs-BiT counterfactual selector labels.")
    parser.add_argument("--base-config", type=Path, required=False)
    parser.add_argument("--base-checkpoint", type=Path, required=False)
    parser.add_argument("--bit-config", type=Path, required=False)
    parser.add_argument("--bit-checkpoint", type=Path, required=False)
    parser.add_argument("--base-eval-dir", type=Path, default=None)
    parser.add_argument("--bit-eval-dir", type=Path, default=None)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=Path("/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1"))
    parser.add_argument("--chunk-name-pattern", default=None)
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--margin-pdm", type=float, default=0.01)
    parser.add_argument("--require-no-nc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-no-ttc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-dac-nonregression", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stochastic:
        args.deterministic = False
    seed_everything(args.seed)
    if args.chunk_name_pattern is None:
        args.chunk_name_pattern = split_chunk_pattern(args.split)
    if args.metric_cache_dir is None:
        args.metric_cache_dir = default_metric_cache(args.split)

    using_eval_dirs = args.base_eval_dir is not None or args.bit_eval_dir is not None
    if using_eval_dirs and not (args.base_eval_dir and args.bit_eval_dir):
        raise ValueError("Set both --base-eval-dir and --bit-eval-dir, or neither.")
    if using_eval_dirs:
        rows = build_from_eval_dirs(base_eval_dir=args.base_eval_dir, bit_eval_dir=args.bit_eval_dir, args=args)
    else:
        required = [args.base_config, args.base_checkpoint, args.bit_config, args.bit_checkpoint]
        if any(value is None for value in required):
            raise ValueError("Model-running mode requires base/bit config and checkpoint arguments.")
        if args.metric_cache_dir is None and not args.analysis_only:
            raise RuntimeError(
                f"No metric cache configured for split={args.split!r}. "
                "Cannot create training labels without PDM metrics."
            )
        rows = build_by_running_models(args)

    split_lower = args.split.lower()
    label_missing_count = sum(1 for row in rows if row.get("label_use_bit") is None)
    # Missing labels can happen when a small number of metric-cache entries are
    # unavailable. They should be filtered by selector training, not reclassify
    # a non-test counterfactual dataset as navtest/analysis-only.
    training_allowed = not args.analysis_only and "test" not in split_lower and args.metric_cache_dir is not None
    metadata = {
        "split": args.split,
        "training_allowed": bool(training_allowed),
        "analysis_only": bool(args.analysis_only),
        "label_missing_count": label_missing_count,
        "all_labels_present": label_missing_count == 0,
        "base_config": str(args.base_config) if args.base_config else None,
        "base_checkpoint": str(args.base_checkpoint) if args.base_checkpoint else None,
        "bit_config": str(args.bit_config) if args.bit_config else None,
        "bit_checkpoint": str(args.bit_checkpoint) if args.bit_checkpoint else None,
        "base_eval_dir": str(args.base_eval_dir) if args.base_eval_dir else None,
        "bit_eval_dir": str(args.bit_eval_dir) if args.bit_eval_dir else None,
        "metric_cache_dir": str(args.metric_cache_dir) if args.metric_cache_dir else None,
        "chunk_cache_root": str(args.chunk_cache_root) if args.chunk_cache_root else None,
        "chunk_name_pattern": args.chunk_name_pattern,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "margin_pdm": args.margin_pdm,
        "require_no_nc_regression": args.require_no_nc_regression,
        "require_no_ttc_regression": args.require_no_ttc_regression,
        "require_dac_nonregression": args.require_dac_nonregression,
    }
    write_outputs(rows, args, metadata)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(rows), "training_allowed": training_allowed}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
