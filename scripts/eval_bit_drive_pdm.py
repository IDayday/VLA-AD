#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import lzma
import pickle
import random
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
from navsim.agents.recogdrive.expert_cache import iter_index, load_sample  # noqa: E402
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.common.dataloader import MetricCacheLoader  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BiT-Drive on VLM-hidden NAVSIM chunks with left-tail reporting.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="navtest")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--chunk-name-pattern", default="navtest_full_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true", help="Override --deterministic and sample with denoising noise.")
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dtype_from_precision(precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_planner(cfg_dict: Dict[str, Any]) -> ReCogDriveDiffusionPlanner:
    cfg = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": 8,
            "head_dim": 48,
            "num_layers": int(cfg_dict.get("num_dit_layers", 16)),
            "output_dim": 512,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=int(cfg_dict.get("planner_dim", 384)),
        planner_dim=int(cfg_dict.get("planner_dim", 384)),
        hidden_size=1024,
        action_dim=int(cfg_dict.get("action_dim", 3)),
        action_horizon=int(cfg_dict.get("action_horizon", 8)),
        sampling_method=str(cfg_dict.get("sampling_method", "ddim")),
        num_inference_steps=int(cfg_dict.get("num_inference_steps", 5)),
        vlm_size="small",
    )
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.vlm_size = "small"
    cfg.allow_future_targets_in_inference = False
    return ReCogDriveDiffusionPlanner(cfg)


def checkpoint_candidates(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for suffix in (".ckpt", ".pth", ".pt", ".safetensors", ".bin"):
        files.extend(p for p in path.rglob(f"*{suffix}") if p.is_file())
    def score(item: Path) -> tuple[int, int, str]:
        name = item.name.lower()
        value = (1000 if "il" in name else 0) + (500 if "model" in name else 0) + (100 if item.suffix == ".safetensors" else 0)
        return value, item.stat().st_size, str(item)
    return sorted(set(files), key=score, reverse=True)


def load_state_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path), device="cpu"))
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"{path} did not contain a state dict.")
    return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}


def normalize_key(key: str) -> str:
    for prefix in ("agent.action_head.", "action_head."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_checkpoint(planner: ReCogDriveDiffusionPlanner, path: Path) -> Dict[str, Any]:
    model_state = planner.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    candidates = checkpoint_candidates(path)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    for file in candidates:
        for raw_key, value in load_state_file(file).items():
            key = normalize_key(raw_key)
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                loaded.setdefault(key, value)
    incompatible = planner.load_state_dict(loaded, strict=False)
    return {"loaded_key_count": len(loaded), "missing_key_count": len(incompatible.missing_keys), "checkpoint_files": [str(p) for p in candidates]}


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def chunk_dirs(args: argparse.Namespace) -> List[Path]:
    if args.chunk_cache_dir is not None:
        return [args.chunk_cache_dir]
    if args.chunk_cache_root is None:
        raise ValueError("Set --chunk-cache-dir or --chunk-cache-root for BiT chunk evaluation.")
    dirs = sorted(path for path in args.chunk_cache_root.glob(args.chunk_name_pattern) if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No chunks matching {args.chunk_name_pattern!r} under {args.chunk_cache_root}")
    return dirs


def sample_paths(chunks: List[Path], max_samples: Optional[int]) -> List[Tuple[Path, Path, Dict[str, Any]]]:
    samples = []
    for chunk in chunks:
        for record in iter_index(chunk):
            samples.append((chunk, resolve_index_path(chunk, record["path"]), record))
            if max_samples is not None and len(samples) >= max_samples:
                return samples
    return samples


class ScannedMetricCacheLoader:
    def __init__(self, cache_path: Path) -> None:
        self.metric_cache_paths = {path.parent.name: path for path in cache_path.rglob("metric_cache.pkl") if path.is_file()}
        if not self.metric_cache_paths:
            raise FileNotFoundError(f"No metric_cache.pkl files found under {cache_path}")

    def get_from_token(self, token: str):
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            return pickle.load(f)


def build_metric_cache_loader(cache_path: Path):
    try:
        metadata_loader = MetricCacheLoader(cache_path)
    except Exception as exc:
        warnings.warn(f"MetricCacheLoader failed for {cache_path}: {exc!r}; scanning recursively.", RuntimeWarning)
        return ScannedMetricCacheLoader(cache_path)
    try:
        scanned = ScannedMetricCacheLoader(cache_path)
    except Exception:
        return metadata_loader
    if len(scanned.metric_cache_paths) > len(getattr(metadata_loader, "metric_cache_paths", {}) or {}):
        return scanned
    return metadata_loader


def build_pdm_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_weight=0.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-3,
            progress_distance_threshold=5.0,
        ),
    )
    return proposal_sampling, simulator, scorer


def make_batch(sample: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, BatchFeature]:
    vl_features = sample["last_hidden_state"].float().unsqueeze(0).to(device=device, dtype=dtype)
    his = sample["history_trajectory"].float().view(1, -1).to(device=device, dtype=dtype)
    status = sample["status_feature"].float().view(1, -1).to(device=device, dtype=dtype)
    data: Dict[str, torch.Tensor] = {"his_traj": his, "status_feature": status}
    if "trajectory" in sample:
        data["action"] = torch.full_like(sample["trajectory"].float().unsqueeze(0).to(device=device, dtype=dtype), float("nan"))
    return vl_features, BatchFeature(data=data)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def median(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def zero_count(values: List[float]) -> int:
    return int(sum(1 for value in values if value <= 1e-9))


def left_tail_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("valid")]
    values = {key: [float(row[key]) for row in valid if row.get(key) is not None] for key in (
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
    )}
    scores = values["score"]
    n = len(scores)
    hist_counts: List[int] = []
    hist_bins: List[float] = []
    if scores:
        counts, bins = np.histogram(np.asarray(scores, dtype=np.float32), bins=20, range=(0.0, 1.0))
        hist_counts = [int(x) for x in counts.tolist()]
        hist_bins = [float(x) for x in bins.tolist()]
    return {
        "num_samples": len(rows),
        "num_pdm_valid": len(valid),
        "mean_pdms": mean(scores),
        "median_pdms": median(scores),
        "p5_pdms": percentile(scores, 5),
        "p10_pdms": percentile(scores, 10),
        "zero_score_count": zero_count(scores),
        "zero_score_rate": zero_count(scores) / n if n else None,
        "no_at_fault_collision_zero_count": zero_count(values["no_at_fault_collisions"]),
        "no_at_fault_collision_zero_rate": zero_count(values["no_at_fault_collisions"]) / n if n else None,
        "drivable_area_compliance_zero_count": zero_count(values["drivable_area_compliance"]),
        "drivable_area_compliance_zero_rate": zero_count(values["drivable_area_compliance"]) / n if n else None,
        "time_to_collision_zero_count": zero_count(values["time_to_collision_within_bound"]),
        "time_to_collision_zero_rate": zero_count(values["time_to_collision_within_bound"]) / n if n else None,
        "ego_progress_mean": mean(values["ego_progress"]),
        "ego_progress_median": median(values["ego_progress"]),
        "comfort_mean": mean(values["comfort"]),
        "comfort_median": median(values["comfort"]),
        "histogram": {"bins": hist_bins, "counts": hist_counts},
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "sample_token",
        "scene_token",
        "chunk",
        "valid",
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "comfort",
        "trajectory_l1",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> int:
    args = parse_args()
    if args.stochastic:
        args.deterministic = False
    seed_everything(args.seed)
    chunks = chunk_dirs(args)
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_precision(args.precision, device)
    planner = build_planner(cfg).to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    load_report = load_checkpoint(planner, args.checkpoint)
    planner.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_cache_loader = build_metric_cache_loader(args.metric_cache_dir) if args.metric_cache_dir is not None else None
    pdm_tools = build_pdm_tools() if metric_cache_loader is not None else None

    predictions: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    missing_metric_cache = 0
    failed_pdm = 0
    for chunk, path, index_record in sample_paths(chunks, args.max_samples):
        sample = load_sample(path)
        vl_features, action_input = make_batch(sample, device, dtype)
        with torch.no_grad():
            output = planner.get_action(vl_features, action_input, deterministic=args.deterministic)
        pred = output["pred_traj"].detach().float().cpu().squeeze(0)
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"Non-finite prediction for {path}")
        sample_token = str(sample.get("sample_token", index_record.get("sample_token", path.stem)))
        scene_token = str(sample.get("scene_token", index_record.get("scene_token", path.stem)))
        pred_record = {
            "path": str(path),
            "chunk": str(chunk),
            "scene_token": scene_token,
            "sample_token": sample_token,
            "pred_traj": pred.tolist(),
        }
        if "bit_terminal_pred" in output:
            pred_record["bit_terminal_pred"] = output["bit_terminal_pred"].detach().float().cpu().squeeze(0).tolist()
        if "bit_path_anchor_pred" in output:
            pred_record["bit_path_anchor_pred"] = output["bit_path_anchor_pred"].detach().float().cpu().squeeze(0).tolist()
        row: Dict[str, Any] = {"sample_token": sample_token, "scene_token": scene_token, "chunk": chunk.name, "valid": None}
        if "trajectory" in sample:
            row["trajectory_l1"] = float(torch.nn.functional.l1_loss(pred, sample["trajectory"].float()).item())
        if metric_cache_loader is not None and pdm_tools is not None:
            if sample_token not in metric_cache_loader.metric_cache_paths:
                missing_metric_cache += 1
                row.update({"valid": False, "error": "missing_metric_cache"})
            else:
                try:
                    metric_cache = metric_cache_loader.get_from_token(sample_token)
                    future_sampling, simulator, scorer = pdm_tools
                    result = pdm_score(
                        metric_cache=metric_cache,
                        model_trajectory=Trajectory(poses=pred.numpy()),
                        future_sampling=future_sampling,
                        simulator=simulator,
                        scorer=scorer,
                    )
                    result_dict = asdict(result)
                    row.update({"valid": True, **result_dict})
                    pred_record["pdm"] = result_dict
                except Exception as exc:
                    failed_pdm += 1
                    row.update({"valid": False, "error": repr(exc)})
        rows.append(row)
        predictions.append(pred_record)

    aggregate = left_tail_metrics(rows)
    aggregate.update({
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "chunk_cache_dirs": [str(chunk) for chunk in chunks],
        "metric_cache_dir": str(args.metric_cache_dir) if args.metric_cache_dir else None,
        "target_gt_disabled_in_eval": True,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "load_report": load_report,
        "num_pdm_missing_metric_cache": missing_metric_cache,
        "num_pdm_failed": failed_pdm,
    })
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in predictions:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")
    with (args.output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    write_csv(args.output_dir / "per_sample_metrics.csv", rows)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "histogram.json").write_text(json.dumps(aggregate["histogram"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# BiT-Drive Evaluation Report",
        "",
        f"Config: `{args.config}`",
        f"Checkpoint: `{args.checkpoint}`",
        f"Split: {args.split}",
        f"Samples: {aggregate['num_samples']}",
        f"PDM valid: {aggregate['num_pdm_valid']}",
        f"Mean PDMS: {aggregate['mean_pdms']}",
        f"P5/P10 PDMS: {aggregate['p5_pdms']} / {aggregate['p10_pdms']}",
        f"Zero-score count: {aggregate['zero_score_count']}",
        f"DAC zero count: {aggregate['drivable_area_compliance_zero_count']}",
        f"NC zero count: {aggregate['no_at_fault_collision_zero_count']}",
        f"TTC zero count: {aggregate['time_to_collision_zero_count']}",
        "",
        "Ground-truth future terminal/path targets are disabled in evaluation; if an action target is present, it is passed as NaN for leakage checking.",
    ]
    (args.output_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
