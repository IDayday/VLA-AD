from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.offline_action_explorer import build_structured_perturbations
from navsim.agents.recogdrive.offline_rl_buffer import REQUIRED_COMPONENT_KEYS, elite_record_exists, save_elite_record
from navsim.agents.recogdrive.recogdrive_diffusion_planner import OfflineRLConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache_fast
from navsim.planning.training.dataset import load_feature_target_from_pickle
from scripts.eval_recogdrive_expert_pdm import build_pdm_tools


PDM_TO_COMPONENT = {
    "score": "pdms",
    "no_at_fault_collisions": "no_at_fault_collisions",
    "drivable_area_compliance": "drivable_area_compliance",
    "ego_progress": "ego_progress",
    "time_to_collision_within_bound": "time_to_collision_within_bound",
    "comfort": "history_comfort",
    "driving_direction_compliance": "driving_direction_compliance",
}


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_gt(cache_root: Path, log_name: str, token: str) -> torch.Tensor:
    path = cache_root / log_name / token / "trajectory_target.gz"
    payload = load_feature_target_from_pickle(path)
    traj = payload.get("trajectory")
    if not isinstance(traj, torch.Tensor):
        raise KeyError(f"Missing tensor field 'trajectory' in {path}")
    if tuple(traj.shape) != (8, 3):
        raise ValueError(f"GT trajectory shape for token={token} is {tuple(traj.shape)}, expected [8, 3].")
    return traj.detach().float()


def _candidate_cfg(args: argparse.Namespace) -> OfflineRLConfig:
    cfg = OfflineRLConfig()
    if args.progress_endpoint_deltas_m:
        cfg.progress_endpoint_deltas_m = tuple(float(v) for v in args.progress_endpoint_deltas_m.split(","))
    if args.progress_speed_scales:
        cfg.progress_speed_scales = tuple(float(v) for v in args.progress_speed_scales.split(","))
    if args.progress_time_gammas:
        cfg.progress_time_gammas = tuple(float(v) for v in args.progress_time_gammas.split(","))
    if args.lateral_offsets_m:
        cfg.lateral_offsets_m = tuple(float(v) for v in args.lateral_offsets_m.split(","))
    if args.endpoint_lateral_offsets_m:
        cfg.endpoint_lateral_offsets_m = tuple(float(v) for v in args.endpoint_lateral_offsets_m.split(","))
    if args.timing_slow_first_scales:
        cfg.timing_slow_first_scales = tuple(float(v) for v in args.timing_slow_first_scales.split(","))
    if args.timing_delay_strengths:
        cfg.timing_delay_strengths = tuple(float(v) for v in args.timing_delay_strengths.split(","))
    cfg.clip_candidates_to_norm_range = bool(args.clip_candidates_to_norm_range)
    cfg.enforce_forward_monotonic_x = bool(args.enforce_forward_monotonic_x)
    cfg.use_final_heading_guard = bool(args.use_final_heading_guard)
    cfg.max_heading_step_rad = float(args.max_heading_step_rad)
    cfg.max_final_heading_delta_rad = float(args.max_final_heading_delta_rad)
    return cfg


def _build_candidates(gt: torch.Tensor, cfg: OfflineRLConfig) -> tuple[np.ndarray, List[str], np.ndarray]:
    anchor = gt.view(1, 1, 8, 3)
    perturb, perturb_sources, perturb_distance = build_structured_perturbations(anchor, ["gt"], cfg)
    candidates = torch.cat([anchor, perturb], dim=1)[0].contiguous().float()
    sources = ["gt", *perturb_sources]
    anchor_distance = torch.cat([gt.new_zeros(1, 1), perturb_distance], dim=1)[0].float()
    if candidates.shape[0] != len(sources) or anchor_distance.shape != (len(sources),):
        raise ValueError("Structured candidate source/distance shape mismatch.")
    return (
        candidates.detach().cpu().numpy().astype(np.float32),
        sources,
        anchor_distance.detach().cpu().numpy().astype(np.float32),
    )


def _score_components(metric_cache: Any, candidates: np.ndarray, tools: tuple[Any, Any, Any]) -> Dict[str, np.ndarray]:
    future_sampling, simulator, scorer = tools
    results = pdm_score_batch_same_cache_fast(
        metric_cache=metric_cache,
        model_trajectories=candidates,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    if len(results) != candidates.shape[0]:
        raise RuntimeError(f"PDM returned {len(results)} rows for {candidates.shape[0]} candidates.")
    components: Dict[str, np.ndarray] = {}
    for pdm_key, component_key in PDM_TO_COMPONENT.items():
        components[component_key] = np.asarray([float(getattr(item, pdm_key)) for item in results], dtype=np.float32)
    components["lane_keeping"] = components["drivable_area_compliance"].astype(np.float32, copy=True)
    components["traffic_light_compliance"] = np.ones_like(components["pdms"], dtype=np.float32)
    missing = sorted(set(REQUIRED_COMPONENT_KEYS).difference(components))
    if missing:
        raise KeyError(f"Missing component keys after PDM scoring: {missing}")
    return components


def _valid_mask(components: Dict[str, np.ndarray], gt_idx: int = 0) -> np.ndarray:
    nc = components["no_at_fault_collisions"]
    dac = components["drivable_area_compliance"]
    ddc = components["driving_direction_compliance"]
    gt_ddc = float(ddc[gt_idx])
    return (nc >= 1.0) & (dac >= 1.0) & ((ddc >= 0.99) | (ddc >= gt_ddc - 0.01))


def _make_record(
    token: str,
    candidates: np.ndarray,
    sources: Sequence[str],
    anchor_distance: np.ndarray,
    components: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    rewards = components["pdms"].astype(np.float32)
    valid_mask = _valid_mask(components)
    selection_score = rewards.astype(np.float32, copy=True)
    raw_idx = int(np.argmax(rewards)) if rewards.size else 0
    if bool(valid_mask.any()):
        valid_idx = int(np.flatnonzero(valid_mask)[np.argmax(rewards[valid_mask])])
    else:
        valid_idx = raw_idx
    selected_idx = raw_idx
    gt_reward = float(rewards[0])
    sources_l = [str(source) for source in sources]
    return {
        "token": str(token),
        "candidates": candidates.astype(np.float32),
        "rewards": rewards,
        "components": {key: np.asarray(value, dtype=np.float32) for key, value in components.items()},
        "sources": sources_l,
        "anchor_distance": anchor_distance.astype(np.float32),
        "valid_mask": valid_mask.astype(np.bool_),
        "selection_score": selection_score,
        "gt_reward": gt_reward,
        "il_reward": gt_reward,
        "best_reward": float(rewards[valid_idx]),
        "best_source": sources_l[valid_idx],
        "best_raw_reward": float(rewards[raw_idx]),
        "best_valid_reward": float(rewards[valid_idx]),
        "best_selected_reward": float(rewards[selected_idx]),
        "best_raw_source": sources_l[raw_idx],
        "best_valid_source": sources_l[valid_idx],
        "best_selected_source": sources_l[selected_idx],
        "has_valid_candidate": bool(valid_mask.any()),
        "version": 2,
    }


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = _load_rows(args.token_tsv)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")
    shard_rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.max_records is not None:
        shard_rows = shard_rows[: int(args.max_records)]

    args.output_buffer_root.mkdir(parents=True, exist_ok=True)
    cfg = _candidate_cfg(args)
    metric_loader = MetricCacheLoader(args.metric_cache_dir)
    tools = build_pdm_tools()

    stats: Counter[str] = Counter()
    best_delta_values: List[float] = []
    best_valid_delta_values: List[float] = []
    best_source_counts: Counter[str] = Counter()
    started = time.time()
    for local_idx, row in enumerate(shard_rows, start=1):
        token = str(row["token"])
        log_name = str(row["log_name"])
        if args.skip_done and elite_record_exists(args.output_buffer_root, token):
            stats["skipped_done"] += 1
            continue
        try:
            if token not in metric_loader.metric_cache_paths:
                raise FileNotFoundError(f"missing_metric_cache:{token}")
            gt = _load_gt(args.cache_root, log_name, token)
            candidates, sources, anchor_distance = _build_candidates(gt, cfg)
            components = _score_components(metric_loader.get_from_token(token), candidates, tools)
            record = _make_record(token, candidates, sources, anchor_distance, components)
            save_elite_record(args.output_buffer_root, token, record)
            rewards = np.asarray(record["rewards"], dtype=np.float32)
            valid_mask = np.asarray(record["valid_mask"], dtype=np.bool_)
            raw_idx = int(np.argmax(rewards))
            valid_idx = int(np.flatnonzero(valid_mask)[np.argmax(rewards[valid_mask])]) if bool(valid_mask.any()) else raw_idx
            best_delta_values.append(float(rewards[raw_idx] - rewards[0]))
            best_valid_delta_values.append(float(rewards[valid_idx] - rewards[0]))
            best_source_counts[str(record["best_valid_source"])] += 1
            stats["written"] += 1
            if float(rewards[valid_idx]) > float(rewards[0]) + 1e-6:
                stats["valid_better_than_gt"] += 1
            if bool(valid_mask.any()):
                stats["has_valid_candidate"] += 1
        except Exception as exc:
            stats[f"error:{type(exc).__name__}"] += 1
            if args.error_log:
                with args.error_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"token": token, "log_name": log_name, "error": repr(exc)}, sort_keys=True) + "\n")
        if args.progress_every > 0 and local_idx % args.progress_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "processed": local_idx,
                        "total": len(shard_rows),
                        "written": stats["written"],
                        "valid_better_than_gt": stats["valid_better_than_gt"],
                        "rate_tokens_per_s": local_idx / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    summary = {
        "token_tsv": str(args.token_tsv),
        "cache_root": str(args.cache_root),
        "metric_cache_dir": str(args.metric_cache_dir),
        "output_buffer_root": str(args.output_buffer_root),
        "num_input_rows": len(rows),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "num_shard_rows": len(shard_rows),
        "stats": dict(sorted(stats.items())),
        "best_source_hist": dict(sorted(best_source_counts.items())),
        "best_delta_mean": _mean(best_delta_values),
        "best_valid_delta_mean": _mean(best_valid_delta_values),
        "elapsed_s": time.time() - started,
        "candidate_config": {
            "progress_endpoint_deltas_m": list(cfg.progress_endpoint_deltas_m),
            "progress_speed_scales": list(cfg.progress_speed_scales),
            "progress_time_gammas": list(cfg.progress_time_gammas),
            "lateral_offsets_m": list(cfg.lateral_offsets_m),
            "endpoint_lateral_offsets_m": list(cfg.endpoint_lateral_offsets_m),
            "timing_slow_first_scales": list(cfg.timing_slow_first_scales),
            "timing_delay_strengths": list(cfg.timing_delay_strengths),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GT-only structured elite buffer records for Stage2 Pareto support supplementation."
    )
    parser.add_argument("--token-tsv", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metric-cache-dir", type=Path, required=True)
    parser.add_argument("--output-buffer-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--error-log", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--progress-endpoint-deltas-m", default="")
    parser.add_argument("--progress-speed-scales", default="")
    parser.add_argument("--progress-time-gammas", default="")
    parser.add_argument("--lateral-offsets-m", default="")
    parser.add_argument("--endpoint-lateral-offsets-m", default="")
    parser.add_argument("--timing-slow-first-scales", default="")
    parser.add_argument("--timing-delay-strengths", default="")
    parser.add_argument("--clip-candidates-to-norm-range", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-forward-monotonic-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-final-heading-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-heading-step-rad", type=float, default=0.25)
    parser.add_argument("--max-final-heading-delta-rad", type=float, default=0.4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
