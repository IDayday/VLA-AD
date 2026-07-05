#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from dataclasses import asdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.mining_pipeline import mine_cache


def _pdm_row_to_dict(result: Any) -> dict[str, float]:
    raw = asdict(result) if hasattr(result, "__dataclass_fields__") else dict(result)
    out = {}
    for key in (
        "pdms",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "history_comfort",
        "lane_keeping",
        "driving_direction_compliance",
        "traffic_light_compliance",
    ):
        if key == "pdms":
            out[key] = float(raw.get("pdms", raw.get("score", raw.get("pdm_score", 0.0))))
        else:
            out[key] = float(raw.get(key, 0.0))
    return out


def _build_metric_context(metric_cache_path: str, *, use_exact_array_conversion: bool, chunk_size: int):
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    cache_loader = MetricCacheLoader(Path(metric_cache_path))
    proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling)
    scorer = PDMScorer(proposal_sampling, PDMScorerConfig(progress_weight=10.0, ttc_weight=5.0, comfortable_weight=2.0))

    def evaluate(trajs: np.ndarray, tokens: list[str], cfg: Any) -> list[dict[str, float]]:
        rows: list[dict[str, float] | None] = [None] * len(tokens)
        token_to_indices: dict[str, list[int]] = {}
        for idx, token in enumerate(tokens):
            token_to_indices.setdefault(str(token), []).append(idx)
        for token, indices in token_to_indices.items():
            metric_cache = cache_loader.get_from_token(token)
            step = int(chunk_size) if int(chunk_size) > 0 else len(indices)
            for start in range(0, len(indices), step):
                chunk = indices[start : start + step]
                pdm_rows = pdm_score_batch_same_cache(
                    metric_cache=metric_cache,
                    model_trajectories=trajs[chunk],
                    future_sampling=proposal_sampling,
                    simulator=simulator,
                    scorer=scorer,
                    use_exact_array_conversion=bool(use_exact_array_conversion),
                )
                if len(pdm_rows) != len(chunk):
                    raise RuntimeError(f"PDM returned {len(pdm_rows)} rows for {len(chunk)} candidates.")
                for local_idx, result in enumerate(pdm_rows):
                    rows[chunk[local_idx]] = _pdm_row_to_dict(result)
        if any(row is None for row in rows):
            raise RuntimeError("PDM metric context did not produce all candidate rows.")
        return [row for row in rows if row is not None]

    return evaluate


def _load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if cfg_path.suffix in {".yaml", ".yml"}:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine evaluator-verified SG-FPS Pareto support archives.")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--metric_cache_path", default="")
    parser.add_argument("--input_archive_dir", default="")
    parser.add_argument("--output_archive_dir", required=True)
    parser.add_argument("--round", choices=["0", "1", "2", "all"], default="all")
    parser.add_argument("--num_workers", type=int, default=1, help="Reserved for job launcher compatibility; use rank/world_size for sharding.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_scenes", type=int, default=0)
    parser.add_argument("--recogdrive_stage3_checkpoint", default="")
    parser.add_argument("--recogdrive_stage3_precomputed_dir", default="")
    parser.add_argument("--ddv2_precomputed_dir", default="")
    parser.add_argument("--driveor_precomputed_dir", default="")
    parser.add_argument("--scorer_ckpt", default="")
    parser.add_argument("--support_top_m", type=int, default=12)
    parser.add_argument("--true_eval_budget", type=int, default=16)
    parser.add_argument("--missing_external_policy", choices=["error", "skip", "fallback_gt_il"], default="skip")
    parser.add_argument("--allow_unverified_smoke", action="store_true", help="Allow synthetic smoke archives without true evaluator components. Do not use for experiments.")
    parser.add_argument("--use_exact_array_pdm_state_conversion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pdm_batch_chunk_size", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg.update(
        {
            "split": args.split,
            "support_top_m": args.support_top_m,
            "true_eval_budget": args.true_eval_budget,
            "gt_enabled": True,
            "recogdrive_stage3_enabled": bool(args.recogdrive_stage3_checkpoint or args.recogdrive_stage3_precomputed_dir),
            "recogdrive_stage3_checkpoint_path": args.recogdrive_stage3_checkpoint,
            "recogdrive_stage3_precomputed_dir": args.recogdrive_stage3_precomputed_dir,
            "ddv2_enabled": bool(args.ddv2_precomputed_dir),
            "ddv2_mode": "precomputed",
            "ddv2_precomputed_dir": args.ddv2_precomputed_dir,
            "driveor_enabled": bool(args.driveor_precomputed_dir),
            "driveor_mode": "precomputed",
            "driveor_precomputed_dir": args.driveor_precomputed_dir,
            "missing_external_policy": args.missing_external_policy,
            "allow_unverified_smoke": bool(args.allow_unverified_smoke),
        }
    )
    metric_context = None
    if args.metric_cache_path:
        cfg["metric_cache_path"] = args.metric_cache_path
        metric_context = _build_metric_context(
            args.metric_cache_path,
            use_exact_array_conversion=bool(args.use_exact_array_pdm_state_conversion),
            chunk_size=int(args.pdm_batch_chunk_size),
        )
    summary = mine_cache(
        cache_path=args.cache_path,
        output_archive_dir=args.output_archive_dir,
        cfg=cfg,
        metric_context=metric_context,
        scorer=None,
        round_id=args.round,
        limit_scenes=args.limit_scenes,
        rank=args.rank,
        world_size=args.world_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
