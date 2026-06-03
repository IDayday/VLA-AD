#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.last_vla_v2.pdm_scoring_utils import TrajectoryScorer, json_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Last-VLA teacher trajectory cache from best-of-K candidates.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-cache-root", type=Path, required=True)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--deterministic-base", type=str, default="false")
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--score-mode", choices=("pdm", "proxy"), default="pdm")
    parser.add_argument("--only-save-if-better", action="store_true")
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--save-candidates", action="store_true")
    return parser.parse_args()


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _strip_train_only(action_input: Dict[str, Any]) -> Dict[str, Any]:
    blocked = ("target", "teacher_trajectory", "teacher_score", "gt_score", "oracle_best_of_k")
    return {key: value for key, value in action_input.items() if not any(token in key for token in blocked)}


def build_teacher_payload(
    *,
    sample_token: str,
    scene_token: str,
    candidates: List[torch.Tensor],
    candidate_scores: List[float],
    candidate_components: List[Dict[str, Any]],
    best_index: int,
    gt_score: float,
    gt_components: Optional[Dict[str, Any]],
    score_mode: str,
    num_candidates: int,
    save_candidates: bool,
) -> Dict[str, Any]:
    best_score = float(candidate_scores[best_index])
    payload: Dict[str, Any] = {
        "sample_token": sample_token,
        "scene_token": scene_token,
        "teacher_trajectory": candidates[best_index],
        "teacher_score": torch.tensor(best_score, dtype=torch.float32),
        "gt_score": torch.tensor(float(gt_score), dtype=torch.float32),
        "oracle_best_of_k_score": torch.tensor(best_score, dtype=torch.float32),
        "candidate_scores": torch.tensor(candidate_scores, dtype=torch.float32),
        "candidate_count": torch.tensor(int(num_candidates), dtype=torch.int64),
        "teacher_source": "best_of_k_proxy" if score_mode == "proxy" else "best_of_k_pdm",
        "score_mode": score_mode,
    }
    if score_mode == "pdm":
        payload["pdm_components"] = candidate_components[best_index]
        payload["gt_pdm_components"] = gt_components
        payload["candidate_pdm_components"] = candidate_components
    if save_candidates:
        payload["candidate_trajectories"] = torch.stack(candidates, dim=0)
    return payload


def main() -> int:
    from scripts import eval_recogdrive_expert_pdm as base
    from navsim.agents.recogdrive.expert_cache import atomic_torch_save, iter_index, load_sample, write_json

    args = parse_args()
    if args.score_mode == "pdm" and args.metric_cache_dir is None:
        raise ValueError("--score-mode pdm requires --metric-cache-dir.")
    cfg = base.load_yaml(args.config)
    scorer = TrajectoryScorer(args.score_mode, args.metric_cache_dir)
    planner = base.build_planner(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = planner.to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    base.load_checkpoint(planner, args.checkpoint)
    planner.eval()

    chunks = base.chunk_dirs(args)
    sample_records = base.sample_paths(chunks, args.max_samples)
    samples_dir = args.output_cache_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    index_records: List[Dict[str, Any]] = []
    saved = skipped = 0
    deterministic = _as_bool(args.deterministic_base)

    for sample_idx, (chunk_dir, sample_path, record) in enumerate(sample_records):
        sample = load_sample(sample_path)
        vl_features, action_input = base.make_batch(sample, planner, device, dtype)
        action_input = type(action_input)(data=_strip_train_only(dict(action_input)))
        generator = torch.Generator(device=device).manual_seed(int(args.sampling_seed) + sample_idx * 1009)
        gt = sample.get("trajectory")
        sample_token = str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem)
        scene_token = str(sample.get("scene_token") or sample_path.stem)
        gt_components = scorer.score(sample, sample_token, gt.float()) if isinstance(gt, torch.Tensor) else None
        gt_score = float(gt_components["score"]) if gt_components is not None else 0.0
        candidate_scores: List[float] = []
        candidate_components: List[Dict[str, Any]] = []
        candidates: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(int(args.num_candidates)):
                init = torch.zeros(
                    1,
                    planner.config.action_horizon,
                    planner.config.action_dim,
                    device=device,
                    dtype=dtype,
                ) if deterministic else torch.randn(
                    1,
                    planner.config.action_horizon,
                    planner.config.action_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                pred = planner.get_action(vl_features, action_input, init_actions=init, deterministic=deterministic)["pred_traj"]
                traj = pred.detach().float().cpu().squeeze(0)
                candidates.append(traj)
                score_dict = scorer.score(sample, sample_token, traj)
                candidate_components.append(score_dict)
                candidate_scores.append(float(score_dict["score"]))
        best_index = int(torch.tensor(candidate_scores).argmax().item())
        best_score = float(candidate_scores[best_index])
        if args.only_save_if_better and best_score < gt_score + float(args.min_score_margin):
            skipped += 1
            continue
        out_path = samples_dir / f"{sample_token}.pt"
        payload = build_teacher_payload(
            sample_token=sample_token,
            scene_token=scene_token,
            candidates=candidates,
            candidate_scores=candidate_scores,
            candidate_components=candidate_components,
            best_index=best_index,
            gt_score=gt_score,
            gt_components=gt_components,
            score_mode=args.score_mode,
            num_candidates=int(args.num_candidates),
            save_candidates=bool(args.save_candidates),
        )
        atomic_torch_save(payload, out_path)
        index_records.append({"sample_token": sample_token, "scene_token": scene_token, "path": str(out_path.relative_to(args.output_cache_root))})
        saved += 1

    args.output_cache_root.mkdir(parents=True, exist_ok=True)
    with (args.output_cache_root / "index.jsonl").open("w", encoding="utf-8") as f:
        for item in index_records:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    write_json(
        args.output_cache_root / "metadata.json",
        {
            "version": "last_vla_teacher_trajectory_cache_v1",
            "num_candidates": int(args.num_candidates),
            "num_saved": saved,
            "num_skipped": skipped,
            "score_mode": args.score_mode,
            "pdm_scoring_active": scorer.pdm_scoring_active,
            "proxy_scoring_active": scorer.proxy_scoring_active,
            "save_candidates": bool(args.save_candidates),
            "source_checkpoint": str(args.checkpoint),
            "target_tokens_are_train_only": True,
        },
    )
    print(json.dumps({"saved": saved, "skipped": skipped, "output_cache_root": str(args.output_cache_root)}, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
