#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


A0_OFFICIAL_BASELINE_PDMS = 0.864891


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-K oracle diagnostic for Last-VLA teacher trajectory SFT.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--chunk-cache-root", type=Path)
    parser.add_argument("--chunk-name-pattern", default="train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*")
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def proxy_score(trajectory: torch.Tensor, gt: Optional[torch.Tensor] = None) -> float:
    traj = trajectory.detach().float()
    progress = traj[-1, 0] - traj[0, 0]
    lateral = traj[:, 1].abs().mean()
    heading = traj[:, 2].abs().mean()
    comfort = traj[1:, :2].diff(dim=0).norm(dim=-1).mean() if traj.shape[0] > 2 else traj.new_tensor(0.0)
    score = 0.10 * progress - 0.05 * lateral - 0.02 * heading - 0.01 * comfort
    if gt is not None:
        score = score - 0.03 * (traj - gt.detach().float()).abs().mean()
    return float(score.item())


def oracle_metrics(rows: List[Dict[str, Any]], *, k: int) -> Dict[str, Any]:
    deterministic = [float(row["deterministic_score"]) for row in rows]
    means = [sum(row["candidate_scores"]) / len(row["candidate_scores"]) for row in rows]
    best = [float(row["oracle_best_score"]) for row in rows]
    if not rows:
        return {
            "deterministic_PDMS": None,
            "stochastic_mean_PDMS": None,
            "oracle_best_of_K_PDMS": None,
            "oracle_delta_vs_deterministic": None,
            "oracle_delta_vs_A0_baseline": None,
            "candidate_score_mean": None,
            "candidate_score_std": None,
            "num_samples": 0,
            "K": int(k),
        }
    all_candidate_scores = torch.tensor([score for row in rows for score in row["candidate_scores"]], dtype=torch.float32)
    det_mean = float(torch.tensor(deterministic).mean().item())
    oracle_mean = float(torch.tensor(best).mean().item())
    return {
        "deterministic_PDMS": det_mean,
        "stochastic_mean_PDMS": float(torch.tensor(means).mean().item()),
        "oracle_best_of_K_PDMS": oracle_mean,
        "oracle_delta_vs_deterministic": oracle_mean - det_mean,
        "oracle_delta_vs_A0_baseline": oracle_mean - A0_OFFICIAL_BASELINE_PDMS,
        "candidate_score_mean": float(all_candidate_scores.mean().item()),
        "candidate_score_std": float(all_candidate_scores.std(unbiased=False).item()),
        "num_samples": len(rows),
        "K": int(k),
        "hard_gate_teacher_sft_plausible": (oracle_mean - det_mean) >= 0.01,
    }


def synthetic_rows(num_samples: int, k: int, seed: int) -> List[Dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    rows: List[Dict[str, Any]] = []
    for idx in range(num_samples):
        gt = torch.zeros(8, 3)
        base = torch.randn(8, 3, generator=generator) * 0.10
        candidates = []
        scores = []
        for candidate_idx in range(k):
            traj = base + torch.randn(8, 3, generator=generator) * (0.20 / (candidate_idx + 1))
            traj[:, 0] += torch.linspace(0, 1 + 0.1 * candidate_idx, 8)
            candidates.append(traj)
            scores.append(proxy_score(traj, gt))
        best_index = int(torch.tensor(scores).argmax().item())
        rows.append(
            {
                "sample_token": f"synthetic_{idx:04d}",
                "candidate_scores": scores,
                "best_index": best_index,
                "oracle_best_score": scores[best_index],
                "deterministic_score": scores[0],
            }
        )
    return rows


def _strip_train_only(action_input: Dict[str, Any]) -> Dict[str, Any]:
    blocked = ("target", "teacher_trajectory", "teacher_score", "gt_score", "oracle_best_of_k")
    return {key: value for key, value in action_input.items() if not any(token in key for token in blocked)}


def real_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from scripts import eval_recogdrive_expert_pdm as base
    from navsim.agents.recogdrive.expert_cache import load_sample

    if args.config is None or args.checkpoint is None or args.chunk_cache_root is None:
        raise ValueError("Real oracle mode requires --config, --checkpoint, and --chunk-cache-root.")
    cfg = base.load_yaml(args.config)
    planner = base.build_planner(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = base.dtype_from_precision(args.precision) if device.type == "cuda" else torch.float32
    planner = planner.to(device)
    if dtype != torch.float32:
        planner = planner.to(dtype=dtype)
    base.load_checkpoint(planner, args.checkpoint)
    planner.eval()
    paths = base.sample_paths(base.chunk_dirs(args), args.max_samples)
    rows: List[Dict[str, Any]] = []
    for sample_idx, (chunk_dir, sample_path, record) in enumerate(paths):
        sample = load_sample(sample_path)
        vl_features, action_input = base.make_batch(sample, planner, device, dtype)
        action_input = type(action_input)(data=_strip_train_only(dict(action_input)))
        generator = torch.Generator(device=device).manual_seed(int(args.sampling_seed) + sample_idx * 1009)
        scores: List[float] = []
        trajectories: List[torch.Tensor] = []
        with torch.no_grad():
            for candidate_idx in range(int(args.num_candidates)):
                init = torch.randn(
                    1,
                    planner.config.action_horizon,
                    planner.config.action_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                pred = planner.get_action(vl_features, action_input, init_actions=init, deterministic=False)["pred_traj"]
                traj = pred.detach().float().cpu().squeeze(0)
                trajectories.append(traj)
                gt = sample.get("trajectory")
                scores.append(proxy_score(traj, gt if isinstance(gt, torch.Tensor) else None))
        best_index = int(torch.tensor(scores).argmax().item())
        rows.append(
            {
                "sample_token": str(sample.get("sample_token") or record.get("sample_token") or sample_path.stem),
                "scene_token": str(sample.get("scene_token") or sample_path.stem),
                "chunk": chunk_dir.name,
                "candidate_scores": scores,
                "best_index": best_index,
                "oracle_best_score": scores[best_index],
                "deterministic_score": scores[0],
            }
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path, k: int) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = oracle_metrics(rows, k=k)
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    rows = (
        synthetic_rows(args.max_samples or 4, args.num_candidates, args.sampling_seed)
        if args.synthetic_smoke
        else real_rows(args)
    )
    metrics = write_outputs(rows, args.output_dir, args.num_candidates)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
