#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.pareto_support.bpdas import compute_pdas_metrics
from navsim.agents.recogdrive.pareto_support.io import load_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build B-PDAS scene weights from support archives.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--metric_cache_path", default="")
    parser.add_argument("--split", default="navtrain")
    parser.add_argument("--output_weights_jsonl", required=True)
    parser.add_argument("--sample_count", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output_weights_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        archive_root = Path(args.archive_dir)
        if (archive_root / "full_archive").is_dir():
            archive_root = archive_root / "full_archive"
        for path in sorted(archive_root.glob("*.pkl.xz")):
            archive = load_archive(path)
            pool = archive.support_set[: args.sample_count]
            if not pool:
                continue
            rewards = torch.tensor([[float((c.true_metrics or {}).get("utility", (c.true_metrics or {}).get("pdms", 0.0))) for c in pool]], dtype=torch.float32)
            trajs = torch.stack([torch.as_tensor(c.trajectory, dtype=torch.float32) for c in pool], dim=0).unsqueeze(0)
            components = {
                "pdms": torch.tensor([[float((c.true_metrics or {}).get("pdms", 0.0)) for c in pool]], dtype=torch.float32),
                "ego_progress": torch.tensor([[float((c.true_metrics or {}).get("ego_progress", 0.0)) for c in pool]], dtype=torch.float32),
                "driving_direction_compliance": torch.tensor([[float((c.true_metrics or {}).get("driving_direction_compliance", 0.0)) for c in pool]], dtype=torch.float32),
                "feas_cost": torch.tensor([[float((c.true_metrics or {}).get("feas_cost", 0.0)) for c in pool]], dtype=torch.float32),
            }
            ref_metrics = archive.reference.get("ref_metrics", {})
            ref = {
                "reward": torch.tensor([float(ref_metrics.get("utility", ref_metrics.get("pdms", 0.0)))], dtype=torch.float32),
                "pdms": torch.tensor([float(ref_metrics.get("pdms", 0.0))], dtype=torch.float32),
                "ego_progress": torch.tensor([float(ref_metrics.get("ego_progress", 0.0))], dtype=torch.float32),
                "driving_direction_compliance": torch.tensor([float(ref_metrics.get("driving_direction_compliance", 0.0))], dtype=torch.float32),
            }
            metrics = compute_pdas_metrics(rewards, components, trajs, ref)
            row = {
                "scene_token": archive.scene_token,
                "weight": float(metrics.weight[0]),
                "p_success": float(metrics.success_rate[0]),
                "reward_std": float(metrics.reward_std[0]),
                "best_of_n_gap": float(metrics.bon_gap[0]),
                "bucket_diversity": float(metrics.bucket_diversity[0]),
                "regression_risk": float(metrics.regression_risk[0]),
                "mean_reward": float(rewards.mean()),
                "max_reward": float(rewards.max()),
                "ref_reward": float(ref["reward"][0]),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print({"output_weights_jsonl": str(output)})


if __name__ == "__main__":
    main()
