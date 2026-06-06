#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.trajectory_risk_critic import TrajectoryRiskCritic
from scripts.risk_vla.train_trajectory_critic_from_assets import CandidateAssetDataset, collate, evaluate


def write_eval_outputs(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "pdm_selection_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["selection", "pdms", "zero_score", "dac0", "nc0", "ttc0", "progress", "comfort"])
        writer.writeheader()
        for key in ("base_metrics", "selected_metrics", "oracle_metrics"):
            row = dict(summary["metrics"][key])
            row["selection"] = key.replace("_metrics", "")
            writer.writerow(row)
    lines = [
        "# Trajectory Risk Critic Evaluation Report",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Samples: `{summary['num_samples']}`",
        f"Split purpose: `{summary['purpose']}`",
        "",
        "| Selection | PDMS | Zero | DAC0 | NC0 | TTC0 | Progress | Comfort |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("base_metrics", "selected_metrics", "oracle_metrics"):
        item = summary["metrics"][key]
        lines.append(
            f"| {key.replace('_metrics', '')} | {item['pdms']:.6f} | {item['zero_score']} | {item['dac0']} | "
            f"{item['nc0']} | {item['ttc0']} | {item['progress']:.6f} | {item['comfort']:.6f} |"
        )
    lines.extend(["", "Risk metrics:"])
    for key, value in sorted(summary["metrics"]["risk_metrics"].items()):
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "trajectory_risk_critic_eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trajectory-conditioned RISK-VLA critic on candidate utility labels.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="analysis")
    parser.add_argument("--planner-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data = np.load(args.candidate_npz, allow_pickle=False)
    num_tokens = int(len(data["sample_tokens"]))
    if args.dry_run:
        print(json.dumps({"num_tokens": num_tokens, "checkpoint": str(args.checkpoint), "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    if args.min_real_samples and num_tokens < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(
            f"Refusing formal critic evaluation with {num_tokens} samples; minimum is {args.min_real_samples}."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    model = TrajectoryRiskCritic(
        planner_dim=int(model_args.get("planner_dim", args.planner_dim)),
        hidden_dim=int(model_args.get("hidden_dim", args.hidden_dim)),
        num_risk_classes=6,
        horizon=int(data["trajectories"].shape[2]),
        action_dim=int(data["trajectories"].shape[3]),
        num_submetrics=6,
    )
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = CandidateAssetDataset(
        candidate_npz=args.candidate_npz,
        labels_jsonl=args.labels_jsonl,
        indices=list(range(num_tokens)),
        purpose=args.purpose,
        low_score_threshold=0.5,
        progress_threshold=0.5,
        comfort_threshold=0.5,
        zero_eps=1e-9,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    metrics = evaluate(model, loader, device, model.planner_dim, {})
    summary = {
        "checkpoint": str(args.checkpoint),
        "candidate_npz": str(args.candidate_npz),
        "labels_jsonl": str(args.labels_jsonl),
        "num_samples": num_tokens,
        "purpose": args.purpose,
        "device": str(device),
        "metrics": metrics,
    }
    write_eval_outputs(summary, args.output_dir)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "num_samples": num_tokens}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
