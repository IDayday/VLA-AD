#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.utility_router import RiskVLAv2UtilityRouter
from scripts.risk_vla.train_utility_router import UtilityRouterLabelDataset, build_strategy_map, collate, evaluate, load_groups


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a learned RISK-VLA utility router on held-out utility labels.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="analysis")
    parser.add_argument("--planner-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-real-samples", type=int, default=0)
    parser.add_argument("--debug-allow-small", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    groups = load_groups(args.labels_jsonl, args.purpose)
    if args.dry_run:
        print(json.dumps({"tokens": len(groups), "checkpoint": str(args.checkpoint), "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    if args.min_real_samples and len(groups) < args.min_real_samples and not args.debug_allow_small:
        raise RuntimeError(f"Refusing formal router evaluation with {len(groups)} tokens; minimum is {args.min_real_samples}.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    strategy_map = checkpoint.get("strategy_map") or build_strategy_map(groups)
    model_args = checkpoint.get("args", {})
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RiskVLAv2UtilityRouter(
        planner_dim=int(model_args.get("planner_dim", args.planner_dim)),
        num_strategies=max(1, len(strategy_map)),
        hidden_dim=int(model_args.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    tokens = sorted(groups)
    loader = DataLoader(UtilityRouterLabelDataset(groups, tokens, strategy_map, model.planner_dim), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    metrics = evaluate(model, loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "labels_jsonl": str(args.labels_jsonl),
        "num_samples": len(tokens),
        "purpose": args.purpose,
        "strategy_map": strategy_map,
        "metrics": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "router_eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["num_samples", "loss", "accuracy"])
        writer.writeheader()
        writer.writerow({"num_samples": len(tokens), "loss": metrics["loss"], "accuracy": metrics["accuracy"]})
    (args.output_dir / "utility_router_eval_report.md").write_text(
        "\n".join(
            [
                "# Utility Router Evaluation Report",
                "",
                f"Checkpoint: `{args.checkpoint}`",
                f"Samples: `{len(tokens)}`",
                f"Accuracy: `{metrics['accuracy']:.6f}`",
                f"Loss: `{metrics['loss']:.6f}`",
                "",
                "Leakage guard: use `purpose=analysis` for navtest reports only; training mode rejects navtest/test split names.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "num_samples": len(tokens)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
