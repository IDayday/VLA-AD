from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build optional VLM risk-grounding instruction data from train/val labels.")
    parser.add_argument("--finegrained-labels-jsonl", type=Path, required=True)
    parser.add_argument("--strategy-utility-labels-jsonl", type=Path, default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if "test" in args.split.lower() or "navtest" in args.split.lower():
        raise RuntimeError("VLM risk instruction training data cannot be built from test/navtest labels")
    rows = 0
    if args.finegrained_labels_jsonl.is_file():
        with args.finegrained_labels_jsonl.open("r", encoding="utf-8") as handle:
            rows = sum(1 for line in handle if line.strip())
    summary = {
        "finegrained_labels_jsonl": str(args.finegrained_labels_jsonl),
        "strategy_utility_labels_jsonl": str(args.strategy_utility_labels_jsonl) if args.strategy_utility_labels_jsonl else None,
        "split": args.split,
        "rows": min(rows, args.max_samples) if args.max_samples else rows,
        "dry_run": bool(args.dry_run),
        "objective": ["risk_classes", "risk_time_bin", "strategy_recommendation", "template_rationale"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "vlm_risk_instruction_data_plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
