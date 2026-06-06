from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan/evaluate optional VLM risk grounding with corruption checks.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--val-labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_dir": str(args.model_dir),
        "model_exists": args.model_dir.is_dir(),
        "val_labels_jsonl": str(args.val_labels_jsonl),
        "val_labels_exist": args.val_labels_jsonl.is_file(),
        "corruption_ablation_required": True,
        "dry_run": bool(args.dry_run),
    }
    (args.output_dir / "vlm_risk_grounding_eval_plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if args.dry_run or (summary["model_exists"] and summary["val_labels_exist"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
