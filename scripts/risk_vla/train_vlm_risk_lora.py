from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan optional VLM risk LoRA training. Disabled by default.")
    parser.add_argument("--instruction-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "instruction_data_dir": str(args.instruction_data_dir),
        "instruction_plan_exists": (args.instruction_data_dir / "vlm_risk_instruction_data_plan.json").is_file(),
        "planner_frozen": True,
        "enable_train": bool(args.enable_train),
        "dry_run": bool(args.dry_run),
    }
    if not args.enable_train:
        summary["status"] = "not_started_disabled_by_default"
    else:
        summary["status"] = "not_implemented_requires_explicit_training_backend"
    (args.output_dir / "vlm_risk_lora_train_plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not args.enable_train else 1


if __name__ == "__main__":
    raise SystemExit(main())
