from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


STAGES = [
    "discover_inputs",
    "build_labels",
    "build_candidate_bank",
    "train_critic",
    "train_router",
    "eval_small",
    "eval_full_if_ready",
    "build_report",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RISK-VLA v2 Round2 orchestrator.")
    parser.add_argument("--exp-root", type=Path, default=Path("experiments/risk_vla/v2"))
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.exp_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "method": "RISK-VLA v2: Risk-Conditioned Multi-Candidate Strategy Planning for Vision-Language-Action Driving",
        "exp_root": str(args.exp_root),
        "work_root": str(args.work_root) if args.work_root else None,
        "max_samples": args.max_samples,
        "dry_run": bool(args.dry_run),
        "stages": STAGES,
        "guardrails": [
            "no navtest/test labels for training",
            "small pilots before full train/val/navtest",
            "GRPO disabled until supervised safety gates pass",
            "shared caches are read-only inputs",
        ],
    }
    (args.exp_root / "round2_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RISK-VLA v2 Round2 Dry-Run Report" if args.dry_run else "# RISK-VLA v2 Round2 Plan",
        "",
        f"Method: {plan['method']}",
        f"Max samples: `{args.max_samples}`",
        "",
        "## Stages",
    ]
    lines.extend([f"{idx + 1}. {stage}" for idx, stage in enumerate(STAGES)])
    lines.extend(["", "## Guardrails"])
    lines.extend([f"- {item}" for item in plan["guardrails"]])
    report_name = "ROUND2_DRYRUN_REPORT.md" if args.dry_run else "ROUND2_PLAN_REPORT.md"
    (args.exp_root / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
