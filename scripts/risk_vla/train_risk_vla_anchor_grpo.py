#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.risk_vla.anchor_grpo_rewards import supervised_gates_pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare anchor-GRPO only after supervised RISK-VLA v3 gates pass.")
    parser.add_argument("--gate-summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-grpo", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    gates = json.loads(args.gate_summary_json.read_text(encoding="utf-8"))
    passed = supervised_gates_pass(gates)
    summary = {
        "gate_summary_json": str(args.gate_summary_json),
        "supervised_gates_pass": passed,
        "enable_grpo_requested": bool(args.enable_grpo),
        "will_train": bool(args.enable_grpo and passed and not args.dry_run),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "anchor_grpo_gate_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "anchor_grpo_gate_report.md").write_text(
        "\n".join(
            [
                "# Anchor-GRPO Gate Report",
                "",
                f"Supervised gates pass: `{passed}`",
                f"GRPO explicitly enabled: `{args.enable_grpo}`",
                f"Will train: `{summary['will_train']}`",
                "",
                "GRPO remains disabled unless `--enable-grpo` is set and supervised gates pass on >=10k held-out validation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if args.enable_grpo and passed and not args.dry_run:
        raise RuntimeError("GRPO training loop is intentionally not launched by default in this branch; wire policy trainer only after gate approval.")
    return 0 if passed or not args.enable_grpo else 1


if __name__ == "__main__":
    raise SystemExit(main())
