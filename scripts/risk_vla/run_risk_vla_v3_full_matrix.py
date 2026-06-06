#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MATRIX = [
    "A0 Base",
    "B3/BiT direct",
    "D5 conservative BiT",
    "RISK-VLA v1",
    "RISK-VLA v3 heuristic router",
    "RISK-VLA v3 learned router",
    "RISK-VLA v3 scorer_select K=8",
    "RISK-VLA v3 scorer_select K=16",
    "RISK-VLA v3 safealign SFT",
    "RISK-VLA v3 safealign SFT + CVaR",
    "RISK-VLA v3 world tokens",
    "RISK-VLA v3 VLM risk LoRA",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RISK-VLA v3 full-scale experiment matrix from an explicit manifest.")
    parser.add_argument("--manifest-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = yaml.safe_load(args.manifest_yaml.read_text(encoding="utf-8"))
    output: Dict[str, Any] = {
        "manifest": str(args.manifest_yaml),
        "launchable": manifest.get("launchable", {}),
        "blockers": manifest.get("blockers", []),
        "matrix": MATRIX,
        "dry_run": bool(args.dry_run),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "risk_vla_v3_full_matrix_plan.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# RISK-VLA v3 Full Matrix Plan",
        "",
        "## Launchability",
        "",
    ]
    for key, value in output["launchable"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Matrix", ""])
    lines.extend(f"- {item}" for item in MATRIX)
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- {item}" for item in output["blockers"] or ["None"])
    (args.output_dir / "risk_vla_v3_full_matrix_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"plan": str(args.output_dir / "risk_vla_v3_full_matrix_plan.json"), "blockers": output["blockers"]}, sort_keys=True))
    return 0 if not output["blockers"] or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
