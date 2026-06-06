from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def file_status(path: Path) -> dict[str, object]:
    return {"path": str(path), "exists": path.is_file(), "size": path.stat().st_size if path.is_file() else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISK-VLA v2 Round2 artifacts.")
    parser.add_argument("--exp-root", type=Path, default=Path("experiments/risk_vla/v2"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.exp_root
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "utility_labels": file_status(args.exp_root / "utility_labels" / "summary.json"),
        "safe_alignment_pairs": file_status(args.exp_root / "safe_alignment_pairs" / "summary.json"),
        "candidate_bank": file_status(args.exp_root / "candidate_bank" / "summary.json"),
        "critic": file_status(args.exp_root / "critic" / "summary.json"),
        "router": file_status(args.exp_root / "router" / "summary.json"),
    }
    summary = {"exp_root": str(args.exp_root), "artifacts": artifacts}
    (output_dir / "risk_vla_v2_results_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# RISK-VLA v2 Results Summary", ""]
    for name, status in artifacts.items():
        lines.append(f"- {name}: exists=`{status['exists']}` path=`{status['path']}`")
    (output_dir / "risk_vla_v2_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
