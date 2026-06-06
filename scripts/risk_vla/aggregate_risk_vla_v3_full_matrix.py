#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate RISK-VLA v3 result summaries under an experiment root.")
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summaries = []
    if args.exp_root.exists():
        for path in sorted(args.exp_root.rglob("summary.json")):
            try:
                summaries.append({"path": str(path), "summary": json.loads(path.read_text(encoding="utf-8"))})
            except Exception as exc:
                summaries.append({"path": str(path), "error": str(exc)})
    output: Dict[str, Any] = {"exp_root": str(args.exp_root), "num_summaries": len(summaries), "summaries": summaries, "dry_run": bool(args.dry_run)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "risk_vla_v3_full_matrix_aggregate.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# RISK-VLA v3 Full Matrix Aggregate", "", f"Experiment root: `{args.exp_root}`", f"Summary files: `{len(summaries)}`", ""]
    for item in summaries[:50]:
        lines.append(f"- `{item['path']}`")
    (args.output_dir / "risk_vla_v3_full_matrix_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": str(args.output_dir / "risk_vla_v3_full_matrix_aggregate.json"), "num_summaries": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
