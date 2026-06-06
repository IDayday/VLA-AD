from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare/evaluate RISK-VLA v2 candidate bank PDM jobs.")
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cache_npz = args.candidate_cache_dir / "candidate_trajectories.npz"
    metadata = args.candidate_cache_dir / "candidate_metadata.jsonl"
    summary = {
        "candidate_cache_dir": str(args.candidate_cache_dir),
        "cache_exists": cache_npz.is_file(),
        "metadata_exists": metadata.is_file(),
        "metric_cache_dir": str(args.metric_cache_dir) if args.metric_cache_dir else None,
        "metric_cache_exists": args.metric_cache_dir.is_dir() if args.metric_cache_dir else None,
        "max_samples": args.max_samples,
        "dry_run": bool(args.dry_run),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_bank_pdm_eval_plan.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "candidate_bank_pdm_eval_plan.md").write_text(
        "\n".join(
            [
                "# Candidate Bank PDM Eval Plan",
                "",
                f"Candidate cache exists: `{summary['cache_exists']}`",
                f"Metadata exists: `{summary['metadata_exists']}`",
                f"Metric cache exists: `{summary['metric_cache_exists']}`",
                "",
                "This script records the matched-token candidate-eval plan. Full NAVSIM PDM execution is delegated to existing eval runners.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["cache_exists"] or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
