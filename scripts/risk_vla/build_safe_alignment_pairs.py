from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


def build_pairs(labels_jsonl: Path, output_dir: Path, max_pairs: int | None = None) -> Dict[str, Any]:
    rows_by_token: Dict[str, list[Dict[str, Any]]] = {}
    with labels_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_token.setdefault(str(row["sample_token"]), []).append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "safe_alignment_pairs.jsonl"
    num_pairs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for token, rows in sorted(rows_by_token.items()):
            positive_id = str(rows[0].get("positive_anchor"))
            negative_id = str(rows[0].get("negative_anchor"))
            positive = next((row for row in rows if str(row.get("candidate_id")) == positive_id), None)
            negative = next((row for row in rows if str(row.get("candidate_id")) == negative_id), None)
            if positive is None or negative is None or positive_id == negative_id:
                continue
            pair = {
                "sample_token": token,
                "token": token,
                "token_id": token,
                "split": rows[0].get("split"),
                "purpose": rows[0].get("purpose"),
                "positive_candidate_id": positive_id,
                "negative_candidate_id": negative_id,
                "positive_strategy_name": positive.get("strategy_name"),
                "negative_strategy_name": negative.get("strategy_name"),
                "positive_score": positive.get("pdm_score"),
                "negative_score": negative.get("pdm_score"),
                "negative_regresses_nc": negative.get("regresses_nc"),
                "negative_regresses_ttc": negative.get("regresses_ttc"),
                "positive_repairs_path": positive.get("repairs_path"),
                "positive_utility_score": positive.get("utility_score"),
                "negative_utility_score": negative.get("utility_score"),
                "pair_type": "safe_positive_vs_risky_negative",
                "tail_risk_label": bool(positive.get("tail_risk_label") or negative.get("tail_risk_label")),
            }
            out.write(json.dumps(pair, sort_keys=True))
            out.write("\n")
            num_pairs += 1
            if max_pairs is not None and num_pairs >= max_pairs:
                break
    summary = {"labels_jsonl": str(labels_jsonl), "pairs_jsonl": str(out_path), "num_pairs": num_pairs}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "safe_alignment_pairs_report.md").write_text(
        "\n".join(["# Safe Alignment Pair Report", "", f"Pairs: `{num_pairs}`", f"Source: `{labels_jsonl}`"]) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build positive/negative safe-alignment pairs from utility labels.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        rows = sum(1 for line in args.labels_jsonl.open("r", encoding="utf-8") if line.strip()) if args.labels_jsonl.is_file() else 0
        print(json.dumps({"labels_jsonl": str(args.labels_jsonl), "rows": rows, "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    print(json.dumps(build_pairs(args.labels_jsonl, args.output_dir, args.max_pairs), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
