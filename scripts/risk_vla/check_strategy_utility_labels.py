from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence


REQUIRED_FIELDS = {
    "sample_token",
    "candidate_id",
    "strategy_name",
    "pdm_score",
    "delta_score",
    "repairs_path",
    "regresses_nc",
    "regresses_ttc",
    "constrained_best_strategy",
    "positive_anchor",
    "negative_anchor",
    "tail_risk_label",
}


def check_labels(path: Path, *, purpose: str = "training") -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary: Dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "tokens": set(),
        "missing_field_counts": {},
        "candidate_counts": {},
        "tail_risk_rows": 0,
        "blocking_problems": [],
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            summary["rows"] += 1
            missing = REQUIRED_FIELDS - set(row)
            for field in missing:
                summary["missing_field_counts"][field] = summary["missing_field_counts"].get(field, 0) + 1
            split = str(row.get("split", "")).lower()
            if purpose == "training" and ("test" in split or "navtest" in split):
                summary["blocking_problems"].append(f"row {line_no}: training label uses split={split!r}")
            token = str(row.get("sample_token", ""))
            if token:
                summary["tokens"].add(token)
            candidate = str(row.get("candidate_name") or row.get("strategy_name") or row.get("candidate_id"))
            summary["candidate_counts"][candidate] = summary["candidate_counts"].get(candidate, 0) + 1
            summary["tail_risk_rows"] += int(bool(row.get("tail_risk_label")))
    if summary["missing_field_counts"]:
        summary["blocking_problems"].append("missing required fields")
    summary["tokens"] = len(summary["tokens"])
    summary["ready_for_training"] = purpose != "training" or not summary["blocking_problems"]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RISK-VLA v2 strategy utility labels.")
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "analysis"), default="training")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = check_labels(args.labels_jsonl, purpose=args.purpose)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not summary["blocking_problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
