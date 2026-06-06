from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def load_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    labels: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[str(row["sample_token"])] = row
    return labels


def merge_labels(chunk_cache_dir: Path, labels_jsonl: Path, output_dir: Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    index_path = chunk_cache_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    labels = load_labels(labels_jsonl)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_index = output_dir / "index.jsonl"
    total = 0
    matched = 0
    with index_path.open("r", encoding="utf-8") as src, out_index.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            if max_rows is not None and total >= max_rows:
                break
            row = json.loads(line)
            token = str(row.get("sample_token") or row.get("token") or "")
            if token in labels:
                label = labels[token]
                row["risk_labels_scene"] = label["risk_labels_scene"]
                row["risk_labels_horizon"] = label["risk_labels_horizon"]
                row["risk_event_time_bin"] = label["risk_event_time_bin"]
                row["safety_risk_horizon"] = label["safety_risk_horizon"]
                matched += 1
            dst.write(json.dumps(row, sort_keys=True))
            dst.write("\n")
            total += 1
    summary = {
        "chunk_cache_dir": str(chunk_cache_dir),
        "labels_jsonl": str(labels_jsonl),
        "output_dir": str(output_dir),
        "rows": total,
        "matched_labels": matched,
        "index_jsonl": str(out_index),
    }
    (output_dir / "finegrained_label_merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge fine-grained risk labels into a chunk-cache overlay index.")
    parser.add_argument("--chunk-cache-dir", type=Path, required=True)
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        index = args.chunk_cache_dir / "index.jsonl"
        rows = sum(1 for line in index.open("r", encoding="utf-8") if line.strip()) if index.is_file() else 0
        labels = sum(1 for line in args.labels_jsonl.open("r", encoding="utf-8") if line.strip()) if args.labels_jsonl.is_file() else 0
        print(json.dumps({"index_rows": rows, "label_rows": labels, "would_write": str(args.output_dir)}, sort_keys=True))
        return 0
    print(json.dumps(merge_labels(args.chunk_cache_dir, args.labels_jsonl, args.output_dir, args.max_rows), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
