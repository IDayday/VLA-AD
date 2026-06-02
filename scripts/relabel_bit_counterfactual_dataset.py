#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def load_metadata(input_jsonl: Path) -> Dict[str, Any]:
    metadata_path = input_jsonl.parent / "metadata.json"
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def label_row(
    base_metrics: Optional[Dict[str, Optional[float]]],
    bit_metrics: Optional[Dict[str, Optional[float]]],
    *,
    margin_pdm: float,
    require_no_nc_regression: bool,
    require_no_ttc_regression: bool,
    require_dac_nonregression: bool,
) -> Tuple[Optional[int], str, Dict[str, Optional[float]]]:
    delta: Dict[str, Optional[float]] = {"pdm": None, "dac": None, "nc": None, "ttc": None}
    if base_metrics is None or bit_metrics is None:
        return None, "missing_metrics", delta
    for key in delta:
        base_value = base_metrics.get(key)
        bit_value = bit_metrics.get(key)
        if base_value is not None and bit_value is not None:
            delta[key] = float(bit_value) - float(base_value)
    if delta["pdm"] is None:
        return None, "missing_pdm", delta
    if delta["pdm"] <= margin_pdm:
        return 0, "pdm_margin_not_met", delta
    if require_no_nc_regression and float(base_metrics.get("nc") or 0.0) > 1e-9 and float(bit_metrics.get("nc") or 0.0) <= 1e-9:
        return 0, "nc_regression", delta
    if require_no_ttc_regression and float(base_metrics.get("ttc") or 0.0) > 1e-9 and float(bit_metrics.get("ttc") or 0.0) <= 1e-9:
        return 0, "ttc_regression", delta
    if require_dac_nonregression and float(bit_metrics.get("dac") or 0.0) + 1e-9 < float(base_metrics.get("dac") or 0.0):
        return 0, "dac_regression", delta
    return 1, "bit_improves_pdm_and_safety_ok", delta


def label_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {0: 0, 1: 0, None: 0}
    reasons: Dict[str, int] = {}
    for row in rows:
        label = row.get("label_use_bit")
        labels[label] = labels.get(label, 0) + 1
        reason = str(row.get("label_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "num_samples": len(rows),
        "label_0": labels.get(0, 0),
        "label_1": labels.get(1, 0),
        "label_missing": labels.get(None, 0),
        "positive_rate": labels.get(1, 0) / len(rows) if rows else None,
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel an existing Base-vs-BiT counterfactual JSONL without rerunning models.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-pdm", type=float, default=0.005)
    parser.add_argument("--require-no-nc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-no-ttc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-dac-nonregression", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-prediction-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_metadata = load_metadata(args.input_jsonl)
    split = str(source_metadata.get("split", "")).lower()
    if "test" in split:
        raise RuntimeError(f"Refusing to relabel test split for selector training: split={source_metadata.get('split')!r}")

    rows = read_jsonl(args.input_jsonl)
    for row in rows:
        label, reason, delta = label_row(
            row.get("base_metrics"),
            row.get("bit_metrics"),
            margin_pdm=args.margin_pdm,
            require_no_nc_regression=args.require_no_nc_regression,
            require_no_ttc_regression=args.require_no_ttc_regression,
            require_dac_nonregression=args.require_dac_nonregression,
        )
        row["label_use_bit"] = label
        row["label_reason"] = reason
        row["delta"] = delta

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", rows)
    distribution = label_distribution(rows)
    (args.output_dir / "label_distribution.json").write_text(
        json.dumps(distribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        **source_metadata,
        "source_counterfactual_jsonl": str(args.input_jsonl),
        "margin_pdm": args.margin_pdm,
        "require_no_nc_regression": args.require_no_nc_regression,
        "require_no_ttc_regression": args.require_no_ttc_regression,
        "require_dac_nonregression": args.require_dac_nonregression,
        "label_missing_count": distribution["label_missing"],
        "all_labels_present": distribution["label_missing"] == 0,
        "training_allowed": bool(not source_metadata.get("analysis_only", False) and "test" not in split),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.copy_prediction_files:
        for name in ("base_predictions.jsonl", "bit_predictions.jsonl"):
            source = args.input_jsonl.parent / name
            if source.is_file():
                shutil.copy2(source, args.output_dir / name)

    lines = [
        "# Relabeled BiT Counterfactual Dataset",
        "",
        f"Source: `{args.input_jsonl}`",
        f"Split: `{source_metadata.get('split')}`",
        f"Margin PDM: `{args.margin_pdm}`",
        f"Training allowed: `{metadata['training_allowed']}`",
        "",
        "```json",
        json.dumps(distribution, indent=2, sort_keys=True),
        "```",
    ]
    (args.output_dir / "counterfactual_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **distribution}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
