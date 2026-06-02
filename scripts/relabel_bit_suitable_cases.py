#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EPS = 1e-9


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
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def metric(row: Dict[str, Any], side: str, key: str) -> Optional[float]:
    value = (row.get(f"{side}_metrics") or {}).get(key)
    return float(value) if value is not None else None


def zero(value: Optional[float]) -> bool:
    return value is not None and value <= EPS


def positive(value: Optional[float]) -> bool:
    return value is not None and value > EPS


def relabel_row(
    row: Dict[str, Any],
    *,
    margin_pdm: float,
    require_dac_fix: bool,
    allow_zero_fix_without_dac: bool,
    require_no_nc_regression: bool,
    require_no_ttc_regression: bool,
) -> Tuple[Optional[int], str, Dict[str, Optional[float]], List[str]]:
    base_pdm = metric(row, "base", "pdm")
    bit_pdm = metric(row, "bit", "pdm")
    base_dac = metric(row, "base", "dac")
    bit_dac = metric(row, "bit", "dac")
    base_nc = metric(row, "base", "nc")
    bit_nc = metric(row, "bit", "nc")
    base_ttc = metric(row, "base", "ttc")
    bit_ttc = metric(row, "bit", "ttc")
    delta = {
        "pdm": None if base_pdm is None or bit_pdm is None else bit_pdm - base_pdm,
        "dac": None if base_dac is None or bit_dac is None else bit_dac - base_dac,
        "nc": None if base_nc is None or bit_nc is None else bit_nc - base_nc,
        "ttc": None if base_ttc is None or bit_ttc is None else bit_ttc - base_ttc,
    }
    if any(value is None for value in (base_pdm, bit_pdm, base_dac, bit_dac, base_nc, bit_nc, base_ttc, bit_ttc)):
        return None, "missing_metrics", delta, ["missing_metrics"]

    tags: List[str] = []
    if zero(base_pdm) and positive(bit_pdm):
        tags.append("zero_fixed")
    if positive(base_pdm) and zero(bit_pdm):
        tags.append("zero_newly_broken")
    if zero(base_dac) and positive(bit_dac):
        tags.append("dac_fixed")
    if positive(base_dac) and zero(bit_dac):
        tags.append("dac_regressed")
    if positive(base_nc) and zero(bit_nc):
        tags.append("nc_regressed")
    if zero(base_nc) and positive(bit_nc):
        tags.append("nc_fixed")
    if positive(base_ttc) and zero(bit_ttc):
        tags.append("ttc_regressed")
    if zero(base_ttc) and positive(bit_ttc):
        tags.append("ttc_fixed")

    if require_no_nc_regression and "nc_regressed" in tags:
        return 0, "reject_nc_regression", delta, tags
    if require_no_ttc_regression and "ttc_regressed" in tags:
        return 0, "reject_ttc_regression", delta, tags
    if "dac_regressed" in tags:
        return 0, "reject_dac_regression", delta, tags
    if delta["pdm"] is None or delta["pdm"] <= margin_pdm:
        return 0, "reject_pdm_margin", delta, tags

    dac_suitable = "dac_fixed" in tags
    zero_suitable = allow_zero_fix_without_dac and "zero_fixed" in tags
    if require_dac_fix and not dac_suitable:
        return 0, "reject_not_dac_fix", delta, tags
    if not require_dac_fix and not (dac_suitable or zero_suitable):
        return 0, "reject_not_path_zero_domain", delta, tags
    tags.append("bit_suitable_dac_nc_safe")
    return 1, "accept_dac_or_zero_fix_safety_ok", delta, tags


def label_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {0: 0, 1: 0, None: 0}
    reasons: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}
    for row in rows:
        label = row.get("label_use_bit")
        labels[label] = labels.get(label, 0) + 1
        reason = str(row.get("label_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
        for tag in row.get("taxonomy_tags") or []:
            tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
    return {
        "num_samples": len(rows),
        "label_0": labels.get(0, 0),
        "label_1": labels.get(1, 0),
        "label_missing": labels.get(None, 0),
        "positive_rate": labels.get(1, 0) / len(rows) if rows else None,
        "reasons": dict(sorted(reasons.items())),
        "tags": dict(sorted(tag_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel counterfactual data for DAC/zero-fix BiT-suitable domain.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-pdm", type=float, default=0.0)
    parser.add_argument("--require-dac-fix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-zero-fix-without-dac", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-no-nc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-no-ttc-regression", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-prediction-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.input_jsonl)
    split = str(metadata.get("split", "")).lower()
    if "test" in split:
        raise RuntimeError(f"Refusing to relabel test split for training: split={metadata.get('split')!r}")

    rows = []
    for row in read_jsonl(args.input_jsonl):
        row = dict(row)
        label, reason, delta, tags = relabel_row(
            row,
            margin_pdm=args.margin_pdm,
            require_dac_fix=args.require_dac_fix,
            allow_zero_fix_without_dac=args.allow_zero_fix_without_dac,
            require_no_nc_regression=args.require_no_nc_regression,
            require_no_ttc_regression=args.require_no_ttc_regression,
        )
        row["label_use_bit"] = label
        row["label_reason"] = reason
        row["delta"] = delta
        row["taxonomy_tags"] = tags
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "counterfactual_samples.jsonl", rows)
    distribution = label_distribution(rows)
    (args.output_dir / "label_distribution.json").write_text(json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    new_metadata = {
        **metadata,
        "source_counterfactual_jsonl": str(args.input_jsonl),
        "label_policy": "dac_or_zero_fix_safety_ok",
        "margin_pdm": args.margin_pdm,
        "require_dac_fix": args.require_dac_fix,
        "allow_zero_fix_without_dac": args.allow_zero_fix_without_dac,
        "require_no_nc_regression": args.require_no_nc_regression,
        "require_no_ttc_regression": args.require_no_ttc_regression,
        "label_missing_count": distribution["label_missing"],
        "all_labels_present": distribution["label_missing"] == 0,
        "training_allowed": bool(not metadata.get("analysis_only", False) and "test" not in split),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(new_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.copy_prediction_files:
        for name in ("base_predictions.jsonl", "bit_predictions.jsonl"):
            source = args.input_jsonl.parent / name
            if source.is_file():
                shutil.copy2(source, args.output_dir / name)

    lines = [
        "# DAC/NC-Safe Relabeled Counterfactual Dataset",
        "",
        f"Source: `{args.input_jsonl}`",
        f"Training allowed: `{new_metadata['training_allowed']}`",
        f"Policy: require DAC fix `{args.require_dac_fix}`, allow zero fix without DAC `{args.allow_zero_fix_without_dac}`",
        f"Reject NC regression: `{args.require_no_nc_regression}`",
        f"Reject TTC regression: `{args.require_no_ttc_regression}`",
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
