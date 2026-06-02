#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence


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


def metric(row: Dict[str, Any], side: str, key: str) -> float:
    value = (row.get(f"{side}_metrics") or {}).get(key)
    return float(value) if value is not None else 0.0


def is_zero(value: float) -> bool:
    return value <= EPS


def sample_tags(row: Dict[str, Any], *, margin_pdm: float) -> List[str]:
    base_pdm = metric(row, "base", "pdm")
    bit_pdm = metric(row, "bit", "pdm")
    base_dac = metric(row, "base", "dac")
    bit_dac = metric(row, "bit", "dac")
    base_nc = metric(row, "base", "nc")
    bit_nc = metric(row, "bit", "nc")
    base_ttc = metric(row, "base", "ttc")
    bit_ttc = metric(row, "bit", "ttc")
    tags: List[str] = []

    if is_zero(base_pdm):
        tags.append("base_zero")
    if is_zero(bit_pdm):
        tags.append("bit_zero")
    if is_zero(base_pdm) and not is_zero(bit_pdm):
        tags.append("zero_fixed")
    if not is_zero(base_pdm) and is_zero(bit_pdm):
        tags.append("zero_newly_broken")

    if is_zero(base_dac) and not is_zero(bit_dac):
        tags.append("dac_fixed")
    if not is_zero(base_dac) and is_zero(bit_dac):
        tags.append("dac_regressed")
    if is_zero(base_nc) and not is_zero(bit_nc):
        tags.append("nc_fixed")
    if not is_zero(base_nc) and is_zero(bit_nc):
        tags.append("nc_regressed")
    if is_zero(base_ttc) and not is_zero(bit_ttc):
        tags.append("ttc_fixed")
    if not is_zero(base_ttc) and is_zero(bit_ttc):
        tags.append("ttc_regressed")

    if bit_pdm > base_pdm + margin_pdm:
        tags.append("pdm_improved")
    elif base_pdm > bit_pdm + margin_pdm:
        tags.append("pdm_regressed")
    else:
        tags.append("pdm_neutral")

    safety_ok = "nc_regressed" not in tags and "ttc_regressed" not in tags
    if safety_ok:
        tags.append("safety_nonregression")
    else:
        tags.append("interaction_safety_risk")

    if ("dac_fixed" in tags or "zero_fixed" in tags) and safety_ok and bit_pdm > base_pdm + margin_pdm:
        tags.append("bit_suitable_path_domain")
    if ("nc_regressed" in tags or "ttc_regressed" in tags) and not ("dac_fixed" in tags or "zero_fixed" in tags):
        tags.append("bit_unsuitable_interaction_domain")
    if "dac_fixed" in tags and ("nc_regressed" in tags or "ttc_regressed" in tags):
        tags.append("path_safety_tradeoff")
    if bit_pdm > base_pdm + margin_pdm and "dac_fixed" not in tags and "zero_fixed" not in tags:
        tags.append("non_path_pdm_gain")
    return tags


def aggregate_choice(rows: Sequence[Dict[str, Any]], use_bit: Sequence[bool]) -> Dict[str, Any]:
    pdm: List[float] = []
    dac0 = nc0 = ttc0 = zero = 0
    ego: List[float] = []
    for row, choose_bit in zip(rows, use_bit):
        side = "bit" if choose_bit else "base"
        pdm_value = metric(row, side, "pdm")
        pdm.append(pdm_value)
        zero += int(is_zero(pdm_value))
        dac0 += int(is_zero(metric(row, side, "dac")))
        nc0 += int(is_zero(metric(row, side, "nc")))
        ttc0 += int(is_zero(metric(row, side, "ttc")))
        ego.append(metric(row, side, "ego"))
    pdm_sorted = sorted(pdm)
    p10 = pdm_sorted[max(0, min(len(pdm_sorted) - 1, int(0.10 * (len(pdm_sorted) - 1))))] if pdm_sorted else 0.0
    return {
        "samples": len(rows),
        "mean": sum(pdm) / len(pdm) if pdm else 0.0,
        "median": median(pdm) if pdm else 0.0,
        "p10": p10,
        "zero": zero,
        "dac0": dac0,
        "nc0": nc0,
        "ttc0": ttc0,
        "ego": sum(ego) / len(ego) if ego else 0.0,
        "use_bit": sum(1 for value in use_bit if value),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "sample_token",
        "scene_token",
        "base_pdm",
        "bit_pdm",
        "delta_pdm",
        "base_dac",
        "bit_dac",
        "base_nc",
        "bit_nc",
        "base_ttc",
        "bit_ttc",
        "tags",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def analyze_dataset(name: str, jsonl: Path, *, margin_pdm: float) -> Dict[str, Any]:
    rows = read_jsonl(jsonl)
    annotated: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    for row in rows:
        tags = sample_tags(row, margin_pdm=margin_pdm)
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        annotated.append({
            "dataset": name,
            "sample_token": row.get("sample_token"),
            "scene_token": row.get("scene_token"),
            "base_pdm": metric(row, "base", "pdm"),
            "bit_pdm": metric(row, "bit", "pdm"),
            "delta_pdm": metric(row, "bit", "pdm") - metric(row, "base", "pdm"),
            "base_dac": metric(row, "base", "dac"),
            "bit_dac": metric(row, "bit", "dac"),
            "base_nc": metric(row, "base", "nc"),
            "bit_nc": metric(row, "bit", "nc"),
            "base_ttc": metric(row, "base", "ttc"),
            "bit_ttc": metric(row, "bit", "ttc"),
            "tags": ",".join(tags),
            "_raw": row,
        })

    tag_sets = [set(str(row["tags"]).split(",")) for row in annotated]
    base_choice = [False] * len(rows)
    bit_choice = [True] * len(rows)
    path_domain_choice = ["bit_suitable_path_domain" in tags for tags in tag_sets]
    safety_oracle_choice = [
        "nc_regressed" not in tags and "ttc_regressed" not in tags
        for tags in tag_sets
    ]
    oracle_best_choice = [metric(row, "bit", "pdm") > metric(row, "base", "pdm") for row in rows]
    return {
        "name": name,
        "jsonl": str(jsonl),
        "samples": len(rows),
        "tag_counts": dict(sorted(tag_counts.items())),
        "aggregates": {
            "base": aggregate_choice(rows, base_choice),
            "bit": aggregate_choice(rows, bit_choice),
            "path_domain_oracle_analysis_only": aggregate_choice(rows, path_domain_choice),
            "safety_oracle_analysis_only": aggregate_choice(rows, safety_oracle_choice),
            "oracle_best_analysis_only": aggregate_choice(rows, oracle_best_choice),
        },
        "annotated": annotated,
    }


def parse_dataset_args(values: Sequence[str]) -> List[tuple[str, Path]]:
    if len(values) % 2 != 0:
        raise ValueError("--datasets expects NAME JSONL pairs.")
    pairs = []
    for idx in range(0, len(values), 2):
        pairs.append((values[idx], Path(values[idx + 1])))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Base-vs-BiT case taxonomy and BiT-suitable domains.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Pairs: NAME counterfactual_samples.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-pdm", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_pairs = parse_dataset_args(args.datasets)
    results = [analyze_dataset(name, path, margin_pdm=args.margin_pdm) for name, path in dataset_pairs]

    all_rows: List[Dict[str, Any]] = []
    case_dir = args.output_dir / "case_lists"
    case_dir.mkdir(exist_ok=True)
    for result in results:
        rows = result.pop("annotated")
        all_rows.extend(rows)
        for tag in (
            "bit_suitable_path_domain",
            "bit_unsuitable_interaction_domain",
            "path_safety_tradeoff",
            "dac_fixed",
            "nc_regressed",
            "ttc_regressed",
            "zero_fixed",
            "zero_newly_broken",
        ):
            write_jsonl(
                case_dir / f"{result['name']}_{tag}.jsonl",
                (row["_raw"] | {"taxonomy_tags": str(row["tags"]).split(",")} for row in rows if tag in str(row["tags"]).split(",")),
            )
        for row in rows:
            row.pop("_raw", None)

    summary = {
        "margin_pdm": args.margin_pdm,
        "datasets": results,
    }
    (args.output_dir / "case_taxonomy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "case_taxonomy_samples.csv", all_rows)

    lines = [
        "# BiT Case Taxonomy",
        "",
        f"Margin PDM: `{args.margin_pdm}`",
        "",
        "This analysis separates BiT path/DAC-suitable cases from interaction-safety cases. Oracle rows are analysis-only and use per-sample metrics.",
        "",
    ]
    for result in results:
        lines.extend([
            f"## {result['name']}",
            "",
            "| Method | Mean | P10 | Zero | DAC0 | NC0 | TTC0 | Ego | Use Bit |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for method, agg in result["aggregates"].items():
            lines.append(
                f"| {method} | {agg['mean']:.6f} | {agg['p10']:.6f} | {agg['zero']} | {agg['dac0']} | "
                f"{agg['nc0']} | {agg['ttc0']} | {agg['ego']:.6f} | {agg['use_bit']} |"
            )
        lines.extend(["", "| Tag | Count |", "| --- | ---: |"])
        for tag, count in result["tag_counts"].items():
            lines.append(f"| {tag} | {count} |")
        lines.append("")
    (args.output_dir / "case_taxonomy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": [r["name"] for r in results]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
