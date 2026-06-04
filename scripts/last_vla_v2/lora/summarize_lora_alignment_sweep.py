#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Last-VLA VLM-LoRA cot-alignment sweep diagnostics.")
    parser.add_argument("--out-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in args.out_root.iterdir() if path.is_dir()):
        train_args = read_json(run_dir / "train_args.json") or {}
        precision = read_json(run_dir / "precision_report.json") or {}
        target_report = read_json(run_dir / "lora_target_report.json") or {}
        extract_report = read_json(run_dir / "adapter_extract_report.json") or {}
        agent = (train_args.get("agent") or {}) if isinstance(train_args.get("agent"), dict) else {}
        by_cat = target_report.get("matched_by_category") or {}
        counts = precision.get("trainable_parameter_counts") or {}
        row = {
            "run_name": run_dir.name,
            "preset": agent.get("last_vla_vlm_lora_preset") or target_report.get("preset"),
            "scope": agent.get("last_vla_vlm_lora_scope") or target_report.get("scope"),
            "r": agent.get("last_vla_vlm_lora_r"),
            "alpha": agent.get("last_vla_vlm_lora_alpha"),
            "dropout": agent.get("last_vla_vlm_lora_dropout"),
            "use_rslora": agent.get("last_vla_vlm_lora_use_rslora"),
            "use_dora": agent.get("last_vla_vlm_lora_use_dora"),
            "matched_total": target_report.get("matched_total"),
            "matched_llm_attention": by_cat.get("llm_attention"),
            "matched_llm_mlp": by_cat.get("llm_mlp"),
            "matched_vision_attention": by_cat.get("vision_attention"),
            "matched_vision_mlp": by_cat.get("vision_mlp"),
            "trainable_params_lora": (counts.get("vlm_lora") or {}).get("trainable"),
            "trainable_params_cot": (counts.get("last_vla_cot") or {}).get("trainable"),
            "lr_lora": agent.get("lr_vlm_lora"),
            "lr_cot": agent.get("lr_last_vla_cot"),
            "final_geometry_loss": None,
            "final_dynamic_loss": None,
            "final_coarse_loss": None,
            "hidden_anchor_loss": None,
            "hidden_drift_cosine": None,
            "notes": extract_report.get("error") if extract_report else "",
        }
        rows.append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_root / "lora_sweep_summary.csv"
    fields = [
        "run_name",
        "preset",
        "scope",
        "r",
        "alpha",
        "dropout",
        "use_rslora",
        "use_dora",
        "matched_total",
        "matched_llm_attention",
        "matched_llm_mlp",
        "matched_vision_attention",
        "matched_vision_mlp",
        "trainable_params_lora",
        "trainable_params_cot",
        "lr_lora",
        "lr_cot",
        "final_geometry_loss",
        "final_dynamic_loss",
        "final_coarse_loss",
        "hidden_anchor_loss",
        "hidden_drift_cosine",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = [
        "# Last-VLA VLM-LoRA Alignment Sweep Summary",
        "",
        "- Baseline: A0-official-aligned PDMS `0.864891`",
        "- No performance is claimed from diagnostics alone.",
        "",
        "| run | preset | r | alpha | matched | lora params | cot params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['run_name']} | {row['preset']} | {row['r']} | {row['alpha']} | "
            f"{row['matched_total']} | {row['trainable_params_lora']} | {row['trainable_params_cot']} |"
        )
    (args.out_root / "lora_sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(rows), "summary_csv": str(csv_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
