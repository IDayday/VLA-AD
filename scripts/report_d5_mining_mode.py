#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SHARED_EXPERIMENT_ROOT = Path("/mnt/project/VLA-AD/experiments")
EXPECTED_MODES = [
    "base_det",
    "bit_det",
    "bit_stochastic_seed0",
    "bit_stochastic_seed1",
    "bit_stochastic_seed2",
    "bit_low_strength",
    "bit_high_strength",
    "bit_path_only",
    "bit_terminal_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report whether D5 mining outputs are true multi-candidate outputs.")
    parser.add_argument("--input-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=200000)
    return parser.parse_args()


def assert_not_shared_output(path: Path) -> None:
    resolved = path.resolve()
    shared = SHARED_EXPERIMENT_ROOT.resolve()
    if resolved == shared or shared in resolved.parents:
        raise RuntimeError(f"Refusing to write report under shared experiment root: {shared}")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_decode_error": str(path)}


def iter_jsonl(path: Path, max_rows: int) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"_decode_error": index}


def pick_row_file(root: Path) -> Optional[Path]:
    for name in (
        "counterfactual_candidates.jsonl",
        "selected_training_labels.jsonl",
        "counterfactual_mined.jsonl",
    ):
        path = root / name
        if path.is_file():
            return path
    return None


def script_name(metadata: Dict[str, Any], summary: Dict[str, Any], row_file: Optional[Path]) -> str:
    for source in (metadata, summary):
        for key in ("script", "script_name", "generated_by", "generator"):
            if source.get(key):
                return str(source[key])
    if row_file is None:
        return "unknown"
    if row_file.name == "counterfactual_candidates.jsonl":
        return "mine_bit_counterfactual_safety_cases_v2.py or compatible"
    if row_file.name == "counterfactual_mined.jsonl":
        return "mine_bit_counterfactual_safety_cases.py or legacy compatible"
    return "unknown"


def tags_for(row: Dict[str, Any]) -> List[str]:
    tags = row.get("tags") or row.get("bit_counterfactual_tags") or []
    if not isinstance(tags, list):
        return []
    return [str(tag) for tag in tags]


def analyze_root(root: Path, max_rows: int) -> Dict[str, Any]:
    metadata = read_json(root / "metadata.json")
    summary = read_json(root / "mining_summary.json")
    row_file = pick_row_file(root)
    rows = list(iter_jsonl(row_file, max_rows)) if row_file is not None else []
    mode_counter: Counter[str] = Counter()
    scene_to_modes: Dict[str, set[str]] = defaultdict(set)
    tag_counter: Counter[str] = Counter()
    stratum_counter: Counter[str] = Counter()
    rows_with_soft = 0
    rows_with_annotation_meta = 0
    rows_with_sampling_stratum = 0
    candidate_level = False

    for row in rows:
        mode = row.get("candidate_mode") or row.get("counterfactual_candidate_mode")
        if mode is not None:
            mode = str(mode)
            mode_counter[mode] += 1
            candidate_level = True
        scene_token = str(row.get("scene_token") or row.get("sample_token") or row.get("row_index") or len(scene_to_modes))
        if mode is not None:
            scene_to_modes[scene_token].add(mode)
        else:
            scene_to_modes.setdefault(scene_token, set())
        tags = tags_for(row)
        tag_counter.update(tags)
        if bool(row.get("soft_safety_mask")) or any(tag in tags for tag in ("soft_safety_regression", "soft_ttc_regression")):
            rows_with_soft += 1
        scene_meta = row.get("scene_metadata") if isinstance(row.get("scene_metadata"), dict) else {}
        if scene_meta.get("annotation_access") or scene_meta.get("annotation_names_histogram"):
            rows_with_annotation_meta += 1
        stratum = row.get("sampling_stratum") or row.get("sampling_bucket")
        if stratum:
            rows_with_sampling_stratum += 1
            stratum_counter[str(stratum)] += 1

    modes = sorted(mode_counter)
    candidates_per_scene = [len(modes_for_scene) for modes_for_scene in scene_to_modes.values()]
    multi_candidate = bool(rows and candidate_level and (max(candidates_per_scene or [0]) > 1 or len(modes) > 1))
    annotation_sampling = bool(
        summary.get("sampling_mode") in {"mixed", "dense_agents", "pedestrian_present", "vehicle_present", "vehicle_count_high", "moving_vehicle_present"}
        or rows_with_annotation_meta > 0
        or rows_with_sampling_stratum > 0
    )
    return {
        "root": str(root),
        "exists": root.exists(),
        "row_file": str(row_file) if row_file else None,
        "script": script_name(metadata, summary, row_file),
        "metadata_present": bool(metadata),
        "summary_present": bool(summary),
        "rows_sampled": len(rows),
        "summary_total_rows": summary.get("total_rows"),
        "summary_candidate_rows": summary.get("candidate_rows"),
        "candidate_modes": {mode: mode in mode_counter or mode in summary.get("candidate_modes", []) for mode in EXPECTED_MODES},
        "candidate_mode_counts": dict(mode_counter),
        "rows_are_candidate_level": candidate_level,
        "rows_are_scene_level": bool(rows and not candidate_level),
        "scene_count_sampled": len(scene_to_modes),
        "candidates_per_scene": {
            "min": min(candidates_per_scene) if candidates_per_scene else 0,
            "max": max(candidates_per_scene) if candidates_per_scene else 0,
            "mean": (sum(candidates_per_scene) / len(candidates_per_scene)) if candidates_per_scene else 0.0,
        },
        "soft_safety_labels_present": rows_with_soft > 0 or int(summary.get("soft_safety_regressions", 0) or 0) > 0,
        "soft_safety_rows_sampled": rows_with_soft,
        "annotation_aware_sampling_used": annotation_sampling,
        "sampling_mode": summary.get("sampling_mode"),
        "sampling_stratum_counts_sampled": dict(stratum_counter),
        "tag_counts_sampled": dict(tag_counter),
        "v2_multi_candidate_mining_ran": multi_candidate and row_file is not None and row_file.name == "counterfactual_candidates.jsonl",
    }


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# D5 Mining Mode Report",
        "",
        f"True v2 multi-candidate mining ran: `{report['v2_multi_candidate_mining_ran']}`",
        "",
        "| Root | Exists | Script | Row file | Candidate level | Max candidates/scene | Soft labels | Annotation-aware sampling |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["roots"]:
        row_file = Path(item["row_file"]).name if item.get("row_file") else "-"
        lines.append(
            f"| {item['root']} | {item['exists']} | {item['script']} | {row_file} | "
            f"{item['rows_are_candidate_level']} | {item['candidates_per_scene']['max']} | "
            f"{item['soft_safety_labels_present']} | {item['annotation_aware_sampling_used']} |"
        )
    lines.extend(["", "## Candidate Modes", ""])
    for item in report["roots"]:
        lines.extend(
            [
                f"### {item['root']}",
                "",
                "```json",
                json.dumps(item["candidate_modes"], indent=2, sort_keys=True),
                "```",
                "",
                "Candidate mode counts:",
                "",
                "```json",
                json.dumps(item["candidate_mode_counts"], indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    assert_not_shared_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = [analyze_root(root, args.max_rows) for root in args.input_roots]
    report = {
        "roots": roots,
        "v2_multi_candidate_mining_ran": any(item["v2_multi_candidate_mining_ran"] for item in roots),
    }
    (args.output_dir / "D5_MINING_MODE_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "D5_MINING_MODE_REPORT.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
