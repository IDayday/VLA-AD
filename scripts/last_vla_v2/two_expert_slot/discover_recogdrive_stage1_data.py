#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import write_json  # noqa: E402


HF_DATASET = "owl10/ReCogDrive_Pretraining"
HF_ROOT = f"https://huggingface.co/datasets/{HF_DATASET}"
NAVSIM_FILES = {
    "NAVSIM-Traj": {
        "subdir": "Navsim_Traj",
        "filename": "dataset_navsim_traj.jsonl",
        "task": "trajectory_replay",
    },
    "NAVSIM-ReCogDrive": {
        "subdir": "Navsim_ReCogDrive",
        "filename": "dataset_navsim_recogdrive.jsonl",
        "task": "driving_qa",
    },
}


def _fetch_hf_tree() -> Dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main?recursive=1&expand=1"
    with urllib.request.urlopen(url, timeout=45) as response:
        rows = json.load(response)
    return {str(row.get("path")): row for row in rows if isinstance(row, dict)}


def _local_readme_hits(repo_root: Path) -> List[str]:
    hits = []
    readme = repo_root / "README.md"
    if not readme.is_file():
        return hits
    for lineno, line in enumerate(readme.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if "NAVSIM-Traj" in line or "NAVSIM-ReCogDrive" in line or "NAVSIM" in line and "specific dataset" in line:
            hits.append(f"{readme}:{lineno}: {line.strip()}")
    return hits


def discover(local_path: Path | None = None) -> Dict[str, Any]:
    tree = _fetch_hf_tree()
    sources = []
    for name, spec in NAVSIM_FILES.items():
        rel = f"{spec['subdir']}/{spec['filename']}"
        row = tree.get(rel, {})
        lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
        size = int(row.get("size") or lfs.get("size") or 0)
        sources.append(
            {
                "data_source_name": name,
                "source_url": f"{HF_ROOT}/tree/main/{spec['subdir']}",
                "resolve_url": f"{HF_ROOT}/resolve/main/{rel}",
                "license": "Use must comply with original dataset licenses; ReCogDrive README states academic research only and no commercial application.",
                "expected_files": [rel],
                "sha256": None,
                "lfs_oid": lfs.get("oid"),
                "size_bytes": size,
                "download_command": (
                    f"huggingface-cli download {HF_DATASET} --repo-type dataset "
                    f"--include \"{rel}\" --local-dir $EXP_ROOT/data/recogdrive_pretraining_jsonl"
                ),
                "local_path": str(local_path / rel) if local_path else None,
                "prompt_format": "InternVL conversations JSONL; human turns contain prompt text and <image> marker.",
                "answer_format": "[PT, (...)] trajectory for NAVSIM-Traj; free-form driving QA for NAVSIM-ReCogDrive.",
                "sample_count": 85109,
                "task": spec["task"],
            }
        )
    return {
        "status": "official_navsim_stage1_jsonl_found",
        "dataset": HF_DATASET,
        "searched_locations": [
            "README.md Driving Pretraining Datasets section",
            "internvl_chat/shell/data_info/recogdrive_pretrain.json",
            f"{HF_ROOT}/tree/main/Navsim_Traj",
            f"{HF_ROOT}/tree/main/Navsim_ReCogDrive",
            f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main?recursive=1&expand=1",
        ],
        "local_repo_hits": _local_readme_hits(REPO_ROOT),
        "sources": sources,
        "navsim_only_recommendation": (
            "Use NAVSIM-Traj for trajectory replay CE. NAVSIM-ReCogDrive is NAVSIM driving QA and may be used as mixed QA CE, "
            "but it must not be audited as parseable [8,3] trajectory replay."
        ),
    }


def write_markdown(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# ReCogDrive Stage1 NAVSIM Data Discovery",
        "",
        f"- status: `{manifest['status']}`",
        f"- dataset: `{manifest['dataset']}`",
        "- conclusion: official NAVSIM-only data is two JSONL files; image blobs come from the existing NAVSIM dataset.",
        "",
        "## Sources",
        "",
        "| name | task | file | size | lfs oid |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in manifest["sources"]:
        lines.append(
            f"| `{item['data_source_name']}` | `{item['task']}` | `{item['expected_files'][0]}` | "
            f"{item['size_bytes']} | `{item.get('lfs_oid')}` |"
        )
    lines.extend(["", "## Download", ""])
    includes = " ".join(f"\"{item['expected_files'][0]}\"" for item in manifest["sources"])
    lines.extend(
        [
            "```bash",
            f"huggingface-cli download {manifest['dataset']} --repo-type dataset --include {includes} --local-dir $EXP_ROOT/data/recogdrive_pretraining_jsonl",
            "```",
            "",
            "## Notes",
            "",
            f"- {manifest['navsim_only_recommendation']}",
            "- This is official NAVSIM JSONL data, not NAVSIM GT fallback labels.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover official ReCogDrive NAVSIM Stage1 JSONL data.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--local-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = discover(local_path=args.local_path)
    write_json(args.manifest, manifest)
    write_markdown(args.report, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
