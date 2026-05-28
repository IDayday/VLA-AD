#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

CHECKPOINT_SUFFIXES = (".ckpt", ".pth", ".pt", ".safetensors", ".bin")
OUTPUT_KEYS = {
    "recogdrive_2b_il": "recogdrive_2b_il_path",
    "recogdrive_vlm_2b": "recogdrive_vlm_2b_path",
    "vjepa2": "vjepa2_path",
    "vggt_1b": "vggt_1b_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve local ReCogDrive/V-JEPA2/VGGT weight paths from configs/weights.yaml.")
    parser.add_argument("--weights-config", type=Path, default=Path("configs/weights.yaml"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints/resolved_weight_paths.json"))
    parser.add_argument("--json", action="store_true", help="Print only JSON to stdout.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"weights config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"weights config must be a mapping, got {type(data).__name__}")
    return data


def is_plausible_candidate(path: Path) -> bool:
    lower = path.name.lower()
    if ".cache" in path.parts or lower.endswith(".incomplete"):
        return False
    if lower in {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}:
        return False
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return False
    if path.suffix.lower() == ".bin" and path.stat().st_size < 1024 * 1024:
        return False
    return True


def candidate_files(local_dir: Path) -> List[Path]:
    if local_dir.is_file():
        return [local_dir] if is_plausible_candidate(local_dir) else []
    if not local_dir.exists():
        return []
    candidates: List[Path] = []
    for suffix in CHECKPOINT_SUFFIXES:
        candidates.extend(path for path in local_dir.rglob(f"*{suffix}") if path.is_file() and is_plausible_candidate(path))
    seen = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return sorted(unique)


def score_candidate(path: Path, root: Path, key: str) -> tuple[int, int, int, str]:
    name = path.name.lower()
    rel = str(path.relative_to(root)) if root.exists() and path.is_relative_to(root) else str(path)
    size = path.stat().st_size if path.is_file() else 0
    score = 0
    if "il" in name or "il" in rel.lower():
        score += 1000
    if "model" in name:
        score += 500
    if "pytorch_model" in name:
        score += 250
    if path.suffix == ".safetensors":
        score += 100
    if key.startswith("recogdrive") and path.suffix in {".ckpt", ".pth", ".pt"}:
        score += 75
    plausible = 1 if size >= 1024 * 1024 else 0
    return (score, plausible, size, rel)


def select_candidate(key: str, local_dir: Path, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    return max(candidates, key=lambda path: score_candidate(path, local_dir, key))


def resolve_entry(key: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    local_dir = Path(str(spec.get("local_dir", ""))).expanduser()
    candidates = candidate_files(local_dir)
    selected = select_candidate(key, local_dir, candidates)
    return {
        "key": key,
        "hf_repo": spec.get("hf_repo"),
        "modelscope_repo": spec.get("modelscope_repo"),
        "local_dir": str(local_dir),
        "exists": local_dir.exists(),
        "candidate_count": len(candidates),
        "candidates": [
            {"path": str(path), "size_bytes": path.stat().st_size, "score": score_candidate(path, local_dir, key)[0]}
            for path in candidates
        ],
        "selected_path": str(selected) if selected is not None else None,
        "selected_size_bytes": selected.stat().st_size if selected is not None else None,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.weights_config)
    records = {key: resolve_entry(key, spec or {}) for key, spec in config.items() if key in OUTPUT_KEYS}
    output: Dict[str, Any] = {
        "weights_config": str(args.weights_config),
        "records": records,
    }
    for key, output_key in OUTPUT_KEYS.items():
        output[output_key] = records.get(key, {}).get("selected_path")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Resolved weights written to {args.output}")
        for key, record in records.items():
            print(f"\n{key}: {record['local_dir']}")
            if not record["exists"]:
                print("  local dir missing")
            if not record["candidates"]:
                print("  no checkpoint candidates found")
            for item in record["candidates"]:
                selected = " <= SELECTED" if item["path"] == record["selected_path"] else ""
                print(f"  - {item['path']} ({item['size_bytes']} bytes, score={item['score']}){selected}")
            print(f"  selected: {record['selected_path']}")
    missing = [key for key, record in records.items() if not record.get("selected_path")]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
