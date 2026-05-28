#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

REQUIRED_KEYS = ["recogdrive_2b_il", "recogdrive_vlm_2b", "vjepa2", "vggt_1b"]
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".py", ".txt", ".model"}
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt", ".pth", ".model"}


def is_plausible_weight_file(path: Path) -> bool:
    lower = path.name.lower()
    if ".cache" in path.parts or lower.endswith(".incomplete"):
        return False
    if lower in {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}:
        return False
    if path.suffix.lower() not in WEIGHT_SUFFIXES:
        return False
    if path.suffix.lower() == ".model":
        return True
    return path.stat().st_size >= 1024 * 1024


def has_file_with_suffix(root: Path, suffixes: set[str]) -> bool:
    if not root.exists():
        return False
    if suffixes == WEIGHT_SUFFIXES:
        return any(path.is_file() and is_plausible_weight_file(path) for path in root.rglob("*"))
    return any(path.is_file() and ".cache" not in path.parts and path.suffix.lower() in suffixes for path in root.rglob("*"))


def load_config(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"weights config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"weights config must be a mapping, got {type(data).__name__}")
    return data


def local_dir_for(spec: Dict[str, Any], root: Path) -> Path:
    configured = Path(str(spec["local_dir"])).expanduser()
    if not configured.is_absolute():
        return root / configured
    parts = configured.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        return root.joinpath(*parts[idx + 1 :])
    return configured


def check_dir(name: str, path: Path) -> Dict[str, object]:
    record: Dict[str, object] = {"name": name, "path": str(path.resolve()), "ok": False}
    if not path.is_dir():
        record["error"] = "missing directory"
        return record
    has_config = has_file_with_suffix(path, CONFIG_SUFFIXES)
    has_weights = has_file_with_suffix(path, WEIGHT_SUFFIXES)
    requires_config = name != "recogdrive_2b_il"
    record["has_config"] = has_config
    record["has_weights"] = has_weights
    record["requires_config"] = requires_config
    if not has_weights:
        record["error"] = "no weight files found (.safetensors/.bin/.pt/.ckpt/.pth/.model)"
    elif requires_config and not has_config:
        record["error"] = "no config/tokenizer/source files found"
    else:
        record["ok"] = True
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check required local ReCogDrive expert-token weights.")
    parser.add_argument("--root", type=Path, default=Path("/mnt/project/VLA-AD/checkpoints"))
    parser.add_argument("--weights-config", type=Path, default=Path("configs/weights.yaml"))
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.weights_config)
    records: List[Dict[str, object]] = []
    for key in REQUIRED_KEYS:
        if key not in config:
            records.append({"name": key, "path": "<missing config entry>", "ok": False, "error": "missing config entry"})
            continue
        records.append(check_dir(key, local_dir_for(config[key] or {}, args.root)))
    payload = {"root": str(args.root.resolve()), "weights_config": str(args.weights_config), "records": records}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Weight root: {args.root.resolve()}")
        print("Required training paths:")
        for record in records:
            status = "OK" if record["ok"] else "MISSING"
            print(f"  {record['name']}: {record['path']} [{status}]")
            if not record["ok"]:
                print(f"    error: {record.get('error')}")
    failed = [record for record in records if not record["ok"]]
    if failed:
        raise SystemExit("Required weights are missing or incomplete; run scripts/download_required_weights.py first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
