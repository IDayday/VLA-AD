#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PATHS = {
    "recogdrive_2b_il": "recogdrive/ReCogDrive-2B-IL",
    "recogdrive_vlm_2b": "recogdrive/ReCogDrive-VLM-2B",
    "vjepa2": "teachers/vjepa2-vitl-fpc64-256",
    "vggt_1b": "teachers/VGGT-1B",
    "recogdrive_2b_rl": "recogdrive/ReCogDrive-2B-RL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print canonical local weight paths.")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--format", choices=("shell", "yaml", "plain"), default="plain")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = {key: str((args.root / rel).resolve()) for key, rel in PATHS.items()}
    if args.format == "shell":
        for key, value in resolved.items():
            print(f"export {key.upper()}={value}")
    elif args.format == "yaml":
        for key, value in resolved.items():
            print(f"{key}: {value}")
    else:
        for key, value in resolved.items():
            print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
