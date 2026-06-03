#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _path_or_none(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return str(path) if path.is_file() or path.is_dir() else None


def _pdm_entry(csv: Optional[str], split: str, use_for_training: bool) -> Optional[Dict[str, Any]]:
    resolved = _path_or_none(csv)
    if resolved is None:
        return None
    return {"csv": resolved, "split": split, "use_for_training": bool(use_for_training)}


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}
    trainval: Dict[str, Any] = {}
    a0 = _pdm_entry(args.a0_pdm_csv, args.analysis_split, False)
    b3 = _pdm_entry(args.b3_pdm_csv, args.analysis_split, False)
    train = _pdm_entry(args.train_pdm_csv, "navtrain", True)
    val = _pdm_entry(args.val_pdm_csv, "navval", True)
    if a0:
        analysis["A0_base"] = a0
    if b3:
        analysis["B3_direct_bit"] = b3
    if train:
        trainval["train"] = train
    if val:
        trainval["val"] = val
    return {
        "round": "risk_vla_round1",
        "analysis_pdm": analysis,
        "trainval_pdm": trainval,
        "cache": {
            "source_chunk_cache_dir": str(Path(args.source_chunk_cache_dir).expanduser()) if args.source_chunk_cache_dir else None,
            "overlay_output_dir": str(Path(args.overlay_output_dir).expanduser()) if args.overlay_output_dir else None,
        },
        "checkpoints": {
            "base_or_bit_checkpoint": str(Path(args.checkpoint).expanduser()) if args.checkpoint else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an explicit RISK-VLA Round 1 input manifest from provided paths.")
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--a0-pdm-csv", default=None)
    parser.add_argument("--b3-pdm-csv", default=None)
    parser.add_argument("--train-pdm-csv", default=None)
    parser.add_argument("--val-pdm-csv", default=None)
    parser.add_argument("--source-chunk-cache-dir", default=None)
    parser.add_argument("--overlay-output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--analysis-split", default="analysis_only")
    args = parser.parse_args()
    manifest = build_manifest(args)
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.output_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
