#!/usr/bin/env python3
"""Validate a local Hugging Face checkpoint before training or inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--require-tokenizer",
        action="store_true",
        help="Require tokenizer/processor files needed by local inference.",
    )
    parser.add_argument(
        "--transformers-smoke",
        action="store_true",
        help="Load AutoConfig and AutoProcessor with local_files_only=True.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def load_index(model_path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = model_path / name
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, None


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    errors: list[str] = []
    warnings: list[str] = []

    if not model_path.is_dir():
        raise SystemExit(f"MODEL_PATH is not a directory: {model_path}")

    aria2_files = sorted(str(path) for path in model_path.glob("*.aria2"))
    if aria2_files:
        errors.append(f"found incomplete .aria2 files: {aria2_files[:10]}")

    required_files = ["config.json"]
    if args.require_tokenizer:
        required_files.extend(["tokenizer_config.json", "preprocessor_config.json"])
    missing_required = [name for name in required_files if not (model_path / name).is_file()]
    if missing_required:
        errors.append(f"missing required file(s): {missing_required}")

    index_path, index = load_index(model_path)
    weight_files: list[str] = []
    if index is not None:
        weight_map = index.get("weight_map") or {}
        if not weight_map:
            errors.append(f"empty weight_map in {index_path}")
        weight_files = sorted(set(str(name) for name in weight_map.values()))
        missing_weights = [name for name in weight_files if not (model_path / name).is_file()]
        empty_weights = [name for name in weight_files if (model_path / name).is_file() and file_size(model_path / name) == 0]
        if missing_weights:
            errors.append(f"missing weight shard(s): {missing_weights[:10]}")
        if empty_weights:
            errors.append(f"empty weight shard(s): {empty_weights[:10]}")
    else:
        local_weights = sorted(path.name for path in model_path.glob("*.safetensors"))
        local_weights += sorted(path.name for path in model_path.glob("pytorch_model*.bin"))
        weight_files = local_weights
        if not local_weights:
            errors.append("no model.safetensors.index.json, pytorch_model.bin.index.json, *.safetensors, or pytorch_model*.bin found")

    if args.require_tokenizer:
        tokenizer_candidates = [
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        ]
        present_tokenizer_files = [name for name in tokenizer_candidates if (model_path / name).is_file()]
        if "tokenizer.json" not in present_tokenizer_files and not {"vocab.json", "merges.txt"}.issubset(
            present_tokenizer_files
        ):
            errors.append("tokenizer files are incomplete: need tokenizer.json or vocab.json+merges.txt")

    transformers_smoke: dict[str, Any] = {"requested": args.transformers_smoke, "ok": False}
    if args.transformers_smoke and not errors:
        try:
            from transformers import AutoConfig, AutoProcessor

            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            transformers_smoke.update(
                {
                    "ok": True,
                    "model_type": getattr(cfg, "model_type", None),
                    "processor_class": processor.__class__.__name__,
                }
            )
        except Exception as exc:  # pragma: no cover - environment-dependent message
            errors.append(f"transformers local load failed: {type(exc).__name__}: {exc}")

    report = {
        "model_path": str(model_path),
        "aria2_files": aria2_files,
        "required_files": required_files,
        "missing_required": missing_required,
        "index_path": str(index_path) if index_path else None,
        "weight_files": weight_files,
        "weight_file_count": len(weight_files),
        "transformers_smoke": transformers_smoke,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
