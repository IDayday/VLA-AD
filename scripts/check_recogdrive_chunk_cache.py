#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import iter_index, load_metadata, load_sample, validate_sample_payload  # noqa: E402


def resolve_index_path(chunk_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = chunk_dir / path
        if candidate.is_file():
            return candidate
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ReCogDrive expert chunk cache.")
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--require-vlm-hidden", action="store_true")
    parser.add_argument("--require-targets", action="store_true")
    parser.add_argument("--expect-vlm-dim", type=int, default=None)
    parser.add_argument("--expect-jepa-shape", type=int, nargs=2, metavar=("TOKENS", "DIM"), default=None)
    parser.add_argument("--expect-vggt-shape", type=int, nargs=2, metavar=("TOKENS", "DIM"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def check_shape(payload: Dict[str, Any], key: str, expected: Optional[List[int]], path: Path) -> None:
    if expected is None or key not in payload:
        return
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} key {key} is {type(value).__name__}, expected tensor")
    if list(value.shape) != list(expected):
        raise ValueError(f"{path} key {key} shape {tuple(value.shape)} != {tuple(expected)}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{path} key {key} contains non-finite values")


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.chunk_dir)
    records = list(iter_index(args.chunk_dir))
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError(f"No samples found in {args.chunk_dir}")
    checked = 0
    errors: List[str] = []
    for record in records:
        path = resolve_index_path(args.chunk_dir, record["path"])
        try:
            payload = load_sample(path)
            validate_sample_payload(
                payload,
                require_jepa=metadata.get("contains_jepa", True),
                require_vggt=metadata.get("contains_vggt", True),
                require_targets=args.require_targets,
            )
            if (args.require_vlm_hidden or args.expect_vlm_dim is not None) and "last_hidden_state" not in payload:
                raise KeyError(f"{path} missing last_hidden_state")
            if args.expect_vlm_dim is not None and "last_hidden_state" in payload:
                hidden = payload["last_hidden_state"]
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[-1] != args.expect_vlm_dim:
                    raise ValueError(f"{path} last_hidden_state shape {getattr(hidden, 'shape', None)} has dim != {args.expect_vlm_dim}")
                if not torch.isfinite(hidden.float()).all():
                    raise ValueError(f"{path} last_hidden_state contains non-finite values")
            if args.expect_jepa_shape:
                check_shape(payload, "jepa_context_tokens", args.expect_jepa_shape, path)
                if args.require_targets:
                    check_shape(payload, "jepa_target_tokens", args.expect_jepa_shape, path)
            if args.expect_vggt_shape:
                check_shape(payload, "vggt_context_tokens", args.expect_vggt_shape, path)
                if args.require_targets:
                    check_shape(payload, "vggt_target_tokens", args.expect_vggt_shape, path)
            checked += 1
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            raise
    payload = {"chunk_dir": str(args.chunk_dir), "checked": checked, "metadata": metadata, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# ReCogDrive Chunk Cache Check", "", f"Chunk: `{args.chunk_dir}`", f"Checked samples: {checked}", f"Errors: {len(errors)}", "", "## Metadata", "", "```json", json.dumps(metadata, indent=2, sort_keys=True), "```", ""]
        args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
