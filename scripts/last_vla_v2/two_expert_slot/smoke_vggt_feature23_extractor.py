#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_cache import load_sample, write_json  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.build_vggt_feature23_cache import (  # noqa: E402
    extract_feature23_tokens,
    load_vggt_image,
    load_vggt_model,
)
from scripts.last_vla_v2.two_expert_slot.smoke_internvl_soft_slots import decode_path_tensor  # noqa: E402
from scripts.last_vla_v2.two_expert_slot.two_expert_cache_utils import iter_indexed_records  # noqa: E402


def resolve_sample_image(args: argparse.Namespace) -> Optional[Path]:
    if args.image_path is not None:
        return Path(args.image_path)
    if args.base_cache_root is None:
        return None
    for _, sample_path, record in iter_indexed_records(args.base_cache_root, max_records=args.max_records):
        if args.sample_token and str(record.get("sample_token") or sample_path.stem) != str(args.sample_token):
            continue
        sample = load_sample(sample_path)
        if "image_path_tensor" not in sample:
            raise KeyError(f"Base cache sample {sample_path} missing image_path_tensor.")
        return Path(decode_path_tensor(sample["image_path_tensor"]))
    raise FileNotFoundError(f"No matching image sample found under {args.base_cache_root}.")


def run_smoke(args: argparse.Namespace) -> Dict[str, object]:
    image_path = resolve_sample_image(args)
    if image_path is None:
        raise ValueError("Provide --image-path or --base-cache-root for VGGT smoke.")
    model = load_vggt_model(args)
    image = load_vggt_image(image_path, int(args.image_size))
    tokens, metadata = extract_feature23_tokens(
        model,
        image,
        layer_index=int(args.feature_layer_index),
        pack_tokens=int(args.pack_tokens),
        device=str(args.device),
        precision=str(args.precision),
    )
    return {
        "ok": True,
        "status": "passed",
        "image_path": str(image_path),
        "layer_name": metadata["layer_name"],
        "feature_shape": metadata["original_shape"],
        "packed_shape": list(tokens.shape),
        "feature_dim": int(tokens.shape[-1]),
        "strict_geometry_teacher": bool(metadata["strict_geometry_teacher"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test VGGT Feature(23) extraction.")
    parser.add_argument("--vggt-model-path", type=Path, default=os.getenv("VGGT_MODEL_PATH"))
    parser.add_argument("--vggt-model-class", default=os.getenv("VGGT_MODEL_CLASS"))
    parser.add_argument("--image-path", type=Path, default=os.getenv("IMAGE_PATH"))
    parser.add_argument("--base-cache-root", type=Path, default=os.getenv("BASE_CACHE_ROOT"))
    parser.add_argument("--sample-token", default=os.getenv("SAMPLE_TOKEN"))
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--feature-layer-index", type=int, default=23)
    parser.add_argument("--pack-tokens", type=int, default=12)
    parser.add_argument("--device", default=os.getenv("DEVICE", "cuda"))
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default=os.getenv("PRECISION", "bf16"))
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def emit(args: argparse.Namespace, payload: Dict[str, object], code: int) -> int:
    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


def main() -> int:
    args = parse_args()
    if os.getenv("RUN_SMOKE", "0") != "1":
        return emit(args, {"ok": False, "status": "not_run", "reason": "Set RUN_SMOKE=1 to load VGGT."}, 0)
    if args.vggt_model_path is None:
        return emit(args, {"ok": False, "status": "not_run", "reason": "VGGT_MODEL_PATH or --vggt-model-path is required."}, 2)
    try:
        return emit(args, run_smoke(args), 0)
    except Exception as exc:
        return emit(args, {"ok": False, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}, 1)


if __name__ == "__main__":
    raise SystemExit(main())
