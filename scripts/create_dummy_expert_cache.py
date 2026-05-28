#!/usr/bin/env python3
"""Create a dummy JEPA/VGGT expert cache for computation smoke tests.

The generated tensors are deterministic fake features. They are useful for
checking cache loading and tensor plumbing before real teachers or NAVSIM data
are available, but they must not be used for real training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.recogdrive.expert_backends import build_dummy_expert_backend
from navsim.agents.recogdrive.recogdrive_features import DUMMY_EXPERT_CACHE_WARNING


def create_dummy_expert_cache(
    output_dir: Path,
    *,
    num_samples: int = 16,
    num_jepa_tokens: int = 12,
    num_vggt_tokens: int = 12,
    jepa_dim: int = 1024,
    vggt_dim: int = 2048,
    include_targets: bool = False,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = build_dummy_expert_backend(
        num_jepa_tokens=num_jepa_tokens,
        num_vggt_tokens=num_vggt_tokens,
        jepa_dim=jepa_dim,
        vggt_dim=vggt_dim,
        use_jepa=True,
        use_vggt=True,
        seed=seed,
    )

    index_records = []
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(num_samples):
        token = f"sample_{sample_idx:06d}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sample = backend.generate_single(sample_key=token, include_targets=include_targets)
        sample.update({"scene_token": token, "sample_token": token})
        path = samples_dir / f"{token}.pt"
        torch.save(sample, path)
        index_records.append({"scene_token": token, "sample_token": token, "path": str(path.relative_to(output_dir))})

    with (output_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in index_records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "version": "recogdrive2b_expert768_chunk_v1",
        "is_dummy": True,
        "contains_vlm_hidden": False,
        "contains_jepa": True,
        "contains_vggt": True,
        "target_tokens_are_train_only": True,
        "num_samples": num_samples,
        "num_jepa_tokens": num_jepa_tokens,
        "num_vggt_tokens": num_vggt_tokens,
        "jepa_dim": jepa_dim,
        "vggt_dim": vggt_dim,
        "include_targets": include_targets,
        "seed": seed,
        "warning": DUMMY_EXPERT_CACHE_WARNING,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-jepa-tokens", type=int, default=12)
    parser.add_argument("--num-vggt-tokens", type=int, default=12)
    parser.add_argument("--jepa-dim", type=int, default=1024)
    parser.add_argument("--vggt-dim", type=int, default=2048)
    parser.add_argument("--include-targets", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_dummy_expert_cache(
        args.output_dir,
        num_samples=args.num_samples,
        num_jepa_tokens=args.num_jepa_tokens,
        num_vggt_tokens=args.num_vggt_tokens,
        jepa_dim=args.jepa_dim,
        vggt_dim=args.vggt_dim,
        include_targets=args.include_targets,
        seed=args.seed,
    )
    print(f"Wrote {args.num_samples} dummy expert-cache samples to {args.output_dir}")
    print(DUMMY_EXPERT_CACHE_WARNING)


if __name__ == "__main__":
    main()
