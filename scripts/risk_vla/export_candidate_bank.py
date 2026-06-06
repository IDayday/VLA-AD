from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch

from navsim.agents.recogdrive.risk_vla.candidate_bank import CandidateBank, save_candidate_cache


def read_predictions(path: Path, max_samples: Optional[int] = None) -> tuple[list[str], torch.Tensor]:
    tokens = []
    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            token = str(row.get("sample_token") or row.get("token") or row.get("scene_token"))
            traj = row.get("trajectory") or row.get("pred_traj") or row.get("prediction")
            if token and traj is not None:
                tokens.append(token)
                trajectories.append(traj)
            if max_samples is not None and len(tokens) >= max_samples:
                break
    if not trajectories:
        raise RuntimeError(f"No trajectories found in {path}")
    return tokens, torch.tensor(trajectories, dtype=torch.float32)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export a RISK-VLA v2 candidate trajectory bank.")
    parser.add_argument("--base-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--bit-predictions-jsonl", type=Path, default=None)
    parser.add_argument("--risk-vla-predictions-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True))
        return 0
    tokens, base = read_predictions(args.base_predictions_jsonl, args.max_samples)
    bit = read_predictions(args.bit_predictions_jsonl, args.max_samples)[1] if args.bit_predictions_jsonl else None
    risk = read_predictions(args.risk_vla_predictions_jsonl, args.max_samples)[1] if args.risk_vla_predictions_jsonl else None
    trajectories, metadata = CandidateBank().generate(base, bit_trajectory=bit, risk_vla_trajectory=risk, k=args.k)
    save_candidate_cache(args.output_dir, tokens, trajectories, metadata)
    summary = {"num_samples": len(tokens), "num_candidates": trajectories.shape[1], "output_dir": str(args.output_dir), "metadata": metadata}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
