"""Materialize a uniform distribution over the valid slots in a PSI support index."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from navsim.agents.recogdrive.pareto_support import load_pareto_support_index

SOURCE_SHA256 = "b3009a61d93a0760288685660c59d950788fe98f1dab62a8de59a68ba3570498"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("use a new output path to preserve the original support index")
    source_hash = sha256(args.source)
    if source_hash != SOURCE_SHA256:
        parser.error("source checksum does not match the archived full PSI support index")
    support = load_pareto_support_index(args.source)
    mask = support["support_mask"].bool()
    counts = mask.sum(dim=1, keepdim=True)
    if len(counts) != 103_288 or (counts == 0).any():
        raise ValueError("Expected 103,288 support rows, each with at least one valid slot")
    support["support_weights"] = mask.float() / counts

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp")
    torch.save(support, temp)
    checked = load_pareto_support_index(temp)
    for key in ("support_mask", "support_weights", "support_trajectories", "support_scores"):
        if not torch.equal(checked[key], support[key]):
            raise ValueError(f"Support index did not round-trip: {key}")
    if checked["tokens"] != support["tokens"]:
        raise ValueError("Support index token order changed")
    temp.replace(args.output)
    print(json.dumps({
        "source_sha256": source_hash,
        "uniform_sha256": sha256(args.output),
        "rows_by_valid_candidate_count": torch.bincount(counts.flatten()).tolist(),
    }, indent=2))

if __name__ == "__main__":
    main()
