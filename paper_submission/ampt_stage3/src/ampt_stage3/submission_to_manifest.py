"""Convert a trusted local NAVSIM submission to one-trajectory JSON."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    source = args.submission.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # Pickle is accepted only because this tool is explicitly for a locally
    # generated, trusted NAVSIM submission; never use it on downloaded input.
    with source.open("rb") as stream:
        payload = pickle.load(stream)
    predictions = payload.get("predictions") if isinstance(payload, Mapping) else None
    if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], Mapping):
        raise TypeError("submission must contain exactly one single-trajectory prediction mapping")
    json_rows = []
    for token, trajectory in sorted(predictions[0].items(), key=lambda item: str(item[0])):
        poses = np.asarray(getattr(trajectory, "poses", trajectory), dtype=np.float64)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError(f"prediction {token} must be finite [8,3]")
        json_rows.append({"sample_token": str(token), "pred_traj": poses.tolist()})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_rows, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
