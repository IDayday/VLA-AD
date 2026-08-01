"""Validate a supplied evaluation checkpoint and emit an anonymous receipt."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import inspect_checkpoint, write_checkpoint_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint, expected_sha256=args.expected_sha256)
    write_checkpoint_report(report, args.output)
    print(report.artifact_id)


if __name__ == "__main__":
    main()
