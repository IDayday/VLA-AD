#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine support candidates with a scorer, then require true evaluator verification.")
    parser.add_argument("--candidate_archive_dir", required=True)
    parser.add_argument("--scorer_ckpt", required=True)
    parser.add_argument("--output_archive_dir", required=True)
    parser.add_argument("--true_eval_required", action="store_true", default=True)
    args = parser.parse_args()
    raise RuntimeError(
        "refine_support_with_scorer requires a NAVSIM true evaluator context in this process. "
        "Scorer predictions are never written as final labels; run this through the evaluator-enabled mining pipeline."
    )


if __name__ == "__main__":
    main()
