#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pareto_support.mine_pareto_support import main as mine_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SG-FPS Round0 seed archives.")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args()
    sys.argv = [sys.argv[0], "--round", "0", *parsed.args]
    mine_main()


if __name__ == "__main__":
    main()
