#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT = Path("/mnt/project/skill/stable-gpu-job-launch/scripts/gpu_progress_dashboard.py")


if __name__ == "__main__":
    if "--latest-glob" not in sys.argv and "--out-root" not in sys.argv:
        sys.argv.extend(["--latest-glob", "/mnt/project/VLA-AD/outputs/last_vla_v2/direct_text_navtest_eval*"])
    runpy.run_path(str(SCRIPT), run_name="__main__")
