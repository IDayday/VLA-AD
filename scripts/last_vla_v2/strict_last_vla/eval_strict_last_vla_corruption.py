#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


VALID_MODES = {"zero_h_dyn", "zero_h_geo", "zero_h_plan", "zero_all_latent", "raw_vlm_only"}


def parse_modes(raw: str) -> List[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in VALID_MODES]
    if invalid:
        raise ValueError(f"Unknown strict latent corruption modes: {invalid}. Valid modes: {sorted(VALID_MODES)}")
    return modes


def corruption_overrides(mode: str) -> Dict[str, str]:
    if mode == "zero_h_dyn":
        return {"last_vla_h_dyn": "zero"}
    if mode == "zero_h_geo":
        return {"last_vla_h_geo": "zero"}
    if mode == "zero_h_plan":
        return {"last_vla_h_plan": "zero"}
    if mode == "zero_all_latent":
        return {"last_vla_h_dyn": "zero", "last_vla_h_geo": "zero", "last_vla_h_plan": "zero"}
    if mode == "raw_vlm_only":
        return {"last_vla_h_dyn": "drop", "last_vla_h_geo": "drop", "last_vla_h_plan": "drop"}
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare strict LaST-VLA latent-slot corruption eval plan.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", default="zero_h_dyn,zero_h_geo,zero_h_plan,zero_all_latent,raw_vlm_only")
    parser.add_argument("--write-plan-only", action="store_true")
    args = parser.parse_args()

    modes = parse_modes(args.modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "checkpoint": str(args.checkpoint),
        "cache_root": str(args.cache_root),
        "modes": [{"mode": mode, "latent_overrides": corruption_overrides(mode)} for mode in modes],
        "strict_latent_keys": ["last_vla_h_dyn", "last_vla_h_geo", "last_vla_h_plan"],
        "note": "This plan targets VLM-side latent slots only; it does not call the legacy action-side CoT corruption path.",
    }
    (args.output_dir / "strict_latent_corruption_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.write_plan_only:
        return 0
    raise NotImplementedError(
        "Full PDM corruption execution needs the NAVSIM evaluator wired to a corrupted strict latent cache view. "
        "The corruption plan was written; use it to materialize zero/drop latent cache variants before full eval."
    )


if __name__ == "__main__":
    raise SystemExit(main())
