#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_A4_MARKERS = (
    "jepa_projector",
    "vggt_projector",
    "jepa_adapter",
    "vggt_adapter",
    "jepa_alignment_head",
    "vggt_alignment_head",
    "branch_logits",
    "jepa_gate",
    "vggt_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a saved LaST-RD adapter checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_rd_adapter_audit.json")
    parser.add_argument("--allow-legacy", action="store_true")
    return parser.parse_args()


def load_state_dict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Checkpoint must be a dict or contain state_dict, got {type(checkpoint).__name__}")


def audit_checkpoint(path: Path, *, allow_legacy: bool = False) -> Dict[str, Any]:
    state = load_state_dict(path)
    keys = sorted(str(key) for key in state.keys())
    last_rd_keys = [key for key in keys if key.startswith("action_head.last_rd.") or key.startswith("last_rd.")]
    legacy_keys = [key for key in keys if any(marker in key for marker in LEGACY_A4_MARKERS)]
    warnings = []
    blockers = []
    if not last_rd_keys:
        blockers.append("checkpoint contains no action_head.last_rd.* or last_rd.* tensors")
    if legacy_keys and not allow_legacy:
        blockers.append("checkpoint contains legacy A4 expert tensors; pass --allow-legacy only for intentional mixed adapters")
        warnings.append("legacy_a4_tensors_present")
    elif legacy_keys:
        warnings.append("legacy_a4_tensors_present_allowed")
    result = {
        "checkpoint": str(path),
        "allow_legacy": bool(allow_legacy),
        "num_total_tensors": len(keys),
        "num_last_rd_tensors": len(last_rd_keys),
        "num_legacy_a4_tensors": len(legacy_keys),
        "examples": {
            "last_rd": last_rd_keys[:20],
            "legacy_a4": legacy_keys[:20],
        },
        "warnings": warnings,
        "blockers": blockers,
        "pass": not blockers,
    }
    return result


def main() -> int:
    args = parse_args()
    try:
        result = audit_checkpoint(args.checkpoint, allow_legacy=args.allow_legacy)
    except Exception as exc:
        result = {
            "checkpoint": str(args.checkpoint),
            "allow_legacy": bool(args.allow_legacy),
            "pass": False,
            "blockers": [repr(exc)],
            "warnings": [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
