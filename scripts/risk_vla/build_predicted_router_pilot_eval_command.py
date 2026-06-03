#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.risk_vla.run_risk_head_diagnostic_train import apply_overrides


def eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if args.base_config and args.base_config.is_file():
        cfg.update(yaml.safe_load(args.base_config.read_text(encoding="utf-8")) or {})
    cfg.update(
        {
            "use_risk_vla": True,
            "risk_vla_use_oracle_router": False,
            "risk_vla_strategy_token_scale": 0.25,
            "risk_vla_horizon_residual_scale": 0.25,
            "risk_vla_risk_loss_weight": float(args.risk_loss_weight),
        }
    )
    return apply_overrides(cfg, args.extra_override or [])


def build_eval_command(args: argparse.Namespace, snapshot_path: Path) -> List[str]:
    command = [
        sys.executable,
        "scripts/eval_bit_drive_pdm.py",
        "--config",
        str(snapshot_path),
        "--checkpoint",
        str(args.checkpoint),
        "--split",
        args.split,
        "--data-root",
        str(args.navsim_root),
        "--output-dir",
        str(args.output_dir / "predicted_router_pilot_eval"),
        "--precision",
        args.precision,
    ]
    if args.chunk_cache_dir:
        command.extend(["--chunk-cache-dir", str(args.chunk_cache_dir)])
    if args.chunk_cache_root:
        command.extend(["--chunk-cache-root", str(args.chunk_cache_root)])
    if args.metric_cache_dir:
        command.extend(["--metric-cache-dir", str(args.metric_cache_dir)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    return command


def write_artifacts(args: argparse.Namespace) -> tuple[Path, Path, List[str]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / "predicted_router_pilot_config_snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(eval_config(args), sort_keys=True), encoding="utf-8")
    command = build_eval_command(args, snapshot_path)
    command_path = args.output_dir / "resolved_predicted_router_eval_command.sh"
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n", encoding="utf-8")
    return snapshot_path, command_path, command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run the predicted-router pilot PDM command.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--split", default="navval")
    parser.add_argument("--navsim-root", type=Path, default=Path("/mnt/navsim"))
    parser.add_argument("--chunk-cache-dir", type=Path, default=None)
    parser.add_argument("--chunk-cache-root", type=Path, default=None)
    parser.add_argument("--metric-cache-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--risk-loss-weight", type=float, default=0.0)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path, command_path, command = write_artifacts(args)
    print("Predicted-router pilot uses learned risk predictions; oracle routing is disabled.")
    print(f"Config snapshot: {snapshot_path}")
    print(f"Resolved command: {command_path}")
    print(shlex.join(command))
    if not args.execute:
        print("Dry-run only. Pass --execute to launch pilot evaluation.")
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
