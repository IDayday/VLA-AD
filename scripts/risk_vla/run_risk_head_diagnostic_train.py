#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}.")
        key, value = item.split("=", 1)
        config[key] = parse_value(value)
    return config


def diagnostic_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "use_bit_drive": False,
        "use_risk_vla": True,
        "risk_vla_num_classes": 6,
        "risk_vla_router_mode": "independent",
        "risk_vla_use_bit_summary": True,
        "risk_vla_use_oracle_router": False,
        "risk_vla_strategy_token_scale": 0.0,
        "risk_vla_horizon_residual_scale": 0.0,
        "risk_vla_risk_loss_weight": float(args.risk_loss_weight),
        "risk_vla_focal_loss_weight": 0.0,
        "risk_vla_strategy_entropy_weight": 0.0,
        "risk_vla_detach_risk_for_strategy": True,
        "risk_vla_log_diagnostics": True,
        "sampling_method": "ddim",
        "num_inference_steps": 5,
        "risk_vla_stage4_num_workers": int(args.num_workers),
        "risk_vla_stage4_limit_val_batches": args.limit_val_batches,
    }
    return apply_overrides(cfg, args.extra_override or [])


def build_train_command(args: argparse.Namespace, snapshot_path: Path) -> List[str]:
    command: List[str] = [
        sys.executable,
        "scripts/train_bit_drive_chunked.py",
        "--config",
        str(snapshot_path),
        "--chunk-cache-dir",
        str(args.cache_path),
        "--output-dir",
        str(args.output_dir / "train"),
        "--epochs-per-chunk",
        str(args.max_epochs),
    ]
    if args.checkpoint_path:
        command.extend(["--base-il-checkpoint", str(args.checkpoint_path)])
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.limit_train_batches is not None:
        command.extend(["--num-steps", str(args.limit_train_batches)])
    return command


def write_artifacts(args: argparse.Namespace) -> tuple[Path, Path, List[str]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = diagnostic_config(args)
    snapshot_path = args.output_dir / "risk_vla_diagnostic_config_snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    command = build_train_command(args, snapshot_path)
    command_path = args.output_dir / "resolved_train_command.sh"
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n", encoding="utf-8")
    return snapshot_path, command_path, command


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or run the RISK-VLA risk-head diagnostic training command.")
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--master-port", type=int, default=29541)
    parser.add_argument("--risk-loss-weight", type=float, default=0.05)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.risk_loss_weight <= 0.0:
        raise ValueError("Diagnostic mode requires risk_vla_risk_loss_weight > 0.")
    snapshot_path, command_path, command = write_artifacts(args)
    print("RISK-VLA diagnostic mode: strategy conditioning is disabled with token/residual scales set to 0.")
    print(f"Config snapshot: {snapshot_path}")
    print(f"Resolved command: {command_path}")
    print(shlex.join(command))
    if not args.execute:
        print("Dry-run only. Pass --execute to launch training.")
        return 0
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
