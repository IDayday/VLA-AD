#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve LaST-RD Hydra experiment config without training.")
    parser.add_argument("--experiment", required=True, choices=("last_rd_stage1_5", "last_rd_progressive_sft"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "last_rd_hydra_config_check.json")
    return parser.parse_args()


def selected_report(cfg: Any) -> Dict[str, Any]:
    agent = cfg.agent
    trainer_params = cfg.trainer.params
    dataloader_params = cfg.dataloader.params
    return {
        "agent.use_last_rd": bool(agent.use_last_rd),
        "agent.last_rd_stage": str(agent.last_rd_stage),
        "agent.use_expert_features": bool(agent.use_expert_features),
        "agent.allow_expert_target_features": bool(agent.allow_expert_target_features),
        "loss_weights": {
            "diffusion_loss_weight": float(agent.diffusion_loss_weight),
            "future_jepa_loss_weight": float(agent.future_jepa_loss_weight),
            "vggt_geometry_loss_weight": float(agent.vggt_geometry_loss_weight),
            "coarse_traj_loss_weight": float(agent.coarse_traj_loss_weight),
            "coarse_heading_loss_weight": float(agent.coarse_heading_loss_weight),
            "risk_loss_weight": float(agent.risk_loss_weight),
            "policy_kd_loss_weight": float(agent.policy_kd_loss_weight),
            "policy_kd_mode": str(agent.policy_kd_mode),
        },
        "trainer_params": OmegaConf.to_container(trainer_params, resolve=True),
        "dataloader_batch_size": int(dataloader_params.batch_size),
    }


def main() -> int:
    args = parse_args()
    result: Dict[str, Any] = {"experiment": args.experiment, "pass": False}
    try:
        with initialize_config_module(config_module="navsim.planning.script.config.training", version_base=None):
            cfg = compose(
                config_name="default_training",
                overrides=[
                    "train_test_split=navtrain",
                    f"+experiment={args.experiment}",
                    "output_dir=reports/last_rd_hydra_dryrun",
                    "cache_path=/tmp/last_rd_hydra_dryrun_cache",
                ],
            )
        result.update(selected_report(cfg))
        expected_stage = "stage1_5" if args.experiment == "last_rd_stage1_5" else "progressive_sft"
        checks = [
            result["agent.use_last_rd"] is True,
            result["agent.last_rd_stage"] == expected_stage,
            result["agent.allow_expert_target_features"] is True,
        ]
        if args.experiment == "last_rd_stage1_5":
            checks.extend([
                result["agent.use_expert_features"] is False,
                result["loss_weights"]["diffusion_loss_weight"] == 0.0,
                result["loss_weights"]["risk_loss_weight"] == 0.0,
            ])
        result["pass"] = bool(all(checks))
        if not result["pass"]:
            result["error"] = "resolved config did not satisfy LaST-RD invariants"
    except Exception as exc:
        result["error"] = repr(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
