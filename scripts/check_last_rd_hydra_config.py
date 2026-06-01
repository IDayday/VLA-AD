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
        "agent._target_": str(agent.get("_target_", "")),
        "agent.use_last_rd": bool(agent.use_last_rd),
        "agent.last_rd_stage": str(agent.last_rd_stage),
        "agent.use_expert_features": bool(agent.use_expert_features),
        "agent.allow_expert_target_features": bool(agent.allow_expert_target_features),
        "agent.num_jepa_tokens": int(agent.get("num_jepa_tokens", -1)),
        "agent.num_vggt_tokens": int(agent.get("num_vggt_tokens", -1)),
        "agent.jepa_dim": int(agent.get("jepa_dim", -1)),
        "agent.vggt_dim": int(agent.get("vggt_dim", -1)),
        "agent.freeze_base_action_head": bool(agent.get("freeze_base_action_head", False)),
        "agent.train_expert_only": bool(agent.get("train_expert_only", False)),
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
            result["agent._target_"] == "navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent",
            result["agent.use_last_rd"] is True,
            result["agent.last_rd_stage"] == expected_stage,
            result["agent.allow_expert_target_features"] is True,
            result["agent.num_jepa_tokens"] == 12,
            result["agent.num_vggt_tokens"] == 12,
            result["agent.jepa_dim"] == 1024,
            result["agent.vggt_dim"] == 2048,
            result["dataloader_batch_size"] == 16,
        ]
        if args.experiment == "last_rd_stage1_5":
            checks.extend([
                result["agent.use_expert_features"] is False,
                result["loss_weights"]["diffusion_loss_weight"] == 0.0,
                result["loss_weights"]["future_jepa_loss_weight"] == 0.30,
                result["loss_weights"]["risk_loss_weight"] == 0.0,
                result["agent.freeze_base_action_head"] is True,
                result["agent.train_expert_only"] is True,
                result["trainer_params"].get("strategy") == "ddp_find_unused_parameters_true",
                int(result["trainer_params"].get("max_epochs", -1)) == 20,
            ])
        else:
            checks.extend([
                result["agent.use_expert_features"] is True,
                result["loss_weights"]["diffusion_loss_weight"] == 1.0,
                result["loss_weights"]["policy_kd_loss_weight"] == 0.05,
                result["loss_weights"]["policy_kd_mode"] == "noise",
                int(result["trainer_params"].get("max_epochs", -1)) == 200,
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
