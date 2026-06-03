#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.testing.smoke_recogdrive_expert_planner import _install_dependency_stubs  # noqa: E402

_install_dependency_stubs()


def _patch_agent_stubs() -> None:
    dataclasses_mod = sys.modules["navsim.common.dataclasses"]

    @dataclass
    class AgentInput:
        pass

    class SensorConfig:
        @staticmethod
        def build_all_sensors(include=None):
            return SensorConfig()

    dataclasses_mod.AgentInput = getattr(dataclasses_mod, "AgentInput", AgentInput)
    dataclasses_mod.SensorConfig = getattr(dataclasses_mod, "SensorConfig", SensorConfig)

    training_mod = sys.modules.setdefault(
        "navsim.planning.training.abstract_feature_target_builder",
        type(sys)("navsim.planning.training.abstract_feature_target_builder"),
    )

    class AbstractFeatureBuilder:
        pass

    class AbstractTargetBuilder:
        pass

    training_mod.AbstractFeatureBuilder = getattr(training_mod, "AbstractFeatureBuilder", AbstractFeatureBuilder)
    training_mod.AbstractTargetBuilder = getattr(training_mod, "AbstractTargetBuilder", AbstractTargetBuilder)


_patch_agent_stubs()

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent  # noqa: E402


def build_agent(use_risk_vla: bool, device: torch.device) -> ReCogDriveAgent:
    agent = ReCogDriveAgent(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path="",
        allow_random_init=True,
        cache_hidden_state=True,
        use_expert_features=False,
        use_last_rd=False,
        use_bit_drive=False,
        use_risk_vla=use_risk_vla,
        risk_vla_strategy_token_scale=0.25,
        risk_vla_horizon_residual_scale=0.25,
        risk_vla_risk_loss_weight=0.05,
    )
    agent.to(device)
    return agent


def make_features(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "last_hidden_state": torch.randn(batch_size, 5, 1536, device=device),
        "history_trajectory": torch.randn(batch_size, 4, 3, device=device),
        "status_feature": torch.randn(batch_size, 8, device=device),
        "high_command_one_hot": torch.eye(3, device=device)[torch.tensor([0, 2], device=device)[:batch_size]],
        "risk_labels": torch.randint(0, 2, (batch_size, 8, 6), device=device).float(),
        "pred_terminal_intent": torch.randn(batch_size, 3, device=device),
        "pred_path_intent": torch.randn(batch_size, 8, 3, device=device),
    }


def run_case(use_risk_vla: bool, device: torch.device) -> None:
    agent = build_agent(use_risk_vla, device)
    agent.train()
    features = make_features(2, device)
    targets = {"trajectory": torch.randn(2, 8, 3, device=device)}
    output = agent(features, targets=targets)
    if not torch.isfinite(output["loss"]):
        raise RuntimeError("training loss is not finite.")
    if use_risk_vla:
        if "risk_vla_risk_loss" not in output:
            raise RuntimeError("RISK-VLA loss key missing in agent training output.")
    elif "risk_vla_risk_loss" in output:
        raise RuntimeError("use_risk_vla=false unexpectedly produced RISK-VLA output keys.")

    agent.eval()
    eval_features = {key: value for key, value in features.items() if key != "risk_labels"}
    with torch.no_grad():
        pred = agent(eval_features)
    if pred["pred_traj"].shape != (2, 8, 3):
        raise RuntimeError(f"pred_traj shape mismatch: {tuple(pred['pred_traj'].shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke test for config-gated RISK-VLA agent forward.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    torch.manual_seed(20260603)
    run_case(False, device)
    run_case(True, device)
    print("RISK-VLA agent smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
