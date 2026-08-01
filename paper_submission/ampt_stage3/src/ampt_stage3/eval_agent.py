"""ReCogDrive evaluation adapter with deterministic per-scene sampling."""

from __future__ import annotations

from typing import Any

import torch

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

from .seeding import seed_for_token


class AMPTReproductionAgent(ReCogDriveAgent):
    """Emit one trajectory using a seed independent of GPU/shard assignment."""

    def __init__(self, *args: Any, evaluation_seed: int = 20260726, **kwargs: Any) -> None:
        if int(evaluation_seed) < 0:
            raise ValueError("evaluation_seed must be non-negative")
        self.evaluation_seed = int(evaluation_seed)
        super().__init__(*args, **kwargs)

    def compute_trajectory(self, agent_input):
        token = str(getattr(agent_input, "token", ""))
        if not token:
            raise ValueError("deterministic evaluation requires AgentInput.token")
        scene_seed = seed_for_token(self.evaluation_seed, token)
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(scene_seed)
            if cuda_devices:
                torch.cuda.manual_seed(scene_seed)
            return super().compute_trajectory(agent_input)

    def name(self) -> str:
        return "AMPTReproductionAgent"
