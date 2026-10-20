from dataclasses import dataclass, field
from typing import Any, Dict
import torch


@dataclass
class TransitionStats:
    mean: torch.Tensor
    std: torch.Tensor
    x0: torch.Tensor
    index: int
    timestep: int
    kernel: str
    noise_clip: float
    deterministic: bool


@dataclass
class NoisePlan:
    initial: torch.Tensor
    # Absolute execution index, even when used by a suffix starting after index i.
    steps: torch.Tensor


@dataclass
class Rollin:
    states: torch.Tensor
    trajectory: torch.Tensor
    student_version: int
    parameter_versions: tuple
    source: int
    noise: NoisePlan


@dataclass
class TargetRecord:
    scene_token: str
    student_version: int
    reference_version: int
    index: int
    timestep: int
    target: torch.Tensor
    proposed: torch.Tensor
    reference_mean: torch.Tensor
    actual_std: torch.Tensor
    accepted: bool
    reason: str
    statistics: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
