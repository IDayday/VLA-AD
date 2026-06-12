from __future__ import annotations

from typing import Any, Dict

import torch


TWO_EXPERT_PROMPT_VERSION = "two_expert_slot_prompt_v1"


def _format_number(value: float, decimal_places: int = 2) -> str:
    rounded = round(float(value), decimal_places)
    return f"{rounded:+.{decimal_places}f}" if abs(rounded) > 1e-2 else "0.0"


def build_minimal_two_expert_prompt() -> str:
    return (
        "<image>\nAs an autonomous driving system, complete an autonomous driving trajectory prediction task.\n"
        "Use front camera visual perception and output 8 future trajectory points in [PT, ...] format."
    )


def build_two_expert_prompt(sample: Dict[str, Any], *, allow_minimal_prompt: bool = False) -> str:
    if "history_trajectory" not in sample or "high_command_one_hot" not in sample:
        if allow_minimal_prompt:
            return build_minimal_two_expert_prompt()
        missing = [key for key in ("history_trajectory", "high_command_one_hot") if key not in sample]
        raise KeyError(f"two_expert prompt requires {missing}; pass allow_minimal_prompt=True only for smoke/dev.")

    history = torch.as_tensor(sample["history_trajectory"]).float()
    command = torch.as_tensor(sample["high_command_one_hot"]).float()
    if history.ndim != 2 or history.shape[0] < 4 or history.shape[-1] < 3:
        raise ValueError(f"history_trajectory must have shape [>=4,>=3], got {tuple(history.shape)}.")
    if command.numel() < 3:
        raise ValueError(f"high_command_one_hot must contain at least 3 values, got {tuple(command.shape)}.")

    command_names = ["turn left", "go straight", "turn right"]
    command_str = command_names[int(torch.argmax(command.view(-1)[:3]).item())]
    recent_history = history[-4:]
    history_str = " ".join(
        f"   - t-{3 - i}: ({_format_number(recent_history[i, 0].item())}, "
        f"{_format_number(recent_history[i, 1].item())}, {_format_number(recent_history[i, 2].item())})"
        for i in range(4)
    )
    return (
        "<image>\nAs an autonomous driving system, complete an autonomous driving trajectory prediction task based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]\n"
        "Output requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
