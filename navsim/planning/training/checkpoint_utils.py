from typing import Dict

import torch
from torch import Tensor


_CHECKPOINT_EXCLUDED_PREFIXES = (
    "agent.model",
    "agent.action_head.old_policy",
    "agent.action_head.behavior_policy",
)


def _is_recogdrive_checkpoint_excluded_key(key: str) -> bool:
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in _CHECKPOINT_EXCLUDED_PREFIXES)


def _filter_recogdrive_checkpoint_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Skip frozen backbone/reference weights before Lightning serializes checkpoints."""
    return {key: value for key, value in state_dict.items() if not _is_recogdrive_checkpoint_excluded_key(key)}


def _load_recogdrive_checkpoint_state_dict(
    module: torch.nn.Module,
    state_dict: Dict[str, Tensor],
    *,
    strict: bool = True,
    assign: bool = False,
):
    """Load a filtered checkpoint while retaining strict checks for trainable state."""
    incompatible = torch.nn.Module.load_state_dict(module, state_dict, strict=False, assign=assign)
    missing = [key for key in incompatible.missing_keys if not _is_recogdrive_checkpoint_excluded_key(key)]
    unexpected = list(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        details = []
        if missing:
            details.append(f"Missing key(s): {missing}")
        if unexpected:
            details.append(f"Unexpected key(s): {unexpected}")
        raise RuntimeError(
            f"Error(s) in loading state_dict for {module.__class__.__name__}:\n\t"
            + "\n\t".join(details)
        )
    if strict:
        return incompatible.__class__([], [])
    return incompatible
