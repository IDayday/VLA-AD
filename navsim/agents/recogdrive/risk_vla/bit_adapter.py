from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch


TERMINAL_ALIASES = (
    "bit_terminal",
    "terminal_intent",
    "pred_terminal_intent",
)
PATH_ALIASES = (
    "bit_path",
    "path_intent",
    "bit_pred_path",
    "pred_path_intent",
)


def _get_value(action_input: Any, key: str) -> Any:
    if action_input is None:
        return None
    if isinstance(action_input, dict):
        return action_input.get(key)
    if hasattr(action_input, key):
        return getattr(action_input, key)
    try:
        return action_input[key]
    except Exception:
        return None


def _find_tensor(action_input: Any, aliases: tuple[str, ...]) -> tuple[Optional[str], Optional[torch.Tensor]]:
    for key in aliases:
        value = _get_value(action_input, key)
        if isinstance(value, torch.Tensor):
            return key, value
    return None, None


def _handle_shape_error(message: str, *, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning)


def _validate_terminal(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    action_dim: int,
    reference: torch.Tensor,
    source: str,
    strict: bool,
) -> Optional[torch.Tensor]:
    if tensor.ndim != 2 or tensor.shape[0] != batch_size or tensor.shape[1] != action_dim:
        _handle_shape_error(
            f"BiT terminal alias {source!r} must have shape "
            f"[{batch_size}, {action_dim}], got {tuple(tensor.shape)}.",
            strict=strict,
        )
        return None
    return tensor.to(device=reference.device, dtype=reference.dtype)


def _validate_path(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    reference: torch.Tensor,
    source: str,
    strict: bool,
) -> Optional[torch.Tensor]:
    expected = (batch_size, action_horizon, action_dim)
    if tensor.ndim != 3 or tuple(tensor.shape) != expected:
        _handle_shape_error(
            f"BiT path alias {source!r} must have shape {expected}, got {tuple(tensor.shape)}.",
            strict=strict,
        )
        return None
    return tensor.to(device=reference.device, dtype=reference.dtype)


def extract_bit_intents(
    action_input: Any,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    reference: torch.Tensor,
    use_bit_summary: bool = True,
    strict: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Normalize optional BiT path/terminal aliases for RISK-VLA.

    RISK-VLA never requires BiT outputs. When enabled, BiT is treated only as a
    path/terminal intent strategy, not as the core planning algorithm.
    """

    zero = reference.detach().new_tensor(0.0)
    one = reference.detach().new_tensor(1.0)
    diagnostics: Dict[str, torch.Tensor] = {
        "bit_adapter_enabled": one if use_bit_summary else zero,
        "bit_adapter_has_terminal": zero,
        "bit_adapter_has_path": zero,
        "bit_adapter_terminal_source_code": zero,
        "bit_adapter_path_source_code": zero,
    }
    if not use_bit_summary:
        return None, None, diagnostics

    terminal_source, terminal_tensor = _find_tensor(action_input, TERMINAL_ALIASES)
    path_source, path_tensor = _find_tensor(action_input, PATH_ALIASES)

    terminal: Optional[torch.Tensor] = None
    if terminal_tensor is not None and terminal_source is not None:
        terminal = _validate_terminal(
            terminal_tensor,
            batch_size=batch_size,
            action_dim=action_dim,
            reference=reference,
            source=terminal_source,
            strict=strict,
        )
        if terminal is not None:
            diagnostics["bit_adapter_has_terminal"] = one
            diagnostics["bit_adapter_terminal_source_code"] = reference.detach().new_tensor(
                float(TERMINAL_ALIASES.index(terminal_source) + 1)
            )

    path: Optional[torch.Tensor] = None
    if path_tensor is not None and path_source is not None:
        path = _validate_path(
            path_tensor,
            batch_size=batch_size,
            action_horizon=action_horizon,
            action_dim=action_dim,
            reference=reference,
            source=path_source,
            strict=strict,
        )
        if path is not None:
            diagnostics["bit_adapter_has_path"] = one
            diagnostics["bit_adapter_path_source_code"] = reference.detach().new_tensor(
                float(PATH_ALIASES.index(path_source) + 1)
            )

    return terminal, path, diagnostics
