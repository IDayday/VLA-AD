from __future__ import annotations

from typing import Iterable

import torch


def stable_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Computes an L2 norm without overflowing on large finite float32 gradients."""
    grads = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    device = grads[0].device
    max_abs = torch.stack([grad.abs().max() for grad in grads]).max()
    finite_scale = torch.isfinite(max_abs) & (max_abs > 0.0)
    scale = torch.where(finite_scale, max_abs, max_abs.new_ones(()))
    scaled_sum_sq = torch.zeros((), device=device, dtype=torch.float32)
    for grad in grads:
        scaled_sum_sq = scaled_sum_sq + (grad / scale).square().sum()
    finite_norm = scale * scaled_sum_sq.sqrt()
    return torch.where(torch.isfinite(max_abs), finite_norm, max_abs)
