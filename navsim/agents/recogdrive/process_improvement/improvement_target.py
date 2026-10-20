"""Finite-bank, mean-centered heuristic projection (not an exact Gaussian KL)."""
from dataclasses import dataclass
import torch


@dataclass
class Proposal:
    mean: torch.Tensor
    weights: torch.Tensor
    temperature: float
    ess: float
    radius_before: float
    radius_after: float
    flat: bool
    reason: str


def soft_weights(values, temperature):
    if temperature <= 0 or not torch.isfinite(values).all():
        raise ValueError('Finite values and positive temperature required')
    return torch.softmax((values.float()-values.float().max())/temperature, dim=0)


def effective_sample_size(weights):
    return 1. / weights.square().sum()


def proposal(mean, std, candidates, values, *, temperature=.05, rho=.5, radius=.25, ess_fraction=.5, flat_tolerance=1e-6):
    if candidates.ndim != mean.ndim+1 or candidates.shape[1:] != mean.shape or values.shape != (len(candidates),):
        raise ValueError('Candidate/value shapes do not match a single transition state')
    if not all(torch.isfinite(v).all() for v in (mean, std, candidates, values)) or (std < 0).any():
        raise ValueError('Nonfinite/negative transition data')
    if not 0 < ess_fraction <= 1 or rho < 0 or radius < 0:
        raise ValueError('Invalid projection constraints')
    count = len(candidates)
    mean32, std32 = mean.float(), std.float().expand_as(mean)
    uniform = torch.full_like(values.float(), 1./count)
    flat = count == 1 or float(values.max()-values.min()) <= flat_tolerance
    if flat or rho == 0:
        return Proposal(mean32.clone(), uniform, temperature, float(count), 0., 0., flat, 'FLAT_VALUE' if flat else 'RHO_ZERO')
    low, high = temperature, temperature
    weights = soft_weights(values, high)
    target_ess = ess_fraction*count
    while float(effective_sample_size(weights)) < target_ess:
        high *= 2
        weights = soft_weights(values, high)
    if high > low:
        for _ in range(40):
            middle = (low+high)/2
            if float(effective_sample_size(soft_weights(values, middle))) < target_ess:
                low = middle
            else:
                high = middle
        weights = soft_weights(values, high)
    delta = rho*((weights-uniform).reshape(count, *([1]*mean.ndim))*(candidates.float()-mean32)).sum(0)
    delta = torch.where(std32 > 0, delta, torch.zeros_like(delta))
    scaled = torch.where(std32 > 0, delta/torch.where(std32 > 0, std32, torch.ones_like(std32)), torch.zeros_like(delta))
    before = float(scaled.square().mean().sqrt())
    factor = min(1., radius/before) if before > 0 else 1.
    delta = delta*factor
    return Proposal(mean32+delta, weights, high, float(effective_sample_size(weights)), before, before*factor, False, 'PROPOSED')
