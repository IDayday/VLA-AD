"""Reward estimates from complete reference-policy suffixes, without value nets."""
import numpy as np
import torch


@torch.no_grad()
def evaluate_candidates(adapter, context, candidates, index, key, replicates, namespace,
                        score, scene_token, metric_cache, bank=0):
    if candidates.ndim != 3 or candidates.shape[1:] != (8, 3) or replicates < 1:
        raise ValueError('Expected M complete trajectory states and positive L')
    count = len(candidates)
    # Layout is [candidate, suffix replicate, absolute step, horizon, coordinate].
    noises = torch.cat([key.plan(namespace, bank*10000+l, candidates.device, candidates.dtype).steps
                        for l in range(replicates)])
    raw = noises.unsqueeze(0).expand(count, -1, -1, -1, -1).reshape(count*replicates, 5, 8, 3)
    starts = candidates[:, None].expand(-1, replicates, -1, -1).reshape(-1, 8, 3)
    trajectories = adapter.rollout_suffix(context, starts, index+1, raw)
    values = score.score(scene_token, metric_cache, trajectories.cpu().numpy())
    return values.reshape(count, replicates, 7), trajectories.reshape(count, replicates, 8, 3)


@torch.no_grad()
def evaluate_kernels(adapter, context, stats, means, key, replicates, namespace,
                     score, scene_token, metric_cache, bank=0):
    # Every kernel uses the same current-step raw noise and reference suffix noise.
    plans = [key.plan(namespace, bank*10000+l, stats.mean.device, stats.mean.dtype) for l in range(replicates)]
    current = torch.cat([p.initial for p in plans])
    suffix = torch.cat([p.steps for p in plans])
    starts = torch.cat([adapter.sample_next(stats, current, mean=mean) for mean in means])
    trajectories = adapter.rollout_suffix(context, starts, stats.index+1, suffix.repeat(len(means), 1, 1, 1))
    values = score.score(scene_token, metric_cache, trajectories.cpu().numpy())
    return values.reshape(len(means), replicates, 7), trajectories.reshape(len(means), replicates, 8, 3)
