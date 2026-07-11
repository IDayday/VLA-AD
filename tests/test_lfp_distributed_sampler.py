import torch

from navsim.agents.recogdrive.stage3_frontier_curriculum import DistributedFrontierSampler


def test_sampler_keeps_uniform_mixture_and_is_reproducible() -> None:
    size = 10_000
    rho = 0.2
    frontier = torch.zeros(size, dtype=torch.float64)
    frontier[0] = 1.0
    uniform = torch.full_like(frontier, 1.0 / size)
    mixed = (1.0 - rho) * frontier + rho * uniform
    sampler = DistributedFrontierSampler(
        size,
        mixed,
        seed=17,
        warmup_epochs=1,
        uniform_ratio=rho,
    )
    sampler.set_epoch(1)
    first = list(sampler)
    ratio = sampler.last_uniform_ratio_actual
    sampler.set_epoch(1)
    second = list(sampler)
    assert first == second
    assert 0.18 <= ratio <= 0.22
    assert torch.all(sampler.weights >= rho / size - 1e-15)


def test_two_ranks_receive_strided_global_draw_streams() -> None:
    size = 101
    weights = torch.full((size,), 1.0 / size)
    rank0 = DistributedFrontierSampler(size, weights, num_replicas=2, rank=0, seed=9, warmup_epochs=0)
    rank1 = DistributedFrontierSampler(size, weights, num_replicas=2, rank=1, seed=9, warmup_epochs=0)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    assert len(list(rank0)) == len(rank0) == 51
    assert len(list(rank1)) == len(rank1) == 51
