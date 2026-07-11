import torch

import navsim.agents.recogdrive.stage3_lfp_grpo as lfp


def test_single_rank_global_moments() -> None:
    values = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    mask = torch.tensor([[True, True], [False, True]])
    moments = lfp.distributed_masked_moments(values, mask, std_floor=0.05)
    expected = torch.tensor([1.0, 2.0, 20.0])
    torch.testing.assert_close(moments.mean, expected.mean())
    torch.testing.assert_close(moments.std, expected.std(unbiased=False))
    assert moments.count.item() == 3.0


def test_simulated_all_reduce_matches_concatenated_values(monkeypatch) -> None:
    local = torch.tensor([[1.0, 3.0]])
    remote = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float64)

    monkeypatch.setattr(lfp.dist, "is_available", lambda: True)
    monkeypatch.setattr(lfp.dist, "is_initialized", lambda: True)

    def fake_all_reduce(stats, op=None):
        stats += torch.tensor(
            [remote.numel(), remote.sum(), remote.square().sum()],
            dtype=stats.dtype,
            device=stats.device,
        )

    monkeypatch.setattr(lfp.dist, "all_reduce", fake_all_reduce)
    moments = lfp.distributed_masked_moments(local, torch.ones_like(local, dtype=torch.bool), 0.05)
    full = torch.cat((local.flatten().double(), remote))
    torch.testing.assert_close(moments.mean.double(), full.mean())
    torch.testing.assert_close(moments.std.double(), full.std(unbiased=False))


def test_global_std_is_not_scene_local_std() -> None:
    values = torch.tensor([[0.0, 2.0], [0.0, 20.0]])
    moments = lfp.distributed_masked_moments(values, torch.ones_like(values, dtype=torch.bool), 0.05)
    scene_std_mean = values.std(dim=1, unbiased=False).mean()
    assert not torch.isclose(moments.std, scene_std_mean)
