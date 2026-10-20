"""Analytic/toy correctness only: these are not NAVSIM research results."""
import pytest
import torch
from navsim.agents.recogdrive.process_improvement.improvement_target import proposal, soft_weights, effective_sample_size
from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey, candidate_noise


def bank():
    mean = torch.ones(8, 3)
    offsets = torch.tensor([-3., -1., 1., 3.])[:, None, None].expand(4, 8, 3)
    return mean, torch.ones(1, 1), mean+offsets


@pytest.mark.parametrize('values', [torch.zeros(4), torch.ones(4), torch.full((4,), .7)])
def test_flat_exact_reference(values):
    mean, std, candidates = bank()
    target = proposal(mean, std, candidates, values)
    assert torch.equal(target.mean, mean)
    assert target.reason == 'FLAT_VALUE'


def test_rho_zero_and_zero_variance():
    mean, std, candidates = bank()
    values = torch.tensor([0., 1., 1., 0.])
    assert torch.equal(proposal(mean, std, candidates, values, rho=0).mean, mean)
    assert torch.equal(proposal(mean, std*0, candidates, values).mean, mean)
    assert torch.equal(proposal(mean, std, candidates[:1], values[:1]).mean, mean)
    assert candidate_noise(RNGKey('test',42,0,0,'x',0,0),1).shape == (1,8,3)


def test_ess_increases_and_radius_and_permutation():
    mean, std, candidates = bank()
    values = torch.tensor([0., .2, .7, 1.])
    ess = [float(effective_sample_size(soft_weights(values, temperature))) for temperature in [.02, .05, .1, 1.]]
    assert ess == sorted(ess)
    target = proposal(mean, std, candidates, values)
    assert target.ess >= 2.-1e-6 and target.radius_after <= .25+1e-6
    assert target.temperature >= .05
    perm = torch.tensor([2,0,3,1])
    other = proposal(mean, std, candidates[perm], values[perm])
    torch.testing.assert_close(target.mean, other.mean)
    shifted = proposal(mean, std, candidates, values+10)
    torch.testing.assert_close(target.mean, shifted.mean, atol=2e-6, rtol=1e-5)


def test_antithetic_center_and_stateless_isolation():
    key = RNGKey('test',42,1,0,'scene',7,2)
    global_before = torch.get_rng_state()
    noise = candidate_noise(key, 8)
    assert torch.equal(noise[:4], -noise[4:])
    assert torch.equal(noise.clamp(-1,1).sum(0), torch.zeros(8,3))
    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(noise, candidate_noise(key, 8))
    assert len({key.seed(ns) for ns in ['rollin','candidate','fit_suffix','target_guard','diagnostic_test']}) == 5
    assert not torch.equal(key.plan('fit_suffix').steps, key.plan('target_guard').steps)


def test_quadratic_reward_moves_towards_optimum():
    mean = torch.zeros(8,3)
    candidates = torch.tensor([-1., -.5, .5, 1.])[:,None,None].expand(4,8,3)
    reward = -(candidates[:,0,0]-.8).square()
    p = proposal(mean, torch.ones(1,1), candidates, reward)
    assert 0 < float(p.mean.mean()) <= .25+1e-6
    assert -float((p.mean-.8).square().mean()) > -.8**2


def test_multimodal_selection_does_not_imply_projection_gain():
    mean, std, candidates = bank()
    reward = torch.tensor([1.,0.,1.,0.])
    p = proposal(mean, std, candidates, reward, temperature=.2, rho=1., radius=100., ess_fraction=.01)
    weighted_gain = float((p.weights*reward).sum())
    # The two rewarded modes are -2 and +2. A projected mean near 0 loses both.
    assert weighted_gain > .99
    assert abs(float(p.mean.mean())) < .02
    # Compare the entire *same offset kernel*, not reward at its mean alone.
    offsets=candidates[:,0,0]-mean[0,0]
    projected_support=p.mean[0,0]+offsets
    projected_reward=float((((projected_support-2).abs()<.1) | ((projected_support+2).abs()<.1)).float().mean())
    assert projected_reward == 0 and reward.mean() == .5


def test_random_fit_winner_has_no_independent_gain():
    gen = torch.Generator().manual_seed(102)
    fit = torch.rand(20000,8,generator=gen)
    independent = torch.rand(20000,8,generator=gen)
    selected = independent.gather(1,fit.argmax(1)[:,None]).mean()
    assert abs(float(selected)-.5) < .01
    assert fit.max(1).values.mean() > .85


@pytest.mark.parametrize('count', [0,3])
def test_invalid_antithetic_bank_rejected(count):
    with pytest.raises(ValueError):
        candidate_noise(RNGKey('test',42,0,0,'x',0,0),count)
