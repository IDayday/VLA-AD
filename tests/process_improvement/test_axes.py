"""Analytic layout tests: candidate/replicate/execution-index axes and CRN."""
from types import SimpleNamespace
import numpy as np
import torch
from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
from navsim.agents.recogdrive.process_improvement.candidate_value import evaluate_candidates,evaluate_kernels


class AnalyticSuffix:
    def sample_next(self,stats,noise,mean=None):
        return (stats.mean if mean is None else mean)+stats.std*noise.clamp(-1,1)
    def rollout_suffix(self,context,start,next_index,noises):
        return start+noises[:,next_index:].sum(1)


class SumScore:
    def score(self,token,cache,trajectories):
        assert token=='scene'
        return trajectories.sum((1,2))[:,None]+np.arange(7)[None,:]


def test_candidate_major_suffix_replicate_axes_and_crn():
    key=RNGKey('test',42,0,0,'scene',0,2)
    candidates=torch.stack([torch.full((8,3),float(i)) for i in range(3)])
    values,trajectories=evaluate_candidates(AnalyticSuffix(),{},candidates,2,key,4,'fit_suffix',SumScore(),'scene',None)
    assert values.shape==(3,4,7) and trajectories.shape==(3,4,8,3)
    for candidate in range(3):
        for replica in range(4):
            noise=key.plan('fit_suffix',replica).steps
            expected=candidates[candidate]+noise[0,3:].sum(0)
            assert torch.equal(trajectories[candidate,replica],expected)
            np.testing.assert_allclose(values[candidate,replica],float(expected.sum())+np.arange(7),atol=1e-5)


def test_guard_mean_replica_axes_and_current_step_pairing():
    key=RNGKey('test',42,0,0,'scene',0,1)
    stats=SimpleNamespace(mean=torch.zeros(1,8,3),std=torch.full((1,1,1),.2),index=1)
    means=[torch.full((1,8,3),float(j)) for j in range(3)]
    values,trajectories=evaluate_kernels(AnalyticSuffix(),{},stats,means,key,3,'target_guard',SumScore(),'scene',None)
    for mean in range(3):
        for replica in range(3):
            plan=key.plan('target_guard',replica)
            expected=means[mean][0]+stats.std[0]*plan.initial[0].clamp(-1,1)+plan.steps[0,2:].sum(0)
            assert torch.equal(trajectories[mean,replica],expected)
    torch.testing.assert_close(trajectories[1]-trajectories[0],torch.ones_like(trajectories[0]))
