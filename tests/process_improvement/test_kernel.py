"""CPU mathematical adapter invariants; real-model parity is a separate gate."""
from types import SimpleNamespace
import pytest
import torch
from navsim.agents.recogdrive.process_improvement.contracts import TransitionStats
from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
from navsim.agents.recogdrive.process_improvement.improvement_target import proposal,soft_weights


class AnalyticHead:
    config=SimpleNamespace(sampling_method='ddim')
    ddim_steps=5
    ddim_t=torch.tensor([80,60,40,20,0])
    final_action_clip_value=.7
    def p_mean_variance(self,x,t,index,**kwargs):
        assert torch.equal(t,self.ddim_t[index])
        return x*.3+index[:,None,None],torch.zeros(len(x),1,1)-2,x*.3
    def denorm_odo(self,x):return x


def setup():
    head=AnalyticHead();adapter=NativeAdapter(head)
    context=dict(context_embeds=torch.zeros(1,2,3))
    return head,adapter,context


def test_all_suffix_cuts_and_raw_final():
    head,adapter,context=setup()
    noise=torch.randn(1,5,8,3,generator=torch.Generator().manual_seed(11))
    start=torch.zeros(1,8,3)
    trajectory,states=adapter.rollout_suffix(context,start,0,noise,True)
    assert states[:,-1].max()>head.final_action_clip_value
    assert trajectory.max()<=head.final_action_clip_value
    for cut in range(6):
        final,suffix=adapter.rollout_suffix(context,states[:,cut],cut,noise,True)
        assert torch.equal(final,trajectory)
        assert torch.equal(suffix,states[:,cut:])


def test_last_suffix_does_not_query_dit():
    head,adapter,context=setup()
    def forbidden(*args,**kwargs):raise AssertionError('Repeated a completed step')
    head.p_mean_variance=forbidden
    result=adapter.rollout_suffix(context,torch.ones(1,8,3),5,torch.zeros(1,5,8,3))
    assert torch.equal(result,torch.full((1,8,3),.7))


@pytest.mark.parametrize('index',[-1,6])
def test_invalid_suffix_index(index):
    _,adapter,context=setup()
    with pytest.raises(ValueError):adapter.rollout_suffix(context,torch.zeros(1,8,3),index,torch.zeros(1,5,8,3))


def test_suffix_absolute_index_noise_required():
    _,adapter,context=setup()
    with pytest.raises(ValueError):adapter.rollout_suffix(context,torch.zeros(1,8,3),3,torch.zeros(1,2,8,3))


@pytest.mark.parametrize('bad_index',[-1,5])
def test_native_scheduler_index_guard(bad_index):
    _,adapter,context=setup()
    with pytest.raises(ValueError):adapter.transition_stats(torch.zeros(1,8,3),bad_index,context)


def test_noise_is_clipped_once_and_saved_raw_not_mutated():
    _,adapter,_=setup()
    raw=torch.full((1,8,3),3.)
    stats=TransitionStats(torch.zeros(1,8,3),torch.full((1,1,1),.2),torch.zeros(1,8,3),0,80,'native_inference',1.,False)
    result=adapter.sample_next(stats,raw)
    assert torch.equal(raw,torch.full_like(raw,3.))
    assert torch.equal(result,torch.full_like(raw,.2))
    stats.std.zero_()
    assert torch.equal(adapter.sample_next(stats,raw),stats.mean)


def test_invalid_noise_fails_instead_of_zero_reward():
    _,adapter,_=setup()
    stats=TransitionStats(torch.zeros(1,8,3),torch.zeros(1,1,1),torch.zeros(1,8,3),0,80,'native_inference',1.,False)
    with pytest.raises(ValueError):adapter.sample_next(stats,torch.full((1,8,3),float('nan')))


@pytest.mark.parametrize('key',['future_trajectory','pdm_score','scene_id','teacher_answer','candidate_source'])
def test_label_input_boundary(key):
    _,adapter,_=setup()
    with pytest.raises(ValueError):adapter.prepare_conditioning(torch.zeros(1,2,3),{key:torch.zeros(1)})


def test_clipped_gaussian_atoms_and_second_moment():
    generator=torch.Generator().manual_seed(61)
    raw=torch.randn(200000,generator=generator)
    clipped=raw.clamp(-1,1)
    assert abs(float((clipped==1).float().mean())-.158655)<.004
    assert abs(float((clipped==-1).float().mean())-.158655)<.004
    assert abs(float(clipped.square().mean())-.51605855)<.005


def test_antithetic_rho_one_is_empirical_weighted_mean():
    mean=torch.zeros(8,3)
    candidates=torch.tensor([-1.,-.5,.5,1.])[:,None,None].expand(4,8,3)
    values=torch.tensor([.1,.2,.5,.9])
    result=proposal(mean,torch.ones(1,1),candidates,values,rho=1.,radius=100.,ess_fraction=.01)
    expected=(result.weights[:,None,None]*candidates).sum(0)
    torch.testing.assert_close(result.mean,expected)


def test_finite_state_exponential_tilt_objective():
    values=torch.tensor([0.,.25,.8,1.])
    temperature=.2
    optimum=soft_weights(values,temperature)
    def objective(weights):return (weights*values).sum()-temperature*(weights*(weights.log()+torch.log(torch.tensor(4.)))).sum()
    uniform=torch.ones(4)/4
    assert objective(optimum)>objective(uniform)
    generator=torch.Generator().manual_seed(99)
    for _ in range(30):
        weights=torch.rand(4,generator=generator);weights/=weights.sum()
        assert objective(optimum)>=objective(weights)-1e-6
