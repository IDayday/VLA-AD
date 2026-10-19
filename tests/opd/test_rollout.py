"""Weight-free contract tests; native-checkpoint parity is a separate real gate."""
import copy
from types import SimpleNamespace
import pytest,torch
from torch import nn
from navsim.agents.recogdrive.opd.ddim_adapter import DDIMAdapter
from navsim.agents.recogdrive.opd.losses import transition_loss
from navsim.agents.recogdrive.opd.checkpoint_io import restore


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Linear(3,3);self.his_traj_encoder=nn.Linear(12,3);self.ego_status_encoder=nn.Linear(8,3)
        self.config=SimpleNamespace(sampling_method='ddim',action_horizon=8,use_expert_features=False,use_last_rd=False,use_bit_drive=False,use_risk_vla=False,use_action_aware_aux=False)
        self.register_buffer('ddim_t',torch.tensor([80,60,40,20,0]));self.ddim_steps=5;self.calls=[]
    def _prepare_dit_context(self,tokens,inputs,training):return dict(context_tokens=tokens,context_mean=tokens.mean(1))
    def p_mean_variance(self,x,t,index,deterministic,**kwargs):
        self.calls.append((t.clone(),index.clone()))
        mean=.5*x+.1*self.model(x);return mean,torch.full_like(x[:,:1,:1],-46. if deterministic else -4.),mean
    def denorm_odo(self,x):return x


def observation(batch=1):
    return torch.zeros(batch,3,3),SimpleInputs(his_traj=torch.zeros(batch,12),status_feature=torch.zeros(batch,8))
class SimpleInputs(dict):
    def __getattr__(self,key):return self[key]


def test_T06_T13_T14_T15_current_chain_and_index():
    torch.manual_seed(42);s=ToyHead();t=copy.deepcopy(s);t.requires_grad_(False)
    with torch.no_grad():t.model.bias.add_(1)
    tokens,inputs=observation();a=DDIMAdapter(s);chain=a.sample(tokens,inputs);other=DDIMAdapter(t).sample(tokens,inputs)
    assert chain.source==id(s) and other.source==id(t) and not chain.states.requires_grad
    assert [int(time.item()) for time,_ in s.calls]==[80,60,40,20,0]
    assert [int(index.item()) for _,index in s.calls]==[0,1,2,3,4]
    pred=a.evaluate_chain(chain,tokens,inputs,require_current=True)
    with torch.no_grad():target=DDIMAdapter(t).evaluate_chain(chain,tokens,inputs)
    loss=transition_loss(pred,target)[0];loss.backward();torch.optim.SGD(s.parameters(),lr=.1).step()
    with pytest.raises(ValueError,match='Stale'):a.evaluate_chain(chain,tokens,inputs,require_current=True)
    assert a.sample(tokens,inputs).versions!=chain.versions
    with pytest.raises(ValueError):a.transition(chain.states[:,0],6,a.prepare_conditioning(tokens,inputs))


@pytest.mark.parametrize('batch',[1,4])
def test_T16_T17_T23_T26_sampling_reproducibility(batch):
    s=ToyHead();tokens,inputs=observation(batch);a=DDIMAdapter(s)
    torch.manual_seed(19);first=a.sample(tokens,inputs)
    torch.manual_seed(19);second=a.sample(tokens,inputs)
    assert torch.equal(first.states,second.states) and first.states.shape==(batch,6,8,3)
    torch.manual_seed(19);det=a.sample(tokens,inputs,deterministic=True)
    assert det.sigmas.count_nonzero()==0 and first.sigmas.min()>0
    assert not hasattr(s,'teacher') and torch.isfinite(first.trajectory).all()


def test_T24_input_boundary():
    s=ToyHead();tokens,inputs=observation();inputs['future_occupancy']=torch.full((1,),float('nan'))
    with pytest.raises(ValueError,match='Labels or unaudited'):DDIMAdapter(s).prepare_conditioning(tokens,inputs)


def test_T25_cpu_resume(tmp_path):
    torch.manual_seed(72);s=ToyHead();teacher=copy.deepcopy(s);teacher.requires_grad_(False)
    with torch.no_grad():teacher.model.bias.add_(1)
    opt=torch.optim.AdamW(s.parameters(),lr=.01);tokens,inputs=observation()
    def step(model,optimizer):
        optimizer.zero_grad();chain=DDIMAdapter(model).sample(tokens,inputs)
        with torch.no_grad():target=DDIMAdapter(teacher).evaluate_chain(chain,tokens,inputs)
        loss=transition_loss(DDIMAdapter(model).evaluate_chain(chain,tokens,inputs,require_current=True),target)[0]
        loss.backward();optimizer.step();return loss
    step(s,opt);path=tmp_path/'resume.pt';torch.save(dict(student=s.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),path)
    expected=step(s,opt).detach();resumed=ToyHead();restore(resumed,path);other=torch.optim.AdamW(resumed.parameters(),lr=.01)
    saved=torch.load(path,weights_only=False);other.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    actual=step(resumed,other).detach();assert torch.equal(expected,actual)
    assert all(torch.equal(x,y) for x,y in zip(s.parameters(),resumed.parameters()))
