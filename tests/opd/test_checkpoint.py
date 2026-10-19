import pytest,torch
from torch import nn
from types import SimpleNamespace
from navsim.agents.recogdrive.opd.checkpoint_io import restore,read_state

def model():
    m=nn.Sequential(nn.Linear(3,2));m.config=SimpleNamespace(use_action_aware_aux=False);return m

@pytest.mark.parametrize('suffix',['.ckpt','.pt','.bin','.safetensors'])
def test_T01_T02_strict_formats(tmp_path,suffix):
    original=model();m=model();p=tmp_path/('weights'+suffix)
    state={'agent.action_head.'+k:v for k,v in original.state_dict().items()}
    if suffix=='.safetensors':
        from safetensors.torch import save_file
        save_file(state,str(p))
    else:torch.save({'state_dict':dict(state,**{'agent.action_head.old_policy.0.weight':torch.zeros(2,3)})},p)
    report=restore(m,p);assert not report['missing'] and all(torch.equal(a,b) for a,b in zip(original.parameters(),m.parameters()))

def test_missing_and_wrong_shapes_rejected(tmp_path):
    p=tmp_path/'broken.pt';torch.save({'state_dict':{'agent.action_head.0.weight':torch.zeros(1,3)}},p)
    with pytest.raises(ValueError,match='Strict checkpoint failure'):restore(model(),p)


def test_missing_auxiliary_is_explicit_and_only_allowed_when_disabled(tmp_path):
    m=model();core={k:v.clone() for k,v in m.state_dict().items()};m.action_aware_aux_head=nn.Linear(3,3)
    p=tmp_path/'core_only.pt';torch.save(core,p)
    report=restore(m,p)
    assert not report['missing'] and set(report['missing_all'])=={'action_aware_aux_head.weight','action_aware_aux_head.bias'}
    assert report['missing_noncore']==report['missing_all']
    m.config.use_action_aware_aux=True
    with pytest.raises(ValueError,match='Strict checkpoint failure'):restore(m,p)
