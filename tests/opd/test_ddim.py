import torch,pytest
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner as Planner


def test_T07_native_normalization():
    low=torch.tensor([-1.57,-19.68,-1.67]);high=torch.tensor([65.17,22.32,1.86]);value=torch.stack([low,high])[None].expand(4,-1,-1)
    normalized=Planner.norm_odo(None,value)
    assert torch.allclose(normalized[:,0],torch.full((4,3),-1.))
    assert torch.allclose(normalized[:,1],torch.ones(4,3),atol=2e-7)
    assert torch.allclose(Planner.denorm_odo(None,normalized),value,atol=1e-5)


def test_T28_zero_weight_is_native_call():
    from navsim.agents.recogdrive.opd.trainer import grpo_with_opd
    sentinel=object()
    class Native:
        def forward_grpo(self,*args,**kwargs):return sentinel
    result,aux=grpo_with_opd(Native(),None,None,None,None,weight=0,anchor_weight=0)
    assert result is sentinel and aux is None
