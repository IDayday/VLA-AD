import pytest,torch
from navsim.agents.recogdrive.opd.losses import transition_loss
from navsim.agents.recogdrive.opd.gating import globally_normalized_local_sum

@pytest.mark.parametrize('batch',[1,4])
def test_T10_T11_T12_T16_gradients_and_identity(batch):
    student=torch.randn(batch,5,8,3,requires_grad=True);teacher=student.detach().clone()
    assert transition_loss(student,teacher)[0].item()==0
    teacher=teacher+1;loss,steps=transition_loss(student,teacher);assert loss.item()==pytest.approx(1)
    loss.backward();assert student.grad.norm()>0 and torch.isfinite(student.grad).all()
    assert teacher.grad is None and steps.shape==(batch,5)

def test_T18_native_kernel_and_degenerate_variance():
    x=torch.zeros(1,5,8,3);y=x+1;s=torch.ones(1,5,1,1)*.2
    with pytest.raises(ValueError,match='not a Gaussian'):transition_loss(x,y,objective='shared_variance_kl',sigma=s,teacher_sigma=s)
    assert transition_loss(x,y,objective='shared_variance_kl',sigma=s,teacher_sigma=s,kernel='gaussian')[0].item()==pytest.approx(300)
    with pytest.raises(ValueError,match='nondegenerate'):transition_loss(x,y,objective='shared_variance_kl',sigma=s*0,teacher_sigma=s*0,kernel='gaussian')
    with pytest.raises(ValueError,match='Shared actual'):transition_loss(x,y,objective='shared_variance_kl',sigma=s,teacher_sigma=s*2,kernel='gaussian')

def test_T20_T21_T27_mask_and_global_normalization():
    x=torch.zeros(4,5,8,3,requires_grad=True);target=x.detach()+torch.arange(4)[:,None,None,None]
    mask=torch.tensor([0.,1.,0.,1.]);loss,_=transition_loss(x,target,mask=mask);assert loss.item()==5
    z,_=transition_loss(x,target,mask=mask*0);z.backward();assert z.item()==0 and x.grad.abs().max()==0
    # Two ranks, uneven active counts: mean of rank gradients equals global loss gradient.
    p=torch.tensor(2.,requires_grad=True);values=(p-torch.arange(4).float()).square()
    full=(values*mask).sum()/mask.sum();g=torch.autograd.grad(full,p,retain_graph=True)[0]
    local=[globally_normalized_local_sum(values[idx],mask[idx],mask.sum(),2) for idx in [slice(0,1),slice(1,4)]]
    rank_mean=sum(torch.autograd.grad(v,p,retain_graph=True)[0] for v in local)/2
    assert torch.equal(g,rank_mean)

def test_T08_T09_T19_targets_detached():
    x=torch.randn(2,5,8,3,requires_grad=True);teacher=torch.randn_like(x,requires_grad=True)
    anchor=torch.randn_like(x,requires_grad=True)
    loss=transition_loss(x,teacher)[0]+transition_loss(x,anchor)[0];loss.backward()
    assert x.grad.norm()>0 and teacher.grad is None and anchor.grad is None
