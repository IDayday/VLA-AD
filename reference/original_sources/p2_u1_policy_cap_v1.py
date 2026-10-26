"""Apply an ordinary shared-policy shift, with a cap rather than dose forcing.

Unlike teacher-known G1 matching, this never enlarges a request. Exact zero and
uncapped requests retain their direct FP32 Base+shift arithmetic.
"""
import sys
sys.dont_write_bytecode=True


def apply(base,image,text,requested,caps):
    import torch
    assert base.dtype==requested.dtype==torch.float32
    assert base.ndim==3 and base.shape[0]==1 and base.shape[2]==384
    assert requested.ndim==3 and requested.shape[1:]==(2,384)
    assert image.shape==text.shape==base.shape[:2]
    assert image.dtype==text.dtype==torch.bool and not bool((image&text).any())
    assert caps.shape==(len(requested),) and caps.dtype==torch.float64
    assert bool(torch.isfinite(caps).all()) and bool((caps>=0).all())
    assert bool(torch.isfinite(requested).all())
    ni=int(image.sum());nt=int(text.sum());assert ni and nt
    counts=torch.tensor([ni,nt],device=base.device,dtype=torch.float64)
    nominal=(requested.double().square().mean(-1)*counts[None]).sum(-1).div(ni+nt).sqrt()
    scale=torch.minimum(torch.ones_like(caps),caps/nominal.clamp_min(1e-300)).float()
    scale=torch.where((caps==0)|(nominal==0),torch.zeros_like(scale),scale)
    tolerance=torch.maximum(torch.full_like(caps,1e-7),caps*1e-4)
    valid=(image|text)[0];corrections=0
    while True:
        shifts=requested*scale[:,None,None]
        delta=image[0][None,:,None]*shifts[:,0,None]+text[0][None,:,None]*shifts[:,1,None]
        interfaces=base+delta
        actual=(interfaces[:,valid].double()-base[:,valid].double()).square().mean((1,2)).sqrt()
        over=actual>caps+tolerance
        if not bool(over.any()):break
        assert corrections<8 and bool((actual[over]>0).all())
        factor=torch.where(over,(caps/actual.clamp_min(1e-300)).clamp_max(1),torch.ones_like(caps))
        scale=(scale.double()*factor).float();corrections+=1
    assert bool(torch.isfinite(interfaces).all()) and bool((scale>=0).all()) and bool((scale<=1).all())
    if bool((~valid).any()):
        assert torch.equal(interfaces[:,~valid],base[:,~valid].expand(len(requested),-1,-1))
    no_op=(interfaces==base).flatten(1).all(1)
    return dict(interfaces=interfaces,actual_role_shifts=shifts,scales=scale,
        actual_RMS=actual,cap_RMS=caps,nominal_RMS=nominal,no_op=no_op,
        tolerance=tolerance,cap_corrections=corrections,padding_exact=True)


def selftest():
    import torch
    gen=torch.Generator(device='cpu').manual_seed(2026102490)
    base=torch.randn((1,19,384),generator=gen)
    image=torch.zeros((1,19),dtype=torch.bool);image[:,:11]=True
    text=torch.zeros_like(image);text[:,11:16]=True
    request=torch.randn((6,2,384),generator=gen)
    request[0].zero_();request[1]*=.001;request[2]*=1e-12
    caps=torch.tensor([1.,1.,1.,0.,.05,5.],dtype=torch.float64)
    rng=torch.get_rng_state().clone();r=apply(base,image,text,request,caps)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(r['interfaces'][0:1],base) and torch.equal(r['interfaces'][3:4],base)
    assert bool((r['actual_RMS']<=caps+r['tolerance']).all())
    assert r['scales'][1]==1 and r['scales'][5]==1
    expected=base+image[0][None,:,None]*request[1:2,0,None]+text[0][None,:,None]*request[1:2,1,None]
    assert torch.equal(r['interfaces'][1:2],expected)
    assert r['actual_RMS'][1]<.01 and r['actual_RMS'][4]>.049
    return dict(status='PASS_SHARED_POLICY_CAP_MATH',no_dose_amplification=True,
        uncapped_direct_add_exact=True,zero_exact=True,padding_exact=True,RNG_unchanged=True,
        actual_accelerator_not_yet_validated=True)


if __name__=='__main__':
    import json
    print(json.dumps(selftest()),flush=True)
