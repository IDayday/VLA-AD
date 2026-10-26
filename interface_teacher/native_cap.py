"""Verbatim native FP32 application and post-rounding RMS cap. Source SHA in provenance."""
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
