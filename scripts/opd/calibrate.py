"""Fit-only gradient-scale calibration before hyperparameter comparisons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.opd.common import *


def main():
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.trainer import objective
    cfg=config();configure();s=build_head(ROOT/'runs/tiny_E3_42/latest.pt');t=build_head(cfg['teacher_init'],frozen=True);a=build_head(cfg['anchor_init'],frozen=True)
    lengths=read(ROOT/'manifests/token_lengths.json');groups={}
    for r in rows('train'):groups.setdefault(lengths[r['token']],[]).append(r)
    selected=sorted(max(groups.values(),key=len),key=lambda r:digest([r['token'],'gradient_scale_calibration']))[:16]
    params=[p for p in s.parameters() if p.requires_grad];records=[]
    for offset in (0,8):
        batch=selected[offset:offset+8];pairs=[conditioning(observation(r)) for r in batch]
        tokens=torch.cat([p[0] for p in pairs]);inputs=BatchFeature({k:torch.cat([p[1][k] for p in pairs]) for k in pairs[0][1]})
        seeds(8891+offset)
        out=objective(s,t,a,tokens,inputs,experiment='E4-C',target=torch.stack([gt(r) for r in batch]).cuda(),rollout_seed=4421+offset)
        grads={k:torch.autograd.grad(out[k],params,retain_graph=True,allow_unused=True) for k in ('opd','anchor','il')}
        norms={k:sum(float(g.square().sum()) for g in v if g is not None)**.5 for k,v in grads.items()}
        cosine={k:sum(float((x*y).sum()) for x,y in zip(grads['opd'],grads[k]) if x is not None and y is not None)/max(norms['opd']*norms[k],1e-30) for k in ('anchor','il')}
        records.append(dict(tokens=[r['token'] for r in batch],gradient_norms=norms,gradient_cosine_with_opd=cosine,
            losses={k:float(out[k].detach()) for k in ('opd','anchor','il')}))
    med={k:float(np.median([r['gradient_norms'][k] for r in records])) for k in ('opd','anchor','il')}
    weights=dict(anchor=float(np.clip(.1*med['opd']/max(med['anchor'],1e-30),.001,1.)),
                 il=float(np.clip(.05*med['opd']/max(med['il'],1e-30),.000001,.05)))
    dest=ROOT/'calibration/gradient_scales.json'
    if dest.exists():raise ValueError('Calibration is immutable')
    write(dest,dict(status='LOCKED',records=records,weights=weights,owner=identity(),
        rule='Use E3 tiny qualification checkpoint on fixed16 D_fit scenes. Set auxiliary median gradient to0.1(anchor)/0.05(GT) of OPD, clip anchor[.001,1],GT[1e-6,.05]. No cal/dev/test outcome selection.',
        no_initialization_transfer=True))
    print(weights)

if __name__=='__main__':main()
