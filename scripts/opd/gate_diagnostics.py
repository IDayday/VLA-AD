"""Fixed stratified training diagnostic, separate from held-out evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter,defaultdict
from scripts.opd.common import *
from scripts.opd.analyze import bank


def register():
    g=read(ROOT/'gates/static.json')['scenes'];data=rows('train');selected=[]
    for cat in 'ABCD':selected.extend(sorted([r for r in data if g[r['token']]['category']==cat],key=lambda r:digest([r['token'],'gate_diagnostic_v1']))[:64])
    p=ROOT/'manifests/train_gate_diagnostic.json'
    if p.exists():assert read(p)==selected
    else:write(p,selected)
    write(ROOT/'diagnostics/gate_registration.json',dict(purpose='Training-distribution descriptive diagnostic; not generalization or selection',max_per_class=64,source='Static train-only IL/RL gate',
        scenes=len(selected),counts=dict(Counter(g[r['token']]['category'] for r in selected)),manifest_sha256=sha(p),owner=identity()))


def analyze():
    import torch
    gate=read(ROOT/'gates/static.json')['scenes'];data=rows('train_gate_diagnostic');lookup={r['token']:i for i,r in enumerate(rows('train'))}
    il=bank('train','E0','IL')[[lookup[r['token']] for r in data]];rl=bank('train','E0','RL')[[lookup[r['token']] for r in data]]
    summary={}
    for e in ['E3','E5']:
        y=np.stack([bank('train_gate_diagnostic',f'train_{e}_{s}') for s in config()['train_seeds']]);means=y[...,-1].mean(-1);base=il[...,-1].mean(-1)
        summary[e]={}
        for cat in 'ABCD':
            mask=np.array([gate[r['token']]['category']==cat for r in data]);d=means[:,mask]-base[None,mask]
            summary[e][cat]=dict(scenes=int(mask.sum()),PDMS=float(means[:,mask].mean()),IL_PDMS=float(base[mask].mean()),RL_PDMS=float(rl[mask,...,-1].mean()),G=float(d.mean()),M_minus=float(np.maximum(-d,0).mean()),positive_to_zero=float(((base[None,mask]>1e-12)&(means[:,mask]<=1e-12)).mean()))
    exposures={}
    for seed in config()['train_seeds']:
        dest=ROOT/'runs'/f'train_E5_{seed}';counts=Counter();active=Counter();losses=defaultdict(list)
        for p in sorted((dest/'rollouts').glob('*_0.pt')):
            r=torch.load(p,map_location='cpu',weights_only=False)
            per_scene=(r['student_means']-r['teacher_means']).square().mean((1,2,3)) if r['teacher_means'] is not None else torch.zeros(len(r['scene_tokens']))
            for j,token in enumerate(r['scene_tokens']):
                cat=gate[token]['category'];m=float(r['effective_gate'][j]);counts[cat]+=1;active[cat]+=int(m>0);losses[cat].append(float(per_scene[j])*m)
        assert sum(counts.values())==8000
        exposures[str(seed)]=dict(examples=dict(counts),active_OPD_examples=dict(active),mean_masked_OPD={c:float(np.mean(v)) for c,v in losses.items()},activation_fraction=sum(active.values())/sum(counts.values()))
    write(ROOT/'diagnostics/gate_results.json',dict(scope=f'Fixed{len(data)} training scenes; max64 per original A/B/C/D class; descriptive, not held-out evidence',categories=summary,actual_training_exposure=exposures))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','analyze']);a=p.parse_args();globals()[a.action]()
