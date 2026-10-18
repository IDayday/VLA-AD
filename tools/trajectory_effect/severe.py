"""One preregistered25-seed confirmation of severe drops, selected before rechecks."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import load_observation,baseline_score_path,score_record
from .updates import family,targets,directory,score_path,wait_file,policy_state,restore


def select(name):
    cfg,identity=protocol();f=family(name);dest=ROOT/'confirmation/severe'/name/'selection.json'
    if dest.exists():return read(dest)
    assert (ROOT/'confirmation/activation.json').exists()
    wait_file(directory(name)/'score_complete.json');strata={}
    for row in targets(f):
        token=row['token'];old=np.load(baseline_score_path(token))['scores'][:5,-1]
        delta=[float((np.load(score_path(name,s,token))['scores'][:,-1]-old).mean()) for s in (1,4)]
        if min(delta)<-.1:strata.setdefault(row['log_name'],[]).append(dict(token=token,log=row['log_name'],initial_delta_J=delta))
    for values in strata.values():values.sort(key=lambda v:digest([v['token'],name,'severe']))
    keys=sorted(strata,key=lambda log:digest([log,name,'severe_log']));chosen=[]
    while len(chosen)<64 and any(strata.values()):
        for log in keys:
            if strata[log]:chosen.append(strata[log].pop(0))
            if len(chosen)==64:break
    value=dict(protocol_sha256=identity,family=name,scenes=chosen,seeds=cfg['seeds']['severe_confirmation'],
        selection='Initial5-seed delta_J< -0.10 at step1 or4; log-stratified hash round robin, at most64',
        confirmatory_outcomes_seen=False)
    atomic_json(dest,value);return value


def evaluate(args):
    import torch
    from .runtime import load_agent,action_input,predict
    selection=select(args.family);cfg,identity=protocol();dest=ROOT/'confirmation/severe'/args.family
    lookup={r['token']:r for r in targets(family(args.family))};agent=load_agent(online=False);base=policy_state(agent.action_head);start=time.perf_counter()
    for step in (0,1,4):
        state=base if step==0 else torch.load(directory(args.family)/f'checkpoint_{step}.pt',map_location='cpu',weights_only=False)['policy']
        restore(agent.action_head,state)
        for item in selection['scenes'][args.rank::args.shards]:
            token=item['token'];row=lookup[token];data=load_observation(token);inputs=action_input(data['features'])
            source=dest/f'step_{step}/trajectories/{token}.npz';out=dest/f'step_{step}/scores/{token}.npz'
            if not source.exists():
                ts=np.stack([predict(agent.action_head,data['tokens'].cuda(),inputs,token,s) for s in selection['seeds']])
                atomic_npz(source,dict(token=token,family=args.family,step=step,protocol_sha256=identity,seeds=selection['seeds']),trajectories=ts)
            score_record((row,str(source),str(out),identity))
    atomic_json(dest/f'evaluate_{args.rank}.json',dict(status='COMPLETE',scenes=len(selection['scenes'][args.rank::args.shards]),
        rank=args.rank,shards=args.shards,seconds=time.perf_counter()-start,owner=runtime_identity()))


def analyze(args):
    from .updates import families
    from .statistics import paired_policy_summary
    cfg,identity=protocol();summaries=[];scene_results=[];rng=np.random.default_rng(20261019)
    for f in families():
        if f['split']!='audit':continue
        dest=ROOT/'confirmation/severe'/f['id'];selected=read(dest/'selection.json')
        old=[];new=[];logs=[]
        for item in selected['scenes']:
            token=item['token'];o=np.load(dest/f'step_0/scores/{token}.npz')['scores'][:,-1]
            n=np.stack([np.load(dest/f'step_{s}/scores/{token}.npz')['scores'][:,-1] for s in (1,4)])
            old.append(o);new.append(n);logs.append(item['log']);delta=n-o
            for j,s in enumerate((1,4)):
                boot=delta[j][rng.integers(0,25,size=(2000,25))].mean(-1)
                scene_results.append(dict(family=f['id'],token=token,log=item['log'],step=s,
                    initial_delta_J=item['initial_delta_J'][j],confirmation_delta_J=float(delta[j].mean()),
                    lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                    negative_draws=int((delta[j]<0).sum()),positive_draws=int((delta[j]>0).sum()),
                    note='Conditional, independently resampled confirmation subset; per-scene intervals are unadjusted, not population estimates.'))
        if not old:summaries.append(dict(family=f['id'],scenes=0));continue
        delta=np.stack(new).transpose(1,0,2)-np.stack(old)[None]
        summary=paired_policy_summary(delta[None],np.array(logs),2000)
        summaries.append(dict(family=f['id'],scenes=len(old),logs=len(set(logs)),metrics=summary,
            remaining_material_drops=[int((delta[j].mean(-1)<-.02).sum()) for j in range(2)],
            remaining_severe_drops=[int((delta[j].mean(-1)<-.1).sum()) for j in range(2)]))
    atomic_json(ROOT/'analysis/severe_confirmation.json',dict(protocol_sha256=identity,status='ANALYZED',families=summaries,
        scenes=scene_results,scope='Selection on initial5 draws; validation on25 independent paired draws. Conditional subset, not unbiased population rate.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','analyze']);p.add_argument('--family')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
