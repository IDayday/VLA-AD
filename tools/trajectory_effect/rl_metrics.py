"""Fixed-budget and calibration-only matched-gain policy comparisons."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import baseline_score_path
from .rl_runner import run_dir
from .statistics import paired_policy_summary,log_weights,estimate,interval


def load(arm,split):
    cfg,_=protocol();rows=scene_rows(split);old=np.stack([np.load(baseline_score_path(r['token']))['scores'][:5] for r in rows])
    values=[]
    for seed in cfg['seeds']['training']:
        runs=[]
        for step in [100,250,500]:
            assert read(run_dir(arm,seed)/f'score_{split}_{step}.json')['status']=='COMPLETE'
            runs.append(np.stack([np.load(run_dir(arm,seed)/f'step_{step}/scores/{r["token"]}.npz')['scores'] for r in rows]))
        values.append(runs)
    new=np.array(values);return new-old[None,None],old,new,rows


def select():
    cfg,identity=protocol();path=ROOT/'rl/matched_selection.json'
    if path.exists():return read(path)
    choices={};cal={}
    for arm in cfg['rl']['arms']:
        delta,_,_,_=load(arm,'D_cal');g=delta[...,-1].mean((-1,-2));cal[arm]=g.tolist();choices[arm]=[]
        for seed,values in zip(cfg['seeds']['training'],g):
            eligible=[j for j,value in enumerate(values) if .002<=value<=.003]
            j=min(eligible,key=lambda j:(abs(values[j]-.002),j)) if eligible else None
            choices[arm].append(dict(seed=seed,step=[100,250,500][j] if j is not None else None,
                cal_G=float(values[j]) if j is not None else None,status='MATCHED' if j is not None else 'NO_COMPARABLE_EFFECTIVE_GAIN'))
    result=dict(status='LOCKED',protocol_sha256=identity,choices=choices,cal_gain_at_100_250_500=cal,
        audit_outcomes_used_for_selection=False,target=.002,tolerance=.001)
    atomic_json(path,result);return result


def compare(q,e,logs,reps=2000):
    # [training seed,checkpoint,scene,inference seed], paired all dimensions.
    rng=np.random.default_rng(20261019);point=(estimate(q.mean(-1))-estimate(e.mean(-1))).mean(0);draws=[]
    for _ in range(reps):
        seeds=rng.integers(0,len(q),len(q));z=rng.integers(0,q.shape[-1],q.shape[-1]);w=log_weights(logs,rng)
        draws.append((estimate(q[seeds][...,z].mean(-1),w)-estimate(e[seeds][...,z].mean(-1),w)).mean(0))
    return interval(point,np.array(draws))


def analyze():
    cfg,identity=protocol();selection=select();all_data={};arms={}
    for arm in cfg['rl']['arms']:
        delta,old,new,rows=load(arm,'D_audit');logs=np.array([r['log_name'] for r in rows]);d=delta[...,-1]
        all_data[arm]=d
        arms[arm]=dict(fixed_budget=paired_policy_summary(d,logs),
            positive_to_zero=(((old[None,None,...,-1]>1e-12)&(new[...,-1]<=1e-12)).sum((2,3))).tolist(),
            high_to_nearzero=(((old[None,None,...,-1]>=.9)&(new[...,-1]<=.01)).sum((2,3))).tolist(),
            single_output_material_drop=(d<-.02).mean((2,3)).tolist(),
            new_violations={FIELDS[k]:((old[None,None,...,k]>=1-1e-6)&(new[...,k]<1-1e-6)).sum((2,3)).tolist() for k in [0,1,3,4,5]})
    budget=compare(all_data['RL-Q'],all_data['RL-E'],logs);matched=None
    if all(r['step'] is not None for a in ['RL-Q','RL-E'] for r in selection['choices'][a]):
        selected={a:np.stack([all_data[a][s,[100,250,500].index(r['step'])] for s,r in enumerate(selection['choices'][a])])[:,None] for a in ['RL-Q','RL-E']}
        matched=dict(arms={a:paired_policy_summary(d,logs) for a,d in selected.items()},comparison=compare(selected['RL-Q'],selected['RL-E'],logs))
        q=matched['arms']['RL-Q']['estimate'][0];e=matched['arms']['RL-E']['estimate'][0]
        matched['both_audit_gains_positive']=q[0]>0 and e[0]>0;matched['material_M_minus_threshold']=max(.0005,.1*q[2])
    result=dict(status='ANALYZED',protocol_sha256=identity,steps=[100,250,500],columns=['G','M_plus','M_minus','material_drop_fraction'],
        arms=arms,same_budget_Q_minus_E=budget,matched_gain=matched,selection=selection,
        unmatched_note=None if matched else 'Not all three seeds reached comparable positive calibration gain; no matched-gain advantage claim.')
    atomic_json(ROOT/'analysis/rl.json',result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','analyze']);a=p.parse_args();globals()[a.action]()


if __name__=='__main__':main()
