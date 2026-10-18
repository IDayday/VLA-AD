"""Numerical, detached trajectory candidates and same-label supervision pairs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from .common import *
from .bank import load_observation,baseline_path,baseline_score_path,score_record
from .updates import families,targets,trajectory_path,score_path,wait_file


def extra_path(token):return ROOT/'candidates/extra'/f'{token}.npz'
def extra_score_path(token):return ROOT/'candidates/scores'/f'{token}.npz'
def library_path(token,package='initial'):
    return ROOT/'candidates'/('library' if package=='initial' else 'confirmation_library')/f'{token}.npz'


def scales(args):
    cfg,identity=protocol();dest=ROOT/'candidates/scales.json'
    if dest.exists():raise ValueError('Scales are immutable after calibration')
    values=[];sources=[]
    for f in families():
        if f['split']=='audit':continue
        for row in targets(f):
            path=trajectory_path(f['id'],4,row['token']);wait_file(path)
            new=np.load(path)['trajectories'];old=np.load(baseline_path(row['token']))['trajectories'][:5]
            values.append(np.sqrt(np.mean((new-old)**2,axis=-2)))
            sources.append(str(path))
    rms=np.concatenate(values);quantiles=np.quantile(rms,[.5,.9],axis=0)
    chosen=np.maximum(quantiles,np.array([.01,.01,.001])[None])
    atomic_json(dest,dict(protocol_sha256=identity,quantiles=[.5,.9],raw_quantiles=quantiles,
        amplitudes=chosen,coordinates=['x_m','y_m','heading_rad'],n_trajectories=len(rms),
        source_split='D_fit/D_cal only, real update step4',source_file_manifest_sha256=digest(sources),
        basis='(t/4s)^2 at t=0.5..4s; origin modification zero'))


def extract(args):
    import torch
    from .runtime import load_agent,action_input,predict
    cfg,identity=protocol();wait_file(ROOT/'candidates/scales.json');scale=read(ROOT/'candidates/scales.json')
    amplitude=np.array(scale['amplitudes']);basis=(np.arange(1,9)/8.)**2
    agent=load_agent(online=False);external=load_agent(online=False,checkpoint=IL);start=time.perf_counter()
    rows=[r for split in ['D_fit','D_cal','D_audit'] for r in scene_rows(split)][args.rank::args.shards]
    for i,row in enumerate(rows):
        token=row['token'];dest=extra_path(token)
        if dest.exists():continue
        data=load_observation(token);inputs=action_input(data['features']);tokens=data['tokens'].cuda()
        trajectories=[];provenance=[]
        for seed in cfg['seeds']['candidates']:
            trajectories.append(predict(agent.action_head,tokens,inputs,token,seed));provenance.append(dict(source='own_extra',seed=seed))
        trajectories.append(predict(external.action_head,tokens,inputs,token,3101));provenance.append(dict(source='released_IL',seed=3101,checkpoint=str(IL)))
        base=np.load(baseline_path(token))['trajectories'][0]
        for level in range(2):
            for coordinate in (0,1):
                for direction in (-1,1):
                    value=base.copy();value[:,coordinate]+=direction*amplitude[level,coordinate]*basis
                    trajectories.append(value);provenance.append(dict(source='smooth',coordinate=coordinate,
                        level=level,direction=direction,amplitude=float(amplitude[level,coordinate]),base_seed=1101))
        for direction in (-1,1):
            value=base.copy();value[:,2]+=direction*amplitude[1,2]*basis
            trajectories.append(value);provenance.append(dict(source='smooth',coordinate=2,level=1,direction=direction,
                amplitude=float(amplitude[1,2]),base_seed=1101))
        trajectories=np.array(trajectories,dtype=np.float32)
        assert trajectories.shape==(15,8,3) and np.isfinite(trajectories).all()
        atomic_npz(dest,dict(token=token,log=row['log_name'],protocol_sha256=identity,provenance=provenance,
            scales_sha256=sha(ROOT/'candidates/scales.json'),whole_trajectory_action=True),trajectories=trajectories)
        if (i+1)%32==0:print('CANDIDATES',args.rank,i+1,len(rows),flush=True)
    atomic_json(ROOT/'candidates'/f'extract_{args.rank}_complete.json',dict(status='COMPLETE',scenes=len(rows),
        seconds=time.perf_counter()-start,protocol_sha256=identity,owner=runtime_identity()))


def score(args):
    _,identity=protocol();start=time.perf_counter()
    rows=[r for s in ['D_fit','D_cal','D_audit'] for r in scene_rows(s)][args.rank::args.shards]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for i in range(0,len(rows),args.workers*4):
            batch=rows[i:i+args.workers*4]
            for r in batch:wait_file(extra_path(r['token']))
            tasks=[(r,str(extra_path(r['token'])),str(extra_score_path(r['token'])),identity) for r in batch]
            list(pool.map(score_record,tasks))
            print('SCORE_CANDIDATES',args.rank,i+len(batch),len(rows),flush=True)
    atomic_json(ROOT/'candidates'/f'score_{args.rank}_complete.json',dict(status='COMPLETE',scenes=len(rows),
        seconds=time.perf_counter()-start,protocol_sha256=identity,owner=runtime_identity()))


def select_pairs(token,scores,sources):
    # Fixed deterministic round-robin strata: consequence magnitude and source pair.
    strata={}
    for a in range(len(scores)):
        for b in range(a+1,len(scores)):
            magnitude='near' if abs(scores[a]-scores[b])<.02 else 'material'
            category=tuple(sorted((sources[a][0]['source'],sources[b][0]['source'])))
            strata.setdefault((magnitude,category),[]).append((a,b))
    for key,values in strata.items():values.sort(key=lambda p:digest([token,list(p),'pair']))
    keys=sorted(strata,key=lambda k:digest([token,k,'stratum']));selected=[]
    while len(selected)<32 and any(strata.values()):
        for key in keys:
            if not strata[key]:continue
            a,b=strata[key].pop(0)
            if int(digest([token,a,b,'orientation']),16)%2:a,b=b,a
            selected.append((a,b))
            if len(selected)==32:break
    return np.array(selected,dtype=np.int64)


def assemble(args):
    _,identity=protocol();rows=scene_rows(args.split)
    related=[f for f in families() if 'D_'+f['split']==args.split and (args.package=='confirmation' or not f.get('confirmation'))]
    members={f['id']:{r['token'] for r in targets(f)} for f in related}
    summaries=[]
    for row in rows:
        token=row['token'];dest=library_path(token,args.package);trajectories=[];labels=[];provenance=[];index={}
        def add(path,score_file,sources,n=None):
            wait_file(path);wait_file(score_file)
            ts=np.load(path)['trajectories'];ys=np.load(score_file)['scores']
            for j,(traj,label,origin) in enumerate(zip(ts[:n],ys[:n],sources)):
                key=hashlib.sha256(traj.astype(np.float32).tobytes()).hexdigest()
                if key in index:
                    k=index[key]
                    if not np.allclose(labels[k],label,atol=1e-6,rtol=0):raise ValueError('Identical trajectory has conflicting canonical labels')
                    provenance[k].append(origin)
                else:
                    index[key]=len(trajectories);trajectories.append(traj);labels.append(label);provenance.append([origin])
        add(baseline_path(token),baseline_score_path(token),[dict(source='baseline',seed=s) for s in range(1101,1106)],5)
        extra_meta=json.loads(str(np.load(extra_path(token))['metadata'])) if extra_path(token).exists() else None
        if extra_meta is None:wait_file(extra_path(token));extra_meta=json.loads(str(np.load(extra_path(token))['metadata']))
        add(extra_path(token),extra_score_path(token),extra_meta['provenance'])
        for f in related:
            if token not in members[f['id']]:continue
            for step in (1,4):
                add(trajectory_path(f['id'],step,token),score_path(f['id'],step,token),
                    [dict(source='real_update',family=f['id'],step=step,seed=s) for s in range(1101,1106)])
        ts=np.array(trajectories,dtype=np.float32);ys=np.array(labels);pairs=select_pairs(token,ys[:,-1],provenance)
        assert len(pairs)>0
        atomic_npz(dest,dict(token=token,log=row['log_name'],split=args.split,protocol_sha256=identity,
            candidate_ids=list(index),provenance=provenance,score_fields=FIELDS),
            trajectories=ts,labels=ys,pairs=pairs,target_delta=ys[pairs[:,0],-1]-ys[pairs[:,1],-1])
        delta=ys[:,-1]-ys[0,-1]
        summaries.append(dict(token=token,candidates=len(ts),pairs=len(pairs),range=float(np.ptp(ys[:,-1])),
            both_signs=bool((delta>.02).any() and (delta<-.02).any()),zero_scores=int((ys[:,-1]<=1e-12).sum()),
            min_delta=float(delta.min()),max_delta=float(delta.max())))
    prefix='confirmation_' if args.package=='confirmation' else ''
    atomic_json(ROOT/'candidates'/f'{prefix}{args.split}_complete.json',dict(status='COMPLETE',protocol_sha256=identity,
        scenes=len(rows),summaries=summaries,nonzero_scenes=sum(s['range']>1e-6 for s in summaries),
        material_both_signs=sum(s['both_signs'] for s in summaries)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['scales','extract','score','assemble'])
    p.add_argument('--split',default='D_fit');p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--package',choices=['initial','confirmation'],default='initial')
    args=p.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
