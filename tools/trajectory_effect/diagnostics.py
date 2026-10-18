"""Descriptive candidate coverage and independently screened high-score changes."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import baseline_path,baseline_score_path
from .candidates import extra_path,extra_score_path
from .updates import families,targets,trajectory_path,score_path
from .statistics import paired_policy_summary


def high():
    cfg,identity=protocol();screen=read(ROOT/'baseline/high_screen/complete.json');stable=set(screen['stable_tokens']);values=[]
    for f in families():
        if f['split']!='audit':continue
        rows=[r for r in targets(f) if r['token'] in stable];logs=np.array([r['log_name'] for r in rows])
        old=np.stack([np.load(baseline_score_path(r['token']))['scores'][:5,-1] for r in rows])
        new=np.stack([np.stack([np.load(score_path(f['id'],s,r['token']))['scores'][:,-1] for r in rows]) for s in (1,4)])
        values.append(dict(family=f['id'],stable_scenes=len(rows),summary=paired_policy_summary((new-old)[None],logs)))
    atomic_json(ROOT/'analysis/stable_high.json',dict(protocol_sha256=identity,selection=screen,families=values))


def coverage():
    cfg,identity=protocol();summary={};arrays={}
    for split in ['D_fit','D_cal','D_audit']:
        groups={};high_cliffs=[]
        for row in scene_rows(split):
            token=row['token'];old=np.load(baseline_path(token))['trajectories'][0];base=float(np.load(baseline_score_path(token))['scores'][0,-1])
            extra=np.load(extra_path(token));score=np.load(extra_score_path(token))['scores'][:,-1]
            meta=json.loads(str(extra['metadata']));change=extra['trajectories']-old
            rms=np.sqrt(np.mean(change**2,axis=1))
            for i,source in enumerate(meta['provenance']):
                name=source['source']+(('_level'+str(source['level'])) if source['source']=='smooth' else '')
                groups.setdefault(name,[]).append([float(score[i]-base),*rms[i].tolist()])
                if source['source']=='smooth' and base>=.9 and score[i]-base<=-.1:
                    high_cliffs.append(dict(token=token,source=source,base_score=base,candidate_score=float(score[i]),rms_xyz=rms[i].tolist()))
        for f in families():
            if 'D_'+f['split']!=split:continue
            for row in targets(f):
                token=row['token'];old=np.load(baseline_path(token))['trajectories'][:5];base=np.load(baseline_score_path(token))['scores'][:5,-1]
                new=np.load(trajectory_path(f['id'],4,token))['trajectories'];score=np.load(score_path(f['id'],4,token))['scores'][:,-1]
                rms=np.sqrt(np.mean((new-old)**2,axis=1))
                groups.setdefault('real_update_step4',[]).extend(np.column_stack([score-base,rms]).tolist())
        summary[split]=dict(groups={k:dict(count=len(v),quantile_levels=[0,.1,.5,.9,1],
            columns=['delta_score','rms_x_m','rms_y_m','rms_heading_rad'],quantiles=np.quantile(v,[0,.1,.5,.9,1],axis=0),
            material_improvement_fraction=float((np.array(v)[:,0]>.02).mean()),material_degradation_fraction=float((np.array(v)[:,0]<-.02).mean())) for k,v in groups.items()},
            high_small_perturbation_cliffs=high_cliffs,cliff_count=len(high_cliffs))
        for k,v in groups.items():arrays[split+'_'+k]=np.array(v)
    atomic_json(ROOT/'analysis/candidate_coverage.json',dict(protocol_sha256=identity,splits=summary,
        note='Artificial perturbation sign changes are not evidence that a parameter update harmed another scene. High-score candidate counts here are descriptive; stable policy comparisons use independent screening seeds.'))
    atomic_npz(ROOT/'analysis/candidate_coverage.npz',dict(protocol_sha256=identity),**arrays)


def tracking():
    _,identity=protocol();records=[];tokens=[];rms=[];maximum=[];heading=[]
    for row in [r for s in ['D_fit','D_cal','D_audit'] for r in scene_rows(s)]:
        token=row['token'];requested=np.load(extra_path(token))['trajectories']
        states=np.load(extra_score_path(token))['simulated_states'];origin=states[0,0,:3]
        c,s=np.cos(origin[2]),np.sin(origin[2]);rotation=np.array([[c,-s],[s,c]])
        tracked=(states[:,5::5,:2]-origin[:2])@rotation
        error=np.linalg.norm(tracked-requested[:,:,:2],axis=-1)
        angular=np.angle(np.exp(1j*(states[:,5::5,2]-origin[2]-requested[:,:,2])))
        tokens.append(token);rms.append(np.sqrt(np.mean(error**2,axis=-1)));maximum.append(error.max(-1));heading.append(np.sqrt(np.mean(angular**2,axis=-1)))
    atomic_npz(ROOT/'analysis/candidate_tracking.npz',dict(protocol_sha256=identity,
        definition='Requested versus simulated rear-axle trajectory at0.5..4.0s, continuous per-candidate error; no candidate discarded.',
        filtering=False),tokens=np.array(tokens),xy_RMS_m=np.array(rms),xy_max_m=np.array(maximum),heading_RMS_rad=np.array(heading))
    atomic_json(ROOT/'analysis/candidate_tracking.json',dict(protocol_sha256=identity,candidates=len(tokens)*15,
        quantiles=[.5,.9,.99,1.],xy_RMS_m=np.quantile(rms,[.5,.9,.99,1.]),xy_max_m=np.quantile(maximum,[.5,.9,.99,1.]),
        heading_RMS_rad=np.quantile(heading,[.5,.9,.99,1.]),categorical_difficulty_threshold=None,
        note='No environment-defined tracking-failure threshold was available; every candidate is annotated with continuous tracking error. No retrospective threshold changes primary outcomes.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['high','coverage','tracking']);a=p.parse_args();globals()[a.action]()


if __name__=='__main__':main()
