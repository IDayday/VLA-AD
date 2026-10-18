"""Clearly privileged simple control and outcome-blind grouped image permutation."""
from __future__ import annotations
import argparse
import numpy as np
from .common import *
from .bank import load_observation,baseline_path,baseline_score_path
from .updates import families,targets,trajectory_path,score_path


def shuffle(args):
    _,identity=protocol();groups={}
    for row in scene_rows(args.split):
        token=row['token'];value=load_observation(token);state=value['features']['status_feature'].numpy()[0]
        trajectory=np.load(baseline_path(token))['trajectories'][:5].mean(0)
        key=(int(np.argmax(state[:4])),int(np.linalg.norm(state[4:6])//2.5),
             int(trajectory[-1,0]//5),int(trajectory[-1,1]//2),int(trajectory[-1,2]//.2))
        groups.setdefault(str(key),[]).append(token)
    donors={}
    for values in groups.values():
        values.sort(key=lambda token:digest([token,'image_shuffle']))
        donors.update({token:values[(i+1)%len(values)] for i,token in enumerate(values)})
    atomic_json(ROOT/'manifests'/f'{args.split}_observation_shuffle.json',dict(protocol_sha256=identity,donors=donors,
        groups=groups,moved=sum(a!=b for a,b in donors.items()),total=len(donors),labels_used=False,
        grouping='command, speed2.5m/s, terminal trajectory x5m/y2m/heading0.2rad bins',
        intervention='Permute current images within groups; keep target ego history/command and numerical trajectories. Input-dependence diagnostic, not causal factor identification.'))


def control_data(split):
    fs=[f for f in families() if 'D_'+f['split']==split];rows=targets(fs[0]);xs=[];ys=[];records=[]
    for row in rows:
        token=row['token'];payload=load_observation(token)
        state=np.concatenate([payload['features']['status_feature'].numpy().ravel(),payload['features']['history_trajectory'].numpy().ravel()])
        old=np.load(baseline_path(token))['trajectories'][:5];base=np.load(baseline_score_path(token))['scores'][:5,-1]
        scene_x=[];scene_y=[]
        for f in fs:
            family_x=[];family_y=[]
            for step in (1,4):
                delta=np.load(trajectory_path(f['id'],step,token))['trajectories']-old
                features=np.concatenate([base[:,None],np.sqrt(np.mean(delta**2,axis=1)),delta.mean(1),np.broadcast_to(state,(5,len(state)))],axis=1)
                family_x.append(features);family_y.append(np.load(score_path(f['id'],step,token))['scores'][:,-1]-base)
            scene_x.append(family_x);scene_y.append(family_y)
        xs.append(scene_x);ys.append(scene_y);records.append(dict(token=token,log=row['log_name']))
    return np.array(xs),np.array(ys),records


def ridge(args):
    _,identity=protocol();train_x,train_y,_=control_data('D_fit');cal_x,cal_y,_=control_data('D_cal')
    choices=[];models=[]
    x=train_x.reshape(-1,train_x.shape[-1]);y=train_y.ravel();mean=x.mean(0);std=x.std(0);std[std==0]=1.
    z=(x-mean)/std;ym=y.mean()
    for alpha in [.1,1.,10.]:
        coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-ym))
        pred=((cal_x.reshape(-1,cal_x.shape[-1])-mean)/std)@coef+ym
        choices.append(dict(alpha=alpha,cal_RMSE=float(np.sqrt(np.mean((pred-cal_y.ravel())**2)))));models.append(coef)
    selected=min(range(3),key=lambda i:(choices[i]['cal_RMSE'],i));model=models[selected]
    # Choice is fixed before touching audit outcomes.
    atomic_json(ROOT/'controls/ridge_selection.json',dict(protocol_sha256=identity,choices=choices,selected=selected,
        privileged_input='True baseline PDMS; remaining features are numerical trajectory change and observable ego state.'))
    x,y,records=control_data('D_audit');prediction=(((x.reshape(-1,x.shape[-1])-mean)/std)@model+ym).reshape(y.shape)
    atomic_npz(ROOT/'controls/ridge_audit.npz',dict(protocol_sha256=identity,records=records,selected_alpha=choices[selected]['alpha']),
        prediction=prediction,target=y)
    atomic_json(ROOT/'controls/ridge_complete.json',dict(status='COMPLETE',RMSE=float(np.sqrt(np.mean((prediction-y)**2))),
        MAE=float(np.mean(abs(prediction-y))),selection='D_cal only',privileged=True,owner=runtime_identity()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['shuffle','ridge']);p.add_argument('--split',default='D_audit')
    args=p.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
