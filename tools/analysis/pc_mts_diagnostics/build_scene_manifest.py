"""Stratified Navtrain sampling; no checkpoint outcomes enter selection."""
import argparse, concurrent.futures, pickle, sys
from collections import Counter
from common import *

LOG_ROOT=Path('/mnt/navsim/trainval_navsim_logs/trainval')

def inspect_log(path):
    frames=pickle.load(open(path,'rb'));rows=[]
    for i in range(len(frames)-13):
        f=frames[i+3]
        if not len(f['roadblock_ids']):continue
        cmd=np.asarray(f['driving_command'])[:3]
        command=['left','straight','right'][int(cmd.argmax())] if (cmd==1).sum()==1 else 'other'
        rows.append(dict(token=f['token'],log=Path(path).stem,frame_start=i,command=command))
    return rows

def details(row,frames):
    from pyquaternion import Quaternion
    f=frames[row['frame_start']+3:row['frame_start']+12]
    p=np.array([[v['ego2global_translation'][0],v['ego2global_translation'][1],Quaternion(*v['ego2global_rotation']).yaw_pitch_roll[0]] for v in f])
    h=p[0,2];rot=np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    xy=(p[1:,:2]-p[0,:2])@rot;heading=np.arctan2(np.sin(p[1:,2]-h),np.cos(p[1:,2]-h))
    gt=np.column_stack([xy,heading]);curvature=float(np.abs(np.diff(np.unwrap(np.r_[0,heading]))).sum())
    return dict(row,gt=gt.tolist(),curvature=curvature,metric_cache_path=str(METRIC_CACHE/row['log']/'unknown'/row['token']/'metric_cache.pkl'))

def main(args):
    out=OUT/'manifests'/f'scenes_{args.num_scenes}.json'
    if out.exists():
        assert read(out)['seed']==20260913;print(f'Frozen manifest exists: {out}');return
    filt=yaml.safe_load((CODE/'navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text())
    paths=[LOG_ROOT/(log+'.pkl') for log in filt['log_names']]
    assert all(p.is_file() for p in paths)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as pool:
        rows=[r for batch in pool.map(inspect_log,paths,chunksize=2) for r in batch]
    allowed=set(filt['tokens'])
    rows=[r for r in rows if r['token'] in allowed]
    assert {r['token'] for r in rows}==allowed, 'Canonical Navtrain token coverage differs'
    assert len({r['token'] for r in rows})==len(rows)
    counts=Counter(r['command'] for r in rows)
    quotas={c:int(args.num_scenes*n/len(rows)) for c,n in counts.items()}
    for c in sorted(counts,key=lambda c:-(args.num_scenes*counts[c]/len(rows)-quotas[c]))[:args.num_scenes-sum(quotas.values())]:quotas[c]+=1
    selected=[]
    for c,n in quotas.items():
        selected+=sorted((r for r in rows if r['command']==c),key=lambda r:digest([20260913,r['token']]))[:n]
    selected.sort(key=lambda r:digest([20260913,r['token']]))
    frames_by_log={}
    full=[]
    for row in selected:
        if row['log'] not in frames_by_log:frames_by_log[row['log']]=pickle.load(open(LOG_ROOT/(row['log']+'.pkl'),'rb'))
        full.append(details(row,frames_by_log[row['log']]))
    missing=[r['token'] for r in full if not Path(r['metric_cache_path']).exists()]
    # Do not substitute different scenes to hide missing evaluator assets.
    representatives=[]
    for label,subset in [('straight',[r for r in full if r['command']=='straight']),('turn',[r for r in full if r['command'] in ['left','right']]),('high_curvature',[r for r in full if r['curvature']>=.55])]:
        chosen=[r for r in subset if r['token'] not in {v['token'] for v in representatives}][:2]
        if len(chosen)<2:raise RuntimeError(f'Insufficient preregistered {label} representatives')
        representatives += [dict(token=r['token'],kind=label) for r in chosen]
    value=dict(seed=20260913,split='navtrain',population=len(rows),population_commands=dict(counts),selected_commands=dict(quotas),scene_count=len(full),scenes=full,representatives=representatives,missing_metric_cache_tokens=missing)
    save(out,value)
    for size,label in [(20,'smoke'),(50,'benchmark')]:
        child=dict(value,scenes=full[:size],scene_count=size,subset_of=str(out),representatives=[r for r in representatives if r['token'] in {s['token'] for s in full[:size]}])
        save(OUT/'manifests'/f'scenes_{label}.json',child)
    print(json.dumps({k:v for k,v in value.items() if k not in ['scenes']},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--num-scenes',type=int,default=1000);main(p.parse_args())
