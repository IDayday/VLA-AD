"""Replay a SAVED scene-fixed choice, export compact or full teacher targets."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent


def main(a):
    if a.population=='test' and not a.allow_test_diagnostic:raise ValueError('Test export needs --allow-test-diagnostic and must not become student training data')
    if not 0<=a.rank<a.world_size:raise ValueError('Invalid configurable shard')
    from interface_teacher.runtime import bootstrap,read_rows,Runtime
    bootstrap(a.paths,a.out)
    from portable_common import atomic,sha
    from interface_teacher import Library,DecisionTable
    from seeds import scene_seed
    import torch
    torch.set_num_threads(1)
    paths=json.loads(a.paths.read_text());rows=read_rows(a.population,paths)
    if a.tokens:
        tokens=set(a.tokens.split(','));rows=[r for r in rows if r['token'] in tokens]
        if {r['token'] for r in rows}!=tokens:raise ValueError('Unknown token requested')
    if a.limit:rows=rows[:a.limit]
    rows=rows[a.rank::a.world_size]
    if not rows:raise ValueError('No scene for this shard')
    lib=Library(a.library);table=DecisionTable(a.library,a.population,allow_test=a.allow_test_diagnostic)
    protocol=dict(library=a.library,population=a.population,source=a.source,
        paths_sha256=sha(a.paths),catalog_sha256=sha(ROOT/'assets/CATALOG.json'),
        library_sha256=sha(ROOT/f'assets/libraries/{a.library}.json'),
        decisions_sha256=table.manifest['choice_table']['sha256'],noise_indices=list(range(5)),
        scene_fixed=True,saved_posthoc_selection=True,save_full_interface=a.save_full_interface,score=a.score,
        new_search=False,TEST_is_diagnostic=a.population=='test')
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    atomic(out/f'PROTOCOL_rank{a.rank:04d}.json',dict(protocol,protocol_sha256=digest,rank=a.rank,world_size=a.world_size))
    engine=Runtime(a.source);atomic(out/f'MODEL_rank{a.rank:04d}.json',engine.identity)
    start=time.monotonic();done=0
    for row in rows:
        token=row['token'];folder=out/'cases'/token;marker=folder/'COMPLETE.json'
        if marker.exists():
            prev=json.loads(marker.read_text())
            if prev['protocol_sha256']!=digest:raise ValueError('Existing output belongs to another protocol')
            if sha(folder/'TARGETS.npz')!=prev['arrays_sha256']:raise ValueError('Existing output is corrupted')
            done+=1;continue
        features,hidden,base,image,text,ids=engine.scene(row)
        choice=table.choice(token);target=lib.targets(base,image,text,[choice]);teacher=target['interfaces']
        trajectories=np.stack([engine.trajectory(features,hidden,teacher,token,j) for j in range(5)])
        base_trajectories=np.stack([engine.trajectory(features,hidden,base,token,j) for j in range(5)])
        valid=image|text
        values=dict(choice=np.array(choice),actual_role_shifts=target['actual_role_shifts'].cpu().numpy(),
            requested_shifts_float64=target['requested_shifts_float64'],cap_RMS=target['cap_RMS'].cpu().numpy(),
            actual_RMS=target['actual_RMS'].cpu().numpy(),image_mask=image.cpu().numpy(),text_mask=text.cpu().numpy(),
            token_ids=ids.cpu().numpy(),valid_positions=np.flatnonzero(valid.cpu().numpy()),
            teacher_trajectories=trajectories,Base_trajectories=base_trajectories,
            noise_indices=np.arange(5),integer_seeds=np.array([scene_seed(token,j) for j in range(5)],np.int64))
        if a.save_full_interface:
            values.update(teacher_interface=teacher[0].cpu().numpy(),Base_interface=base[0].cpu().numpy())
        scored=None
        if a.score:
            from repair_score import score_group
            task=dict(metric_cache=row['metric_cache'],metric_sha256=sha(row['metric_cache']),scoring=engine.cfg['scoring'],
                records=[dict(arm=arm,seed_index=j,trajectory=t.tolist()) for arm,ts in [('Base',base_trajectories),('teacher',trajectories)] for j,t in enumerate(ts)])
            scored=score_group(task);values['rescored_PDMS']=np.array([r['score']['score']*100 for r in scored]).reshape(2,5)
        folder.mkdir(parents=True,exist_ok=True)
        tmp=folder/f'.TARGETS.{os.getpid()}.npz';np.savez_compressed(tmp,**values);tmp.replace(folder/'TARGETS.npz')
        atomic(marker,dict(status='COMPLETE',token=token,session=row['session'],index=int(row['index']),
            protocol_sha256=digest,library=a.library,choice=choice,saved_selection=table.row(token),
            source=a.source,arrays_sha256=sha(folder/'TARGETS.npz'),
            actual_Base_interface_sha256=hashlib.sha256(base[0].cpu().numpy().tobytes()).hexdigest(),
            runtime_scores_computed=a.score,
            saved_scores_are_historical_not_new_measurements=True))
        done+=1
        state=dict(status='RUNNING',rank=a.rank,completed=done,total=len(rows),elapsed_seconds=time.monotonic()-start)
        atomic(out/f'STATE_rank{a.rank:04d}.json',state);print(json.dumps(state),flush=True)
    atomic(out/f'COMPLETE_rank{a.rank:04d}.json',dict(status='COMPLETE',scenes=done,protocol_sha256=digest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',default='D_C8_double');p.add_argument('--population',choices=['train','test'],default='train')
    p.add_argument('--source',choices=['raw','cached'],default='raw');p.add_argument('--allow-test-diagnostic',action='store_true')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--tokens')
    p.add_argument('--save-full-interface',action='store_true');p.add_argument('--score',action='store_true')
    args=p.parse_args()
    try:main(args)
    except BaseException:
        import traceback
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/f'EXCEPTION_rank{args.rank:04d}.txt').write_text(traceback.format_exc())
        raise
