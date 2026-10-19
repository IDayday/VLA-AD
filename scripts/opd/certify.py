"""Independent final bank/hash and common-initialization certificates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.analyze import EXPERIMENTS


def models():
    import torch
    torch.set_num_threads(1)
    from navsim.agents.recogdrive.opd.checkpoint_io import read_state
    c=config();raw=read_state(c['student_init']);initial={k.removeprefix('agent.action_head.'):v for k,v in raw.items()}
    records=[];schedules={}
    for e in EXPERIMENTS:
        for seed in c['train_seeds']:
            dest=ROOT/'runs'/f'train_{e}_{seed}';completed=read(dest/'complete.json');cfg=completed['config'];chosen=read(ROOT/'selection/tuning.json')['selected'][e]
            assert cfg['steps']==completed['optimizer_updates']==1000 and cfg['global_batch']==8 and cfg['world']==1
            assert cfg['lr']==chosen['lr'] and completed['teacher_unchanged'] and completed['anchor_unchanged']
            zero=torch.load(dest/'checkpoint_0.pt',map_location='cpu',weights_only=False)
            inactive={k for k in zero['student'] if k.startswith('action_aware_aux_head.')}
            assert not read(ROOT/'audit/architecture.json')['flags']['use_action_aware_aux']
            assert set(zero['student'])-inactive==set(initial) and all(torch.equal(zero['student'][k],v) for k,v in initial.items())
            assert not zero['optimizer']['state'],'Initial optimizer must be fresh for all methods'
            final=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
            changed=[k for k,v in initial.items() if not torch.equal(final['student'][k],v)]
            assert changed and all(float(v['step'])==1000 for v in final['optimizer']['state'].values())
            assert sha(dest/'latest.pt')==completed['checkpoint_sha256']
            schedule=sha(dest/'training_schedule.json');schedules.setdefault(seed,set()).add(schedule)
            records.append(dict(experiment=e,seed=seed,initialization_exactly_official_IL=True,initial_optimizer_empty=True,changed_tensors=len(changed),
                checkpoint_sha256=completed['checkpoint_sha256'],schedule_sha256=schedule,optimizer_updates=1000,
                inactive_noncore_targets=sorted(inactive),inactive_reason='Explicitly disabled/frozen auxiliary module, absent from official checkpoints; never used by native inference or this training objective'))
            del zero,final
    assert all(len(v)==1 for v in schedules.values()),'Exposure schedules differ between methods'
    write(ROOT/'audit/formal_model_certificate.json',dict(passed=True,policies=len(records),records=records,all_methods_share_schedule_per_seed=True));print('MODEL_CERTIFICATE_PASS',len(records))


def evaluation(split):
    from scripts.opd.evaluate import validate_metadata
    cfg=config();selected=rows(split);names=['E0']
    methods=EXPERIMENTS if split!='navtest' else read(ROOT/'selection/navtest.json')['methods']
    names.extend(f'train_{e}_{s}' for e in methods for s in cfg['train_seeds']);summaries=[]
    for name in names:
        dest=ROOT/'evaluation'/split/name;markers=[read(p) for p in dest.glob('complete_*.json')]
        assert markers and len(markers)==markers[0]['shards'] and sum(v['rows'] for v in markers)==len(selected)
        assert {v['rank'] for v in markers}==set(range(markers[0]['shards']))
        expected={r['token'] for r in selected}
        assert {p.stem for p in (dest/'scores').glob('*.json')}==expected
        assert {p.stem for p in (dest/'predictions').glob('*.npz')}==expected
        checkpoint={n:read(ROOT/'audit/checkpoints.json')[n]['sha256'] for n in ['IL','RL','COPY']} if name=='E0' else {name:read(ROOT/'runs'/name/'complete.json')['checkpoint_sha256']}
        models=set(checkpoint)|({'IL_screen'} if name=='E0' else set())
        sums={m:np.zeros(7) for m in models}
        for row in selected:
            token=row['token'];path=dest/'predictions'/f'{token}.npz';score=read(dest/'scores'/f'{token}.json')
            assert score['token']==token and score['log']==row['log_name'] and score['predictions_sha256']==sha(path)
            validate_metadata(score['model_metadata'],checkpoint,cfg,split,token)
            with np.load(path) as predictions:
                meta=json.loads(str(predictions['metadata']));assert meta==score['model_metadata']
                assert set(predictions.files)==models|{'metadata'} and set(score['scores'])==models
                for m in models:
                    tr=predictions[m];values=np.array(score['scores'][m]['scores'])
                    assert tr.shape==(5,8,3) and np.isfinite(tr).all()
                    assert values.shape==(5,7) and np.isfinite(values).all() and (values>=-1e-6).all() and (values<=1+1e-6).all()
                    sums[m]+=values.sum(0)
                if name=='E0':
                    assert np.array_equal(predictions['RL'],predictions['COPY'])
                    assert score['scores']['RL']['scores']==score['scores']['COPY']['scores']
        summaries.append(dict(name=name,scenes=len(selected),seeds=5,checkpoint_sha256=checkpoint,components={m:(v/len(selected)/5).tolist() for m,v in sums.items()}))
        print('BANK_CERTIFIED',split,name,len(selected),flush=True)
    write(ROOT/'audit'/f'evaluation_certificate_{split}.json',dict(passed=True,split=split,policies=len(names),scene_policy_pairs=len(names)*len(selected),records=summaries,
        checks=['Exact full membership','No duplicate or missing files','Every prediction SHA','Checkpoint/config/scene/seed bindings','Finite complete trajectories and components','E0 RL/COPY exact equality']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['models','evaluation']);p.add_argument('--split',default='dev');a=p.parse_args()
    if a.action=='models':models()
    else:evaluation(a.split)
