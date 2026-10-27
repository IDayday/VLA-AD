"""Local Native candidate manifest preparation, with original per-scene menus."""
import csv
import gzip
from pathlib import Path
from gt_numeric import atomic,read,sha,digest,stamp,bind_function,G5,STAGES,ReferenceNoise

OLD=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026')

def ref(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha(path)}

def initialize_local(args,root,configuration_tasks):
    if (root/'runs/QUEUE.json').exists():raise ValueError('Local GT root already initialized; preserve queue and resume')
    candidate_path=Path(args.candidates).resolve();candidate=read(candidate_path)
    selected=candidate['candidates'];count=int(candidate['count'])
    if len(selected)!=count or len({r['token'] for r in selected})!=count or any(r['population']!='train' or not r['Base_S0']<.7 for r in selected):
        raise ValueError('GT candidates must be exactly the verified local Train Base S0<0.7 set')
    drain_path=OLD/'engineering/local_candidate_recheck_20261027/OLD1850_SCOPE_DRAIN_COMPLETE.json';drain=read(drain_path)
    if drain.get('status')!='ALL148_ACTUAL_EXITS_ZERO' or drain.get('actual_exit_zero')!=148 or drain.get('bad'):raise ValueError('Original scope must be actually drained before reuse')
    for row in drain['records']:
        state=read(row['status'])
        if state.get('returncode')!=0 or state.get('state') not in ('done','failed'):raise ValueError('An original source actor is still live or failed')
    train_path=Path(args.train_manifest)
    with (gzip.open(train_path,'rt') if train_path.suffix=='.gz' else train_path.open()) as stream:
        train={r['token']:r for r in csv.DictReader(stream)}
    registry=read(args.metric_registry);paths=read(args.paths);registry_sha=sha(args.metric_registry)
    old={r['token']:r for r in read(OLD/'native_gt/manifests/SCENES.json')}
    rows=[];intersection=[]
    for item in selected:
        src=train[item['token']];metric=registry['entries'][item['token']]
        if int(src['index'])!=int(item['global_index']) or src['session']!=item['session'] or metric['reference_sha256']!=src['metric_sha256'] or metric['relative']!=src['metric_rel']:
            raise ValueError('Local candidate token/index/session/metric identity changed')
        row=dict(item,population='train',subset='exploration',legacy_best=None,legacy_configurations={},
             metric_cache=str(Path(paths['navtrain_metric'])/src['metric_rel']),metric_sha256=metric['local_sha256'],
             metric_reference_sha256=metric['reference_sha256'],local_metric_registry_sha256=registry_sha)
        if item['token'] in old:
            previous=old[item['token']]
            if previous['global_index']!=row['global_index']:raise ValueError('Intersection changed token/index')
            intersection.append(int(row['global_index']))
        rows.append(row)
    source=Path(args.source_root).resolve()
    noise=ReferenceNoise('/mnt/project/recogdrive-first42-a800-20261022/reference/noise/PROTOCOL.json',
                         '/mnt/project/recogdrive-first42-a800-20261022/reference/manifests/navtrain_scenes.json',
                         '/mnt/project/recogdrive-first42-a800-20261022/reference/noise/navtrain_noise.npy')
    atomic(root/'manifests/SCENES.json',rows)
    configurations,identity=bind_function(source/G5/'g5_runtime.py','configurations',{'Path':Path})
    tasks=[{'id':f"scene_{r['global_index']:06d}",'kind':'baseline','index':r['global_index'],'configs':configurations()} for r in rows]
    atomic(root/'runs/baseline/TASKS.json',tasks)
    atomic(root/'runs/baseline/SPEC.json',{'kind':'baseline','case_count':count,'enabled':True})
    atomic(root/'PROTOCOL.json',{'status':'LOCAL_NATIVE_CANDIDATE_FULL_GT_REGISTERED','created':stamp(),
       'source_root':str(source),'local_candidates':ref(candidate_path),'candidate_count':count,'train_failures':count,
       'candidate_rule':'local Native Base S0<0.7 on full Train; original g5_prepare rule',
       'source_train_manifest':ref(train_path),'local_metric_registry':ref(args.metric_registry),
       'native_base_root':str(Path(args.base_root).resolve()),'original_configurations_binding':identity,'stages':STAGES,
       'reuse_old_intersection_indices':intersection,'reuse_source':str(OLD/'native_gt'),'original_scope_drain':ref(drain_path),
       'reference_noise':noise.identity,'selection_noise':'actual S0 PDMS','transfer_noise_indices':[1,2],
       'transfer_not_used_for_selection':True,'no_validation_or_test_optimization':True,
       'all15_original_stages_required':True,'library_to_fit':'D_C8_double only'})
    atomic(root/'runs/QUEUE.json',{'stages':['baseline'],'active':True})
    configuration_tasks(root,'baseline')
