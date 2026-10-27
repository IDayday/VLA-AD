"""Content-addressed completed Native GT reuse; original kernels are unchanged."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

OLD=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026')
sys.path.insert(0,str(OLD/'code'))
from gt_numeric import atomic,digest,read,sha,stamp

_CATALOG={}
_IDENTITY={}

def ref(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha(path)}

def semantic_config(config):
    """Retain every optimizer input; replace only location by verified content."""
    value=copy.deepcopy(config);value.pop('id',None)
    if 'initial' in value:
        init=value['initial'];value['initial']={key:init[key] for key in ('sha256','candidate_index') if key in init}
    for field in ('lift_post','project_pre'):
        if field in value:
            source=value[field];cfg=source['config']
            value[field]={'arrays_sha256':source['arrays_sha256'],
                          'config':{'block':cfg['block'],'cap':cfg['cap'],'mode':cfg.get('mode','shared')}}
    return value

def semantic_key(index,config):return digest({'index':int(index),'config':semantic_config(config)})

def begin_reuse(root,stage):
    root=Path(root);folder=root/'runs'/stage
    if stage=='export_post':return set()
    protocol=read(root/'PROTOCOL.json')
    indices=set(protocol['reuse_old_intersection_indices'])
    if not (folder/'REUSE_IMPORT_COMPLETE.json').exists():
        atomic(folder/'REUSE_PENDING.json',{'status':'IMPORTING_VERIFIED_COMPLETED_CONFIGURATIONS',
                 'blocked_old_intersection_indices':sorted(indices),'candidate_manifest':protocol['local_candidates']})
    return indices

def _source_identity(root,receipt):
    worker=receipt['worker']
    if worker not in _IDENTITY:
        identity=read(Path(root)/'workers'/worker/'NUMERICS_BINDINGS.json')
        if digest(identity)!=receipt['numerics_binding_sha256']:raise ValueError('Original completed worker numerical identity changed')
        for binding in identity['bindings']:
            if sha(binding['file'])!=binding['file_sha256']:raise ValueError('Original GT mathematical source changed')
        _IDENTITY[worker]=identity
    identity=_IDENTITY[worker]
    if digest(identity)!=receipt['numerics_binding_sha256']:raise ValueError('Worker changed numerical identity between configurations')
    return identity

def stage_catalog(stage,indices):
    if stage in _CATALOG:return _CATALOG[stage]
    source=OLD/'native_gt';folder=source/'runs'/stage
    if not (folder/'TASKS.json').exists():_CATALOG[stage]={};return {}
    tasks={f"{s['id']}__c{position:04d}":(int(s['index']),config)
           for s in read(folder/'TASKS.json') if int(s['index']) in indices
           for position,config in enumerate(s['configs'])}
    files=[p for p in (folder/'configuration_receipts').glob('*.json') if p.stem in tasks]
    def inspect(path):
        receipt=read(path);index,config=tasks[path.stem]
        task_path=folder/'configuration_tasks'/path.name;task=read(task_path)
        if receipt['status']!='COMPLETE' or receipt['config_sha256']!=digest(config) or receipt['task_sha256']!=digest(task) or task['config']!=config:
            raise ValueError('Original completed GT configuration/task receipt changed')
        result_path=source/receipt['result_path']
        if sha(result_path)!=receipt['result_sha256']:raise ValueError('Original completed GT result SHA changed')
        result=read(result_path)
        if result['config']!=config or int(result['global_index'])!=index or result['status']!='COMPLETE':raise ValueError('Original result identity changed')
        arrays=source/result['arrays']['path']
        if sha(arrays)!=result['arrays']['sha256']:raise ValueError('Original actual candidate arrays changed')
        identity=_source_identity(source,receipt)
        return semantic_key(index,config),{'receipt':ref(path),'result':ref(result_path),'arrays':{'path':str(arrays.resolve()),'sha256':result['arrays']['sha256']},
                                         'index':index,'token':result['token'],'config':config,
                                         'local_input_sha256':result['local_input_sha256'],
                                         'reference_noise':identity['reference_noise']}
    result={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key,item in pool.map(inspect,files):
            # Same semantic configuration has identical deterministic inputs.
            # Keep the first persisted source; conflicting payloads fail closed.
            if key in result and result[key]['arrays']['sha256']!=item['arrays']['sha256']:
                raise ValueError('Two completed sources disagree for an identical semantic GT configuration')
            result.setdefault(key,item)
    _CATALOG[stage]=result;return result

def _validate_new_inputs(root,index,token,source):
    protocol=read(root/'PROTOCOL.json');base_root=Path(protocol['native_base_root'])
    complete=read(base_root/'train/cases'/token/'COMPLETE.json')
    if complete['arrays_sha256']!=source['local_input_sha256'] or int(complete['global_index'])!=index or complete['token']!=token:
        raise ValueError('Reuse requires the identical locally executed Native Base input bytes')
    old_noise=source['reference_noise'];new_noise=protocol['reference_noise']
    if any(old_noise[key]!=new_noise[key] for key in ('protocol_sha256','bank_sha256','token_index_sha256')):
        raise ValueError('Reuse cannot cross explicit noise bank/protocol/index identities')
    return complete

def reuse_completed(root,stage):
    root=Path(root);folder=root/'runs'/stage
    if stage=='export_post':return {'reused':0,'fresh_final_exports_required':True}
    marker=folder/'REUSE_IMPORT_COMPLETE.json'
    if marker.exists():return read(marker)
    protocol=read(root/'PROTOCOL.json');indices=set(protocol['reuse_old_intersection_indices'])
    catalog=stage_catalog(stage,indices);rows={int(r['global_index']):r for r in read(root/'manifests/SCENES.json')}
    pending=[]
    for scene in read(folder/'TASKS.json'):
        index=int(scene['index'])
        if index not in indices:continue
        for position,config in enumerate(scene['configs']):
            source=catalog.get(semantic_key(index,config))
            if source:pending.append((scene,position,config,source))
    def persist(job):
        scene,position,config,source=job;index=int(scene['index']);token=rows[index]['token']
        _validate_new_inputs(root,index,token,source)
        old_result=read(source['result']['path']);result=copy.deepcopy(old_result)
        result.update(config=config,arrays=source['arrays'],
           completed_configuration_reuse={'source_result':source['result'],'source_receipt':source['receipt'],
               'semantic_config_sha256':digest(semantic_config(config)),'local_candidates':protocol['local_candidates'],
               'same_Base_input_bytes':True,'same_explicit_noise_bank':True,'no_new_GPU_execution_claimed':True})
        result_path=root/'analysis'/stage/scene['id']/config['id']/'RESULT.json'
        atomic(result_path,result)
        task_id=f"{scene['id']}__c{position:04d}";task=read(folder/'configuration_tasks'/(task_id+'.json'))
        original_receipt=read(source['receipt']['path'])
        receipt={'status':'COMPLETE','time':stamp(),'task_id':task_id,'task_sha256':digest(task),'config_sha256':digest(config),
                 'result_path':str(result_path.relative_to(root)),'result_sha256':sha(result_path),
                 'worker':'verified_original_completed_reuse','seconds':0.,
                 'numerics_binding_sha256':original_receipt['numerics_binding_sha256'],
                 'selected_on':'actual S0 PDMS; transfer S1/S2 never selection',
                 'reuse_source_receipt':source['receipt'],'reused_original_GPU_seconds':original_receipt['seconds']}
        atomic(folder/'configuration_receipts'/(task_id+'.json'),receipt)
        return {'task_id':task_id,'source_result':source['result'],'result':ref(result_path)}
    with ThreadPoolExecutor(max_workers=8) as pool:evidence=list(pool.map(persist,pending))
    atomic(folder/'REUSED_CONFIGURATIONS.json',{'stage':stage,'reused':len(evidence),'sources':evidence})
    value={'status':'COMPLETE_VERIFIED_SEMANTIC_REUSE','stage':stage,'reused':len(evidence),
           'source_catalog_entries':len(catalog),'evidence':ref(folder/'REUSED_CONFIGURATIONS.json'),
           'all_remaining_configuration_tasks_compute_fresh':True,'fresh_final_exports_required':True}
    atomic(marker,value)
    print(json.dumps({'event':'REUSE_STAGE_COMPLETE',**value}),flush=True)
    return value
