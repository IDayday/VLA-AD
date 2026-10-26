"""Actual GPU parity proof for C-only RAM frozen-teacher prefix memoization.

Uses real Native post/state inputs and a bounded shared synthetic shift. No old
expert, oracle label, optimizer update, or formal teacher basis is used here.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import os
import random
import sys
import time
import numpy as np
import torch


def _exact(a, b, label):
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.detach().contiguous().view(torch.int32), b.detach().contiguous().view(torch.int32)):
        raise AssertionError(label + ' is not bytewise equal')


def _rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state().clone(),
            'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]}


def _equal(a, b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def _head_identity(head):
    h=hashlib.sha256(); versions={}
    for kind, rows in [('parameter',head.named_parameters()),('buffer',head.named_buffers())]:
        for name,tensor in rows:
            if kind=='parameter' and (tensor.requires_grad or tensor.grad is not None or tensor.dtype!=torch.float32):
                raise AssertionError('Head parameters are not frozen FP32/no-grad')
            h.update((kind+'|'+name+'|'+str(tensor.dtype)+'|'+str(tuple(tensor.shape))).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            versions[kind+'|'+name]=(id(tensor),tensor._version)
    return {'sha256':h.hexdigest(),'versions':versions,'training':head.training,
            'module_training':{n:m.training for n,m in head.named_modules()}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root',default='/mnt/project/recogdrive-no-routing-round1-20261026')
    p.add_argument('--paths',required=True)
    p.add_argument('--feature-manifest',required=True)
    p.add_argument('--inference-graph-receipt',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    repo=Path(args.code_root).resolve();sys.path.insert(0,str(repo))
    from student_distillation.asset_identity import atomic_json,read_json,sha256_file
    from student_distillation.data import SceneDataset,batch_scene
    from student_distillation.noise import make_noise_tape,train_noise_seed
    from student_distillation.train import bridge_factory
    from student_distillation.inference_graph import qualified_inference_bridge,InferenceGraphBridge
    from student_distillation.teacher_query_cache import (cached_teacher_query,clear_teacher_query_cache,
        teacher_query_cache_statistics,teacher_query_cache_debug_last,validate_cache_receipt,
        native_runtime_identity,cache_runtime_identity,SCHEMA,QUALIFIED)
    import student_distillation.teacher_query_cache as cache_module
    from student_distillation.native_bridge import NativeDDIMBridge
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or not args.device.startswith('cuda'):
        raise RuntimeError('Actual GPU qualification is required; CPU tests cannot admit this cache')
    torch.cuda.set_device(torch.device(args.device));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.are_deterministic_algorithms_enabled():raise RuntimeError('Native no-grad teacher queries must retain the original forward flag')
    paths=read_json(args.paths)
    # Construct Native once, then apply exactly the production qualified wrapper.
    paths.pop('inference_graph_receipt',None)
    paths.update(feature_manifest=str(Path(args.feature_manifest).resolve()),results_root=str(out/'runtime'))
    native=bridge_factory(paths,args.device)
    bridge=qualified_inference_bridge(native,args.inference_graph_receipt,feature_manifest=args.feature_manifest)
    if not isinstance(bridge,InferenceGraphBridge):raise AssertionError('Production inference-graph bridge was not qualified')
    dataset=SceneDataset(args.feature_manifest,'train',None)
    selected=sorted(range(len(dataset)),key=lambda i:hashlib.sha256(('NRKD_teacher_cache_qualification_v1|'+dataset.metadata[i]['token']).encode()).hexdigest())[:64]
    batches=[batch_scene(dataset[i],args.device) for i in selected]
    raw=np.sin(np.arange(384,dtype=np.float64)*.017)
    delta=(raw/np.sqrt(np.square(raw).mean())*.01).astype(np.float32)
    shift=torch.from_numpy(delta).to(args.device)
    with torch.no_grad():
        teachers=[torch.where((b['image_mask']|b['text_mask']).unsqueeze(-1),b['base_post']+shift[None,None],b['base_post']).detach() for b in batches]
        contexts=[bridge.encode_context(b['history12'],b['status8']) for b in batches]
        # Prime existing native JIT/graph machinery before measuring the cache's
        # own RNG behavior. Cache allocation/lookup remains cold in every proof.
        bridge.states_from_post(teachers[0],contexts[0],make_noise_tape(train_noise_seed(batches[0]['token'],20261027,0),args.device))
    header_sources={}
    source_paths=[Path(cache_module.__file__).resolve().with_name('native_bridge.py'),
                  Path(cache_module.__file__).resolve().with_name('inference_graph.py'),
                  Path(inspect.getfile(type(bridge.head))).resolve()]
    for path in source_paths:
        if not path.is_relative_to(repo):raise ValueError('Native source is outside the recorded implementation root: '+str(path))
        header_sources[str(path.relative_to(repo))]=sha256_file(path)
    feature_sha=sha256_file(args.feature_manifest)
    cache_sha=sha256_file(cache_module.__file__);native_runtime=native_runtime_identity(bridge)
    head_before=_head_identity(bridge.head);global_before=_rng();began=time.monotonic()
    cases=[];initial_above_one=False;all_both=[]
    with torch.no_grad():
        for number,(batch,h_t,context) in enumerate(zip(batches,teachers,contexts)):
            token=batch['token'];case={'token':token,'native_length':h_t.shape[1],'conditions':[]}
            for training_seed in (20261027,20261028):
                for bank in range(5):
                    tape=make_noise_tape(train_noise_seed(token,training_seed,bank),args.device)
                    initial_above_one |= bool((tape[0].abs()>1).any())
                    baseline=bridge.states_from_post(h_t,context,tape)
                    if not bool((baseline['stds'][-1]>=1e-4).all()):raise AssertionError('Native last-step floor missing')
                    namespace={'feature_manifest_sha256':feature_sha,
                        'targets_manifest_sha256':hashlib.sha256(delta.tobytes()+b'engineering synthetic target').hexdigest(),
                        'teacher_bundle_sha256':hashlib.sha256(b'NO_FORMAL_OR_OLD_TEACHER_CACHE_TRANSPORT_PROBE').hexdigest(),
                        'teacher_target':'shared_synthetic_shift_transport_only',
                        'actual_Base_interface_sha256':batch['metadata']['actual_Base_interface_sha256']}
                    def query(j,t=tape,n=namespace,b=bank,max_entries=4):
                        return cached_teacher_query(bridge,h_t,context,t,j,token=token,training_seed=training_seed,
                                                    noise_bank=b,identity=n,max_entries=max_entries,engineering=True)
                    condition={'training_seed':training_seed,'noise_bank':bank,'patterns':{}}
                    for name,order in [('cold',[0,1,2,3,4]),('reverse',[4,3,2,1,0])]:
                        clear_teacher_query_cache(bridge)
                        before=teacher_query_cache_statistics(bridge).get('native_mean_queries',0)
                        rng_before=_rng();hits=[]
                        for j in order:
                            state,mean,hit=query(j)
                            _exact(state,baseline['states'][j],name+' state');_exact(mean,baseline['means'][j],name+' mean');hits.append(hit)
                        if not _equal(rng_before,_rng()):raise AssertionError('Cold/reverse cache changed RNG')
                        snapshot=teacher_query_cache_debug_last(bridge,engineering=True)
                        _exact(snapshot['states'][5].to(args.device),baseline['before_final_clip'],name+' last stochastic transition')
                        count=teacher_query_cache_statistics(bridge).get('native_mean_queries',0)-before
                        if count!=5:raise AssertionError('Prefix cache ran extra/missing teacher transitions')
                        condition['patterns'][name]={'all5_states_means_bytewise':True,'native_queries':count,'hits':hits,'last_transition_bytewise':True}
                    rng_before=_rng();before=teacher_query_cache_statistics(bridge).get('native_mean_queries',0)
                    for j in range(5):
                        state,mean,hit=query(j)
                        if not hit:raise AssertionError('Warm teacher prefix did not hit')
                        _exact(state,baseline['states'][j],'warm state');_exact(mean,baseline['means'][j],'warm mean')
                        # Caller mutation must never poison the CPU entry.
                        state.zero_();mean.zero_();again_state,again_mean,again_hit=query(j)
                        _exact(again_state,baseline['states'][j],'independent returned state storage')
                        _exact(again_mean,baseline['means'][j],'independent returned mean storage')
                    if teacher_query_cache_statistics(bridge).get('native_mean_queries',0)!=before or not _equal(rng_before,_rng()):raise AssertionError('Warm cache queried teacher or changed RNG')
                    condition['patterns']['warm']={'all5_states_means_bytewise':True,'native_queries':0,'independent_return_storage':True}
                    clear_teacher_query_cache(bridge);query(4,max_entries=2)
                    for other in ((bank+1)%5,(bank+2)%5):
                        other_tape=make_noise_tape(train_noise_seed(token,training_seed,other),args.device)
                        query(0,t=other_tape,b=other,max_entries=2)
                    rng_before=_rng()
                    for j in range(5):
                        state,mean,hit=query(j,max_entries=2)
                        if j==0 and hit:raise AssertionError('Evicted entry unexpectedly hit')
                        _exact(state,baseline['states'][j],'eviction state');_exact(mean,baseline['means'][j],'eviction mean')
                    if not _equal(rng_before,_rng()):raise AssertionError('Eviction refill changed RNG')
                    if teacher_query_cache_statistics(bridge)['entries']>2:raise AssertionError('LRU capacity exceeded')
                    condition['patterns']['eviction']={'all5_states_means_bytewise':True,'capacity':2}
                    case['conditions'].append(condition)
                    all_both.append((token,training_seed,bank))
            cases.append(case)
            atomic_json(out/'LIVE.json',{'status':'QUALIFYING_CACHE_PURE_TRANSPORT','real_scenes_completed':number+1,'total_scenes':64})
            if (number+1)%8==0:print(json.dumps({'scenes':number+1,'total':64,'seconds':time.monotonic()-began}),flush=True)
    if not initial_above_one:raise AssertionError('Unclipped Gaussian initial state was not exercised')
    if not _equal(global_before,_rng()):raise AssertionError('Whole transport proof changed global RNG')
    if not _equal(head_before,_head_identity(bridge.head)):raise AssertionError('Head parameters/buffers/versions/eval metadata changed')
    if sha256_file(cache_module.__file__)!=cache_sha:raise RuntimeError('Cache source changed during qualification')
    proof={'real_scenes':64,'time_indices':[0,1,2,3,4],'training_seeds':[20261027,20261028],
           'noise_banks':[0,1,2,3,4],'conditions':len(all_both),'cold_bytewise':True,'warm_bytewise':True,
           'reverse_order_bytewise':True,'eviction_bytewise':True,'last_step_floor_and_unclipped_initial_exact':True,
           'rng_unchanged':True,'head_unchanged':True,'returned_values_do_not_alias_cache':True,
           'no_gradient_or_optimizer_updates':True,'native_head_primed_before_rng_measurement':True}
    receipt={'schema':SCHEMA,'status':QUALIFIED,'policy':'C_ONLY_FROZEN_TEACHER_PREFIX',
             'cache_module_sha256':cache_sha,'cache_runtime_sha256':cache_runtime_identity(),
             'native_source_sha256':header_sources,'native_runtime_sha256':native_runtime,
             'feature_manifest_sha256':feature_sha,
             'checkpoint_sha256':sha256_file(paths['checkpoint']),
             'inference_graph_receipt_sha256':sha256_file(args.inference_graph_receipt),
             'bridge_class':type(bridge).__name__,'proof':proof,'cases':cases,
             'synthetic_shift':{'sha256':hashlib.sha256(delta.tobytes()).hexdigest(),'RMS':float(np.sqrt(np.square(delta.astype(np.float64)).mean())),
                                'old_expert_used':False,'formal_teacher_or_oracle_label_used':False},
             'GPU':torch.cuda.get_device_name(args.device),'hostname':__import__('socket').gethostname(),
             'elapsed_seconds':time.monotonic()-began,'cache_statistics':teacher_query_cache_statistics(bridge)}
    validate_cache_receipt(receipt)
    # Exercise the actual qualified admission branch too, with exact same native
    # methods and a RAM-preloaded receipt; no old teacher data enters this proof.
    with torch.no_grad():
        clear_teacher_query_cache(bridge)
        namespace['feature_manifest_sha256']=receipt['feature_manifest_sha256']
        state,mean,_=cached_teacher_query(bridge,h_t,context,tape,4,token=token,training_seed=training_seed,
                         noise_bank=bank,identity=namespace,max_entries=2,receipt=receipt)
        _exact(state,baseline['states'][4],'qualified-admission state');_exact(mean,baseline['means'][4],'qualified-admission mean')
    atomic_json(out/'QUALIFICATION.json',receipt)
    print(json.dumps({'status':receipt['status'],'receipt':str(out/'QUALIFICATION.json'),
                      'sha256':sha256_file(out/'QUALIFICATION.json'),'seconds':receipt['elapsed_seconds']}),flush=True)
    return receipt


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        target=sys.argv[sys.argv.index('--out')+1] if '--out' in sys.argv else None
        if target:
            # File I/O is the independent qualification CLI's responsibility;
            # cached_teacher_query itself never opens or writes a file.
            Path(target).mkdir(parents=True,exist_ok=True)
            (Path(target)/'FAILURE.json').write_text(json.dumps({'status':'CACHE_QUALIFICATION_FAILED','traceback':traceback.format_exc()},indent=2))
        raise
