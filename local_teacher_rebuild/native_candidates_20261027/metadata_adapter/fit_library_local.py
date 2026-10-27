#!/usr/bin/env python3
"""Local-S0 metadata/transport adapter around unchanged original D/C_to_P/K8 kernels.

No historical fitter is edited. Numerical functions retain their exact code
objects inside one private module namespace; only metadata and pair transport
are replaced. The student Train85813 population and training budget are untouched.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, sys, types
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
OLD = Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026')
DEFAULT_ROOT = Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027')
METADATA = Path(__file__).resolve().parent
CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
ORIGINAL = OLD / 'code/fit_library.py'
CHECKPOINT_SHA = '23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def reference(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def checked(ref):
    p = Path(ref['path'])
    if sha(p) != ref['sha256']: raise ValueError('Local metadata/artifact SHA mismatch: ' + str(p))
    return p


def load_metadata(folder=METADATA):
    folder = Path(folder)
    c = read(folder / 'LOCAL_CANDIDATE_RECHECK.json')
    candidates_path = checked(c['candidate_list']); candidates = read(candidates_path)
    p = read(folder / 'LOCAL_PANEL_RECHECK.json')
    ordinary_path = checked(p['artifacts']['LOCAL_ORDINARY_PANEL.json'])
    split_path = checked(p['artifacts']['LOCAL_FINAL_SPLIT.json'])
    pilot_path = checked(p['artifacts']['LOCAL_PILOT32.json'])
    ordinary, split, pilot = read(ordinary_path), read(split_path), read(pilot_path)
    rows = ordinary['selected']; by_index = {r['index']: r for r in rows}
    failures = {r['global_index'] for r in candidates['candidates']}
    if (len(rows) != ordinary['count'] or len(by_index) != len(rows) or
            len(failures) != c['local_GT_Train_count'] or
            len(split['fit']['selected_indices']) + len(split['cal']['selected_indices']) != len(rows)):
        raise ValueError('Verified local derived metadata counts changed')
    if failures != {r['index'] for r in rows if r['audit_category'] == 'failure'}:
        raise ValueError('GT candidates and local ordinary failure stratum differ')
    if len({r['token'] for r in rows}) != len(rows) or any(r.get('population', 'train') != 'train' for r in candidates['candidates']):
        raise ValueError('Duplicate or non-Train local candidate metadata')
    fit, cal = split['fit'], split['cal']
    original = read(checked(split['source_original_session_split']))
    for key, old_key in [('fit','final_locked_family_inner_fit'), ('cal','final_locked_family_inner_calibration')]:
        if split[key]['sessions'] != original[old_key]['sessions']:
            raise ValueError('Original Train session membership was changed')
    if set(fit['sessions']) & set(cal['sessions']) or set(fit['selected_indices']) & set(cal['selected_indices']):
        raise ValueError('Local fit/cal overlap')
    if set(fit['selected_indices']) | set(cal['selected_indices']) != set(by_index):
        raise ValueError('Local fit/cal index coverage mismatch')
    for group in [fit, cal]:
        if any(by_index[i]['session'] not in group['sessions'] for i in group['selected_indices']):
            raise ValueError('A local row crossed the frozen session boundary')
        mass = sum(group['conditional_natural_mean_weight_per_sample'][by_index[i]['audit_category']] for i in group['selected_indices'])
        if abs(mass - 1.) > 1e-12: raise ValueError('Local natural stratum weights do not sum to one')
    if len(pilot['panel']) != 32 or any(r['index'] not in cal['selected_indices'] for r in pilot['panel']):
        raise ValueError('Pilot panel is not the 32-scene local calibration panel')
    return {'candidates': candidates, 'rows': rows, 'fit': fit, 'cal': cal, 'panel': pilot['panel'],
            'refs': {'candidate_list': reference(candidates_path), 'ordinary_panel': reference(ordinary_path),
                     'final_split': reference(split_path), 'pilot32': reference(pilot_path)},
            'base_pool': c['local_Base_pool']}


def metadata_labels(value, path, root):
    """Change obsolete count labels only, never numerical arrays or decisions."""
    value = copy.deepcopy(value)
    plan_path = Path(root) / 'fitting/PLAN.json'
    plan = read(plan_path) if plan_path.exists() else None
    if not isinstance(value, dict): return value
    if path.name == 'BANK.json' and path.parent.name == 'INNER' and plan:
        value['scope'] = f"local{len(plan['fit']['selected_indices'])}/{len(plan['fit']['sessions'])}sessions"
    if path.name == 'FROZEN_INNER_SELECTION.json' and plan:
        value.pop('original2021_calibration_indices', None)
        value.update(local_calibration_indices_count=len(plan['cal']['selected_indices']), local_natural_weights=True,
                     original_Train_session_memberships_preserved=True)
    if path.name == 'BANK.json' and path.parent.name == 'ALLTRAIN' and plan:
        value.pop('original10042_selectedTrain_all130sessions', None)
        value.update(local_selected_Train_count=len(plan['rows']), Train_sessions=len(set(plan['fit']['sessions'] + plan['cal']['sessions'])))
    if path.name == 'library.json': value['id'] = 'local_refit_D_C8_double_native_candidates_20261027'
    if path.name == 'BUNDLE.json' and plan:
        value.update(local_candidate_metadata=plan['local_candidate_metadata'], ordinary_Train_count=len(plan['rows']),
                     GT_candidate_count=plan['GT_candidate_count'], full_GT_search_required=True,
                     shortened_cut_or_FAST_formal_data_allowed=False, metadata_adapter=reference(__file__))
    return value


def cloned_fitter(root):
    """Private globals preserve exact function code and multiprocessing identities."""
    sys.path[:0] = [str(OLD / 'code'), str(CODE)]
    name = 'nrkd_original_local_fit_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ORIGINAL)
        source = importlib.util.module_from_spec(spec); sys.modules[name] = source; spec.loader.exec_module(source)
    source = sys.modules[name]
    private_name = 'nrkd_local_candidate_fit_' + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
    module = types.ModuleType(private_name); module.__dict__.update(source.__dict__)
    module.__name__ = private_name
    module.ROOT = Path(root).resolve(); module.DEFAULT_SOURCE = OLD / 'source_snapshot/ReCogDrive'
    module._FIT_CONTEXT = None
    for key, value in source.__dict__.items():
        if isinstance(value, types.FunctionType) and value.__module__ == name:
            cloned = types.FunctionType(value.__code__, module.__dict__, value.__name__, value.__defaults__, value.__closure__)
            cloned.__kwdefaults__ = copy.copy(value.__kwdefaults__); cloned.__annotations__ = copy.copy(value.__annotations__)
            cloned.__module__ = private_name; cloned.__qualname__ = value.__qualname__
            module.__dict__[key] = cloned
            assert cloned.__code__ is value.__code__
    sys.modules[private_name] = module
    original_atomic = module.atomic
    def private_atomic(path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(module.ROOT): raise ValueError('Local fitter write outside its new root')
        original_atomic(path, metadata_labels(value, path, module.ROOT))
    module.atomic = private_atomic
    module.prepare = lambda args: prepare_local(args, module)
    module.load_pairs = lambda plan: load_pairs_local(plan, module)
    original_status = module.status
    def local_status(args):
        value = original_status(args)
        plan = read(args.root / 'fitting/PLAN.json') if (args.root / 'fitting/PLAN.json').exists() else None
        value['GT_exports_required'] = plan['GT_candidate_count'] if plan else len(load_metadata()['candidates']['candidates'])
        return value
    module.status = local_status
    module._original_source = source
    return module


def prepare_local(args, module):
    from student_distillation.data import FeatureStore
    from student_distillation.asset_identity import population_rows, assert_population_isolation
    meta = load_metadata(args.metadata_root); rows = copy.deepcopy(meta['rows'])
    store = FeatureStore(args.features, 'train')
    if store.manifest.get('source_checkpoint_sha256') != CHECKPOINT_SHA: raise ValueError('Wrong frozen Base checkpoint')
    isolation = assert_population_isolation(args.frozen_repo)
    train = {r['token']: r for r in population_rows('train', args.frozen_repo)}
    if len(train) != 85813 or set(train) != set(store.by_token): raise ValueError('Full Train feature coverage changed')
    if sha(args.base_pool) != meta['base_pool']['sha256']: raise ValueError('Full local Base pool differs from screening')
    with np.load(args.base_pool, allow_pickle=False) as pool:
        positions = {int(i): j for j, i in enumerate(pool['indices_int64'])}
        take = [positions[r['index']] for r in rows]
        tokens = pool['tokens'].astype(str); sessions = pool['sessions'].astype(str); quality = np.array(pool['Base_quality_float64'])
        if quality.shape != (85813,5,7) or quality.dtype != np.float64 or not np.isfinite(quality).all(): raise ValueError('Local five-score pool malformed')
        actual_failures = set(pool['indices_int64'][quality[:,0,0] < .7].tolist())
        if actual_failures != {r['global_index'] for r in meta['candidates']['candidates']}: raise ValueError('Local S0 GT screening mismatch')
        if tokens[take].tolist() != [r['token'] for r in rows] or sessions[take].tolist() != [r['session'] for r in rows]: raise ValueError('Local ordinary index/token/session join mismatch')
        for r,j in zip(rows,take):
            source = train[r['token']]
            if int(source['index']) != r['index'] or source['session'] != r['session']: raise ValueError('Local panel crossed Train identity')
            category = 'failure' if quality[j,0,0] < .7 else ('mid' if quality[j,0,0] <= .9 else 'high')
            if r['audit_category'] != category or r['Base_scores'] != quality[j,:,0].tolist(): raise ValueError('Historical category/scores leaked into local panel')
            r.update(category=category, population='train', source_subset='exploration')
        # Reapply the original reference priority against the entire localTrain.
        def priority(group,t): return hashlib.sha256(('OL_SESSION_CV|2026102531|reference8192|' + group + '|' + str(t)).encode()).hexdigest()
        expected = set(actual_failures)
        for group,condition,count in [('mid',(quality[:,0,0]>=.7)&(quality[:,0,0]<=.9),2048),('high',quality[:,0,0]>.9,6144)]:
            order = sorted(np.flatnonzero(condition),key=lambda j:priority(group,tokens[j]))[:count]
            expected.update(int(pool['indices_int64'][j]) for j in order)
        if expected != {r['index'] for r in rows}: raise ValueError('Local control sample does not match original hash rule')
        arrays = {'indices_int64':np.asarray([r['index'] for r in rows],np.int64),'tokens':np.asarray([r['token'] for r in rows]),
                  'Base_pooled_float64':np.array(pool['Base_pooled_float64'][take]),'token_counts_int64':np.array(pool['token_counts_int64'][take]),
                  'Base_quality_float64':np.array(quality[take])}
    states=[]; base_sha=[]
    for r in rows:
        entry = store.rows[store.by_token[r['token']]]
        states.append(np.array(store._array(int(entry['shard']),'state24')[int(entry['state_row'])]))
        base_sha.append(entry['actual_Base_interface_sha256'])
    arrays['raw_aux24_float32']=np.stack(states); arrays['Base_interface_sha256']=np.asarray(base_sha)
    if arrays['raw_aux24_float32'].dtype != np.float32 or arrays['raw_aux24_float32'].shape != (len(rows),24) or not np.isfinite(arrays['raw_aux24_float32']).all(): raise ValueError('Ordinary state24 malformed')
    if not np.array_equal(arrays['raw_aux24_float32'][:,12:16],arrays['raw_aux24_float32'][:,20:24]): raise ValueError('Ordinary command order changed')
    ops,geo,math,binding = module.bind_math(args.source_root)
    panel = math['pilot_panel'](0,meta['cal'],{r['index']:r for r in rows})
    if panel != meta['panel']: raise ValueError('Pilot32 differs from the original local-category/session roundrobin')
    for group in [meta['fit'],meta['cal']]:
        selected=[r for r in rows if r['index'] in set(group['selected_indices'])]
        for category in ['failure','mid','high']:
            selected_count=sum(r['audit_category']==category for r in selected)
            natural_count=sum((quality[:,0,0]<.7 if category=='failure' else ((quality[:,0,0]>=.7)&(quality[:,0,0]<=.9)) if category=='mid' else quality[:,0,0]>.9)&np.isin(sessions,group['sessions']))
            if selected_count != group['selected_counts'][category] or int(natural_count) != group['full_population_counts'][category]: raise ValueError('Local natural stratum count mismatch')
            if group['conditional_natural_mean_weight_per_sample'][category] != natural_count/(group['full_population_scenes']*selected_count): raise ValueError('Local natural weight mismatch')
    folder=args.root/'fitting';folder.mkdir(parents=True,exist_ok=True)
    feature_snapshot=folder/('FEATURES_'+sha(args.features)+'.json')
    if not feature_snapshot.exists(): feature_snapshot.write_bytes(Path(args.features).read_bytes())
    if sha(feature_snapshot) != sha(args.features): raise ValueError('Feature snapshot relocation changed bytes')
    ordinary_path=folder/'ORDINARY_LOCAL.npz'
    if ordinary_path.exists():
        with np.load(ordinary_path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError('Existing local ordinary arrays have another identity')
        ordinary=reference(ordinary_path)
    else:
        ordinary=module.save_npz(ordinary_path,arrays)
    _,old=module.source_paths(args.source_root);doses=read(old/'preparation/l1_pilot_metadata_v1/MANIFEST.json')['doses']
    plan={'schema':'local_dc8_refit_plan_v2_local_candidates','status':'PREPARED_LOCAL_ORDINARY_METADATA',
          'root':str(args.root),'source_root':str(args.source_root),'source_code_root':str(module.source_paths(args.source_root)[0]),
          'original_math_sources':module.source_bindings(args.source_root),'math_bindings':binding,
          'local_candidate_metadata':meta['refs'],'original_split':read(args.metadata_root/'LOCAL_FINAL_SPLIT.json')['source_original_session_split'],
          'local_Base_pool':reference(args.base_pool),'feature_manifest':reference(feature_snapshot),'ordinary':ordinary,
          'rows':rows,'fit':meta['fit'],'cal':meta['cal'],'panel':panel,'doses':doses,'menu':list(ops.MENU),
          'family':{'route':'D','order':'C_to_P','K':8,'rank':48,'view':'valid_shared384'},'whole_variant':{'raw_shift_factor':2.,'cap_factor':2.},
          'gt_root':str(args.gt_root),'native_base_root':str(args.native_base_root),'GT_candidate_count':len(actual_failures),
          'ordinary_count':len(rows),'fit_count':len(meta['fit']['selected_indices']),'cal_count':len(meta['cal']['selected_indices']),
          'source_checkpoint_sha256':CHECKPOINT_SHA,'isolation':isolation,'full_GT_search_required':True,'shortened_cut_or_FAST_formal_data_allowed':False,
          'new_qualified_pair_count':None,'historical_teacher_parameters_choices_or_subspace_used':False,
          'metadata_adapter':reference(__file__),'original_fitter':reference(ORIGINAL),'numeric_function_code_objects_unchanged':True}
    destination=folder/'PLAN.json'
    if destination.exists() and read(destination)!=plan: raise ValueError('Refusing to change an existing local fitting plan')
    module.atomic(destination,plan)
    return {'status':plan['status'],'plan':str(destination),'GT_candidates':len(actual_failures),'ordinary':len(rows),
            'fit':plan['fit_count'],'cal':plan['cal_count'],'pilot32':[r['index']for r in panel],
            'next':'Complete full localGTsearch and all dynamiccandidate Nativepair exports beforefit-inner'}


def load_pairs_local(plan, module):
    rows=[r for r in plan['rows'] if r['audit_category']=='failure'];required=int(plan['GT_candidate_count'])
    if len(rows)!=required or len({r['token']for r in rows})!=required: raise ValueError('Local candidate/export coverage mismatch')
    with np.load(checked(plan['ordinary']),allow_pickle=False)as ordinary:
        index={int(i):j for j,i in enumerate(ordinary['indices_int64'])};expected_pool=np.array(ordinary['Base_pooled_float64']);expected_counts=np.array(ordinary['token_counts_int64']);expected_quality=np.array(ordinary['Base_quality_float64']);expected_sha=ordinary['Base_interface_sha256'].astype(str)
    records=[];pairs={};missing=[]
    for row in rows:
        marker=Path(plan['gt_root'])/'exports/post'/row['token']/'COMPLETE.json'
        if not marker.exists():missing.append(row['token']);continue
        meta=read(marker)
        if meta.get('status')!='PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR' or meta.get('token')!=row['token'] or int(meta.get('global_index',-1))!=row['index'] or meta.get('population')!='train': raise ValueError('GT export metadata mismatch')
        artifact=dict(meta['arrays']);file=Path(artifact['path']);file=file if file.is_absolute()else Path(plan['gt_root'])/file
        if sha(file)!=artifact['sha256']:raise ValueError('GT export arrays SHA mismatch')
        with np.load(file,allow_pickle=False)as saved:
            base=np.asarray(saved['Base_interface']);teacher=np.asarray(saved['teacher_interface']);image=np.asarray(saved['image_mask']);text=np.asarray(saved['text_mask']);valid=image|text;j=index[row['index']]
            if base.dtype!=np.float32 or teacher.dtype!=np.float32 or base.shape!=teacher.shape or base.ndim!=2 or base.shape[1]!=384 or not np.isfinite(base).all()or not np.isfinite(teacher).all():raise ValueError('GTpost must preserve finite fullFP32layout')
            if image.dtype!=np.bool_ or text.dtype!=np.bool_ or image.shape!=base.shape[:1]or text.shape!=image.shape or np.any(image&text)or not image.any()or not text.any():raise ValueError('GT export token roles invalid')
            if base[~valid].tobytes()!=teacher[~valid].tobytes():raise ValueError('GTpost changed native padding bytes')
            if hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest()!=expected_sha[j]:raise ValueError('GTpair Base differs from independent fullFP32mmap SHA')
            bq=np.asarray(saved['Base_quality']) if 'Base_quality'in saved else expected_quality[j]
            quality=np.asarray(saved['quality'])
            if bq.dtype!=np.float64 or quality.dtype!=np.float64 or bq.ndim!=2 or quality.ndim!=2 or bq.shape[1]!=7 or quality.shape[1]!=7 or not np.isfinite(bq).all()or not np.isfinite(quality).all():raise ValueError('GTpair saved scores malformed')
            if not np.array_equal(bq,expected_quality[j,:len(bq)]):raise ValueError('GTpair localBase scores differ from independent five-score pool')
            b0,t0=float(bq[0,0]),float(quality[0,0]);qualified=b0<.7 and t0>.9
            if not b0<.7:raise ValueError('Export is outside the newly screened localS0failure population')
            counts=np.array([image.sum(),text.sum()],np.int64);f=np.stack([base[m].astype(np.float64).mean(0)for m in(image,text)]);t=np.stack([teacher[m].astype(np.float64).mean(0)for m in(image,text)])
            if not np.array_equal(f,expected_pool[j])or not np.array_equal(counts,expected_counts[j]):raise ValueError('GTpair Base differs from independent local pool')
            rms=float(np.sqrt(np.square(teacher[valid].astype(np.float64)-base[valid].astype(np.float64)).mean()))
        records.append({'index':row['index'],'token':row['token'],'session':row['session'],'qualified':qualified,'Base_S0':b0,'teacher_S0':t0,'actual_fulltoken_RMS':rms,'source':reference(marker)});pairs[row['index']]=(f,t,counts,rms)
    if missing:
        module.atomic(Path(plan['root'])/'fitting/PAIR_EXPORTS_PENDING.json',{'status':'WAITING_NEW_GT_POST_EXPORTS','required':required,'present':required-len(missing),'missing_count':len(missing),'tokens':missing})
        raise RuntimeError(f'New local GTpost exports not ready: {len(missing)}/{required} missing')
    if plan.get('full_GT_search_required'):
        search=Path(plan['gt_root'])/'SEARCH_COMPLETE.json';complete=Path(plan['gt_root'])/'GT_POST_COMPLETE.json'
        if not search.exists()or not complete.exists():raise RuntimeError('Complete candidate exports cannot replace mandatory fullGTsearch completion evidence')
        s,c=read(search),read(complete)
        if s.get('status')!='FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE' or s.get('full_original_finite_search_complete')is False or c.get('train_failures')!=required or c.get('actual_exports')!=required:raise ValueError('GTsearch/export completion doesnotprove full dynamiccandidate dataflow')
    module.atomic(Path(plan['root'])/'fitting/LOCAL_PAIR_QUALIFICATION.json',{'status':'COMPLETE','source':'actual localS0','total':required,'qualified':sum(r['qualified']for r in records),'rule':'BaseS0<.7 andselectedGTpostS0>.9','rows':records,'no_historical_count_assumption':True})
    return records,pairs


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','fit-inner','pack-pilot','collect-pilot','pack-cal','collect-cal','refit','freeze','queue-population','collect-population','queue-base-qualification','emit-validation-diagnostics','emit-teacher-report','selftest','status','transport-check'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--source-root',type=Path,default=OLD/'source_snapshot/ReCogDrive');p.add_argument('--metadata-root',type=Path,default=METADATA);p.add_argument('--gt-root',type=Path);p.add_argument('--native-base-root',type=Path);p.add_argument('--frozen-repo',type=Path,default=CODE)
    p.add_argument('--features',type=Path,default=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026/features/MANIFEST.ALL_POPULATIONS.json'));p.add_argument('--base-pool',type=Path,default=Path('/mnt/project/recogdrive-dc8-local-refit-20261026/inputs/LOCAL_TRAIN_BASE_POOL.npz'))
    p.add_argument('--population',choices=['train','validation','test'],default='train');p.add_argument('--action-chunk',type=int,default=8);p.add_argument('--threads',type=int,default=1);p.add_argument('--fit-workers',type=int,choices=range(1,9),default=1);p.add_argument('--targets-manifest',type=Path);p.add_argument('--diagnostics-out',type=Path)
    args=p.parse_args(argv);args.root=args.root.resolve();args.source_root=args.source_root.resolve();args.metadata_root=args.metadata_root.resolve();args.gt_root=(args.gt_root or args.root/'native_gt').resolve();args.native_base_root=(args.native_base_root or args.root/'native_base').resolve()
    module=cloned_fitter(args.root)
    import torch
    torch.set_num_threads(args.threads)
    actions={'prepare':module.prepare,'fit-inner':module.fit_inner,'pack-pilot':lambda a:module.pack(a,'pilot'),'collect-pilot':lambda a:module.collect(a,'pilot'),'pack-cal':lambda a:module.pack(a,'cal'),'collect-cal':lambda a:module.collect(a,'cal'),'refit':module.refit,'freeze':module.freeze,'queue-population':module.queue_population,'collect-population':module.collect_population,'queue-base-qualification':module.queue_base_qualification,'emit-validation-diagnostics':module.emit_validation_diagnostics,'emit-teacher-report':module.emit_teacher_report,'selftest':module.selftest,'status':module.status,'transport-check':lambda a:{'records':len(module.load_pairs(read(a.root/'fitting/PLAN.json'))[0])}}
    value=actions[args.action](args)
    completed_metadata={'fit-inner':'fitting/INNER/BANK.json','collect-cal':'fitting/FROZEN_INNER_SELECTION.json',
                        'refit':'fitting/ALLTRAIN/BANK.json'}
    if args.action in completed_metadata:
        value=read(args.root/completed_metadata[args.action])
    print(json.dumps(value,allow_nan=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else str(v)),flush=True)


if __name__=='__main__':main()
