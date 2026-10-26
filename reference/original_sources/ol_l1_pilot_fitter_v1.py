"""Session-isolated L1 CPU bank fitter and ordinary fixed-panel requests.

No GPU/native/scoring/queue API. Real space/cluster/operator fitting requires
actual complete small-asset ROOT identity plus a source-bound CPU release.
Only selftest uses private synthetic inputs without an execution release.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,copy,hashlib,json,os,time,traceback,uuid
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
import numpy as np
from b5_common import read,sha,atomic,stamp
import ol_l1_data_pack_v1 as data
import ol_l1_geometry_v1 as geo
import ol_l1_operators_v1 as ops
import ol_l1_prepare_v1 as storage
ref,meta,checked=data.ref,data.meta,data.checked
ROOT=OL/'preparation/l1_pilot_fitter_v1'
PILOT=OL/'preparation/l1_pilot_metadata_v1/MANIFEST.json'
DATA_STATUS='PASS_ROOT_OFFLINE_LIBRARY_L1_COMPLETE_SMALL_ASSET_IDENTITY_V1'
RELEASE_STATUS='PASS_ROOT_OFFLINE_LIBRARY_L1_PILOT_CPU_FITTER_EXECUTION_RELEASE_V1'
KNOWN_ROOT_STATUS='PASS_ROOT_OFFLINE_LIBRARY_L1_MLP_KNOWN2000_ACCEPTANCE_V1'
PLAN_STATUS='READY_SOURCE_BOUND_L1_COMPLETE_MENU_CPU_FIT_PLAN_NO_EXECUTION'


def sources():
    x=data.sources();x.update(storage.sources());x.update({str(Path(p).resolve()):sha(p) for p in [__file__,ops.paired.__file__]});return x

def equal(a,b):
    if a.__class__.__module__.startswith('torch') and a.__class__.__name__=='Tensor':
        import torch
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def saved(folder,payload,binding):
    path=Path(folder)/'MANIFEST.json'
    if path.exists():assert equal(storage.load_artifact(ref(path),binding),payload);return ref(path)
    return storage.save_artifact(folder,payload,binding)

def loaded(record):
    m=meta(record);return storage.load_artifact(record,m['binding'])

def atomic_result(path,value):atomic(path,value,exclusive=True);return ref(path)

def asset_data(proof):
    a=read(proof);assert a['passed'] and a['status']==DATA_STATUS
    assert a['inputs']==ref(data.ROOT/'INPUTS10042.json') and a['full1850']==ref(OL/'preparation/l0_v1/FULL1850_IDENTITY_ROOT_ACCEPTANCE.json')
    pilot=read(PILOT);assert a['session_split']==pilot['source_session_split']
    ordinary=data.small_arrays(a['ordinary_arrays']);pairs=data.small_arrays(a['pair_arrays']);inputs=meta(a['inputs']);rows={r['index']:r for r in inputs['cases']}
    oi=ordinary['indices_int64'];pi=pairs['indices_int64'];assert oi.dtype==pi.dtype==np.int64 and oi.tolist()==inputs['indices'] and len(oi)==10042 and len(pi)==1850 and len(set(pi))==1850
    pos={int(i):n for n,i in enumerate(oi)};pp={int(i):n for n,i in enumerate(pi)};where=np.array([pos[int(i)] for i in pi])
    for k,shape,dtype in [('Base_pooled_float64',(10042,2,384),np.float64),('token_counts_int64',(10042,2),np.int64),('raw_aux24_float32',(10042,24),np.float32)]:
        assert ordinary[k].shape==shape and ordinary[k].dtype==dtype and np.isfinite(ordinary[k]).all()
    assert (ordinary['token_counts_int64']>0).all()
    for k in ('Base_pooled_float64','token_counts_int64','raw_aux24_float32'):assert pairs[k].tobytes()==ordinary[k][where].tobytes(),k
    assert pairs['teacher_pooled_float64'].shape==(1850,2,384) and pairs['teacher_pooled_float64'].dtype==np.float64 and np.isfinite(pairs['teacher_pooled_float64']).all()
    assert pairs['actual_fulltoken_teacher_RMS_float64'].shape==(1850,) and pairs['actual_fulltoken_teacher_RMS_float64'].dtype==np.float64
    assert pairs['canonical_qualified_pair_bool'].dtype==np.bool_ and pairs['canonical_qualified_pair_bool'].shape==(1850,) and pairs['canonical_qualified_pair_bool'].sum()==1807
    if 'paired_delta_pooled_float64' in pairs:assert np.max(np.abs(pairs['teacher_pooled_float64']-pairs['Base_pooled_float64']-pairs['paired_delta_pooled_float64']))<1e-12
    # Source score arrays, if present, are native 0..1 units, never PDMS points.
    if 'canonical_Base_S0_score_float64' in pairs:
        assert np.all((pairs['canonical_Base_S0_score_float64']>=0)&(pairs['canonical_Base_S0_score_float64']<=1))
        assert np.array_equal(pairs['canonical_qualified_pair_bool'],(pairs['canonical_Base_S0_score_float64']<.7)&(pairs['canonical_teacher_S0_score_float64']>.9))
    assert {int(i) for i in pi}=={i for i,r in rows.items() if r['audit_category']=='failure'}
    sessions=set(pilot['inner_fit_sessions']);eligible=set(pilot['inner_fit_indices'])
    fit=[int(i) for i in oi if rows[int(i)]['session'] in sessions and int(i) in eligible]
    assert set(fit)=={int(i) for i in oi if rows[int(i)]['session'] in sessions},'All selected ordinary fit-session rows must be retained'
    assert not {rows[i]['session'] for i in fit}&{v['session'] for v in pilot['panel']}
    q=[int(i) for i in pi if int(i) in set(fit) and pairs['canonical_qualified_pair_bool'][pp[int(i)]]]
    normal=[i for i in fit if rows[i]['audit_category'] in ('mid','high')];high=[i for i in normal if rows[i]['audit_category']=='high']
    assert q and normal and high and all(rows[i]['session'] in sessions for i in q+normal)
    assert np.isfinite(pairs['actual_fulltoken_teacher_RMS_float64']).all() and np.all(pairs['actual_fulltoken_teacher_RMS_float64'][[pp[i] for i in q]]>0)
    return dict(proof=a,pilot=pilot,ordinary=ordinary,pairs=pairs,rows=rows,pos=pos,pp=pp,fit=fit,Q=q,normal=normal,high=high)

def prepare(proof,output=ROOT/'PLAN.json'):
    d=asset_data(proof);p=d['pilot'];assert p['outer_fold']==0 and p['repair_menu']==list(ops.MENU) and p['logical_expert_output_requests']==838656
    units=[]
    for family in p['candidate_families']:
        assert family['view']=='valid_shared384' and family['rank']==48 and family['K'] in geo.K_MENU
        ident=f"{family['route']}__{family['order']}__K{family['K']}"
        units.append(dict(id=ident,**family,dependencies=[f"{family['route']}__P_to_C__K1"] if family['K']==1 and family['order']=='C_to_P' else [],
            logical_actions=family['K']*13*7,MLP_fixed_endpoint=2000))
    result=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),data_proof=ref(proof),pilot=ref(PILOT),units=units,
        fit_indices=d['fit'],qualified_fit_indices=d['Q'],normal_fit_indices=d['normal'],high_fit_indices=d['high'],fit_sessions=p['inner_fit_sessions'],
        fit_indices_rule='selected10042 intersect registered innerfit indices AND innerfit sessions; no population-index assumption',
        initial_seed_rule='geo.seeded(MLP_INITIAL,fit_identity,space_recipe,K,method) modulo (2**63-1); excludes order/cluster/random-group',
        menu=list(ops.MENU),negative_controls=list(ops.CONTROLS),doses=p['doses'],noises=p['noise_indices'],panel_indices=p['panel_indices'],
        logical_native_expert_outputs=838656,known_control_required_before_actual_MLP=True,MLP_weak_fit_never_PDMS_veto=True,
        output_root=str(ROOT/'ACTUAL'),no_real_fit_has_run_by_prepare=True)
    return atomic_result(output,result)

def context(plan,release,action,unit=None):
    p=read(plan);r=read(release);assert p['status']==PLAN_STATUS and p['producer']==ref(__file__) and p['source_files']==sources()
    assert r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan) and r['source_files']==sources() and action in r['allowed_actions']
    if unit is not None:assert unit in r['allowed_units'] and unit in {v['id'] for v in p['units']}
    if action in ('fit','pack','pack-controls'):
        proof=meta(r['known_root']);assert proof['passed'] and proof['status']==KNOWN_ROOT_STATUS and proof['plan']==ref(plan) and proof['source_files']==sources()
        assert set(proof['controls'])=={'mlp','mlp_replay'}
        for m,cr in proof['controls'].items():
            c=meta(cr);assert c['passed'] and c['method']==m and c['endpoint']==2000 and c['all_roles_absorption_ge90'] and c['replay1999_to2000_exact'] and c['plan']==ref(plan)
    return p,r,Path(p['output_root'])

def ordinary_view(roles,counts,view):
    if view['view']=='valid_shared384':return np.einsum('ng,ngd->nd',counts/counts.sum(1)[:,None],roles)
    return (roles*np.sqrt(np.array([view['f0'],1-view['f0']]))[None,:,None]).reshape(len(roles),768)

def coordinates(v,center,B):
    """One authoritative FP64 shape1 dot for fit rows AND ordinary inference."""
    v=np.asarray(v,np.float64)
    if v.ndim==1:return (v-center)@B
    assert v.ndim==2
    return np.stack([(row-center)@B for row in v])

def space(plan,release,route):
    p,r,root=context(plan,release,'space');assert route in ('R','D');path=root/'SPACES'/route/'RESULT.json'
    if path.exists():v=read(path);assert v['plan']==ref(plan) and v['source_files']==sources();return ref(path)
    d=asset_data(checked(p['data_proof']));take=np.array([d['pp'][i] for i in d['Q']]);pairs=d['pairs'];f,s,view=geo.old_geometry.view_vectors(pairs['Base_pooled_float64'][take],pairs['teacher_pooled_float64'][take],pairs['token_counts_int64'][take],'valid_shared384',np.arange(len(take)))
    rms=pairs['actual_fulltoken_teacher_RMS_float64'][take];fitted=geo.fit_space(f,s,rms,route,'valid_shared384',48);D0=float(np.median(rms));assert D0>0
    fi=dict(data_proof=p['data_proof'],Q=d['Q'],normal=d['normal'],high=d['high'],fit_sessions=p['fit_sessions'],view='valid_shared384',outer_fold=0)
    fit_identity=hashlib.sha256(json.dumps(fi,sort_keys=True).encode()).hexdigest()
    state=dict(space=fitted,view=view,D0=D0,Q_indices=np.array(d['Q'],np.int64),normal_indices=np.array(d['normal'],np.int64),high_indices=np.array(d['high'],np.int64),
        fail_view=f,success_view=s,actual_fulltoken_RMS=rms,fit_identity=fit_identity,recipe=dict(route=route,view='valid_shared384',rank=48),tokens=[d['rows'][i]['token'] for i in d['Q']],sessions=[d['rows'][i]['session'] for i in d['Q']])
    if fitted['status']=='VALID_SPACE':
        state['geometry']=geo.geometry(fitted['basis'],view,pairs['token_counts_int64'][take,0]/pairs['token_counts_int64'][take].sum(1),s-f,rms)
    artifact=saved(path.parent/'STATE',state,dict(plan=ref(plan),route=route,source_files=sources()))
    decoder=saved(path.parent/'DECODER',dict(space=dict(basis=fitted['basis']),view=view,D0=D0),dict(plan=ref(plan),route=route,source_files=sources()))
    return atomic_result(path,dict(status='COMPLETE_L1_FOLD_LOCAL_SPACE_REQUIRES_ROOT',passed=True,plan=ref(plan),source_files=sources(),route=route,
        actual_release=ref(release),artifact=artifact,decoder=decoder,space_status=fitted['status'],D0=D0,Q=len(take),fit_identity=fit_identity,no_old_basis_or_scaler_reused=True))

def initialization(state,K,method):return geo.seeded('MLP_INITIAL',state['fit_identity'],state['recipe'],K,method)%(2**63-1)

def train_metrics(session,dataset):
    import torch
    with torch.no_grad():pred=session.model(session.X).cpu().numpy().astype(np.float64)*session.D0
    error=np.square(pred-dataset['y']).mean(1);w=dataset['weights'];roles=np.asarray(dataset['row_roles'])
    return dict(weighted_MSE=float(w@error),role_MSE={str(k):float(error[roles==k].mean()) for k in dict.fromkeys(dataset['row_roles'])},
        role_weighted_contribution={str(k):float(w[roles==k]@error[roles==k]) for k in dict.fromkeys(dataset['row_roles'])})

def mlp_run(dataset,D0,seed,folder,binding):
    """Exact fixed2000 kernel; endpoint replay never calls forbidden step2001."""
    import torch
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);final=folder/'RESULT.json'
    if final.exists():v=read(final);assert v['binding']==binding and v['source_files']==sources();return v
    before_rng=torch.get_rng_state().clone();s=ops.MLPTrainingSession(dataset,D0,seed,'cpu');initial=s.snapshot();assert not np.count_nonzero(s.predict(dataset['x'][0]))
    dataset_ref=saved(folder/'DATASET',dataset,binding);initial_ref=saved(folder/'INITIAL',initial,binding)
    curves=[dict(step=0,**train_metrics(s,dataset))];events=[];latest=folder/'RECOVERY.json'
    if latest.exists():
        recover=loaded(meta(ref(latest))['state']);assert recover['binding']==binding;s.restore(recover['snapshot']);curves=recover['curves'];events=recover['events']
    pending_before=None;began=time.monotonic()
    try:
        while s.completed<2000:
            if s.completed==1999:
                first_before=s.snapshot();pending_before=first_before;first=s.step();first_after=s.snapshot()
                clone=ops.MLPTrainingSession(dataset,D0,seed,'cpu');clone.restore(first_before);assert equal(first_before,clone.snapshot())
                second=clone.step();second_after=clone.snapshot();assert equal(first,second) and equal(first_after,second_after)
                s.restore(first_after);assert equal(first_after,s.snapshot())
                replay=saved(folder/'REPLAY1999_TO2000',dict(before=first_before,first_event=first,first_after=first_after,second_event=second,second_after=second_after,final_restored2000=s.snapshot()),binding)
                event=first
            else:event=s.step()
            events.append(event)
            pending_before=None
            if s.completed in (100,400,1000,2000):curves.append(dict(step=s.completed,**train_metrics(s,dataset)))
            if s.completed%100==0:
                status=dict(time=stamp(),step=s.completed,total=2000,elapsed_seconds=time.monotonic()-began,loss=event['loss'],gradient_norm_before_clip=event['gradient_norm_before_clip'])
                atomic(folder/'STATE.json',status);print(json.dumps(dict(event='L1_MLP_FIXED_ENDPOINT_PROGRESS',folder=str(folder),**status)),flush=True)
        if curves[-1]['step']!=2000:curves.append(dict(step=2000,**train_metrics(s,dataset)))
        assert s.completed==len(events)==2000 and torch.equal(before_rng,torch.get_rng_state())
        if not (folder/'REPLAY1999_TO2000/MANIFEST.json').exists():raise RuntimeError('Missing actual endpoint replay; do not manufacture one after the fact')
        replay=ref(folder/'REPLAY1999_TO2000/MANIFEST.json');state=s.snapshot();fresh=ops.MLPTrainingSession(dataset,D0,seed,'cpu');fresh.restore(state);assert equal(state,fresh.snapshot())
        for j in sorted({0,len(dataset['x'])//2,len(dataset['x'])-1}):assert s.predict(dataset['x'][j]).tobytes()==fresh.predict(dataset['x'][j]).tobytes()
        endpoint=saved(folder/'ENDPOINT2000',state,binding);history=saved(folder/'CURVES_AND_BATCHES',dict(curves=curves,events=events),binding)
        return read(checked(atomic_result(final,dict(status='COMPLETE_L1_MLP_FIXED2000_REQUIRES_ROOT',passed=True,binding=binding,source_files=sources(),endpoint=endpoint,
            dataset=dataset_ref,initial=initial_ref,curves=history,replay1999_to2000=replay,completed=2000,initial_seed=int(seed),baseline_MSE=curves[0],final_MSE=curves[-1],
            weak_fit_is_not_native_PDMS_veto=True,RNG_unchanged=True,private_batch_schedule_saved=True))))
    except BaseException:
        error=folder/('FAILED_'+uuid.uuid4().hex+'.json');atomic(error,dict(time=stamp(),binding=binding,step=s.completed,traceback=traceback.format_exc()),exclusive=True)
        try:
            if pending_before is not None and s.completed!=len(events):s.restore(pending_before)
            recover=saved(folder/'RECOVERY_STATES'/uuid.uuid4().hex,dict(binding=binding,snapshot=s.snapshot(),curves=curves,events=events),binding)
            atomic(latest,dict(time=stamp(),state=recover))
        except BaseException:
            torch.save(s.snapshot(),folder/('NONFINITE_OR_UNSERIALIZABLE_FAILURE_'+uuid.uuid4().hex+'.pt'))
        raise

def inputs_for(state,d):
    o=d['ordinary'];v=ordinary_view(o['Base_pooled_float64'],o['token_counts_int64'],state['view']);B=state['space']['basis'];c=state['space']['center']
    z=coordinates(v,c,B);return v,z,o['raw_aux24_float32'].astype(np.float64)

def known(plan,release,method,threads=1):
    import torch
    torch.set_num_threads(threads);p,r,root=context(plan,release,'known');assert method in ('mlp','mlp_replay')
    path=root/'KNOWN'/method/'RESULT.json'
    if path.exists():v=read(path);assert v['plan']==ref(plan) and v['source_files']==sources() and v['method']==method;return ref(path)
    sp=read(root/'SPACES/D/RESULT.json');assert sp['plan']==ref(plan);state=loaded(sp['artifact']);assert state['space']['status']=='VALID_SPACE'
    d=asset_data(checked(p['data_proof']));v,z,aux=inputs_for(state,d);B=state['space']['basis'];c=state['space']['center'];f=coordinates(state['fail_view'],c,B);s=coordinates(state['success_view'],c,B)
    assert f.tobytes()==z[[d['pos'][i] for i in d['Q']]].tobytes()
    high=z[[d['pos'][i] for i in d['high']]];dataset=ops.paired_rows(f,s,high if method=='mlp_replay' else None)
    seed=initialization(state,1,method);normalized,_,_=ops.standardize(dataset['x'],dataset['weights']);teacher_seed=geo.seeded('MLP_KNOWN_TEACHER',state['fit_identity'],method)%(2**63-1)
    teacher,target,detail=ops.old.control_teacher(normalized.astype(np.float32),48,seed,teacher_seed,.25*state['D0']);dataset['y']=target.astype(np.float64)
    binding=dict(plan=ref(plan),space=ref(root/'SPACES/D/RESULT.json'),mode='known',method=method,teacher_seed=int(teacher_seed),source_files=sources())
    folder=root/'KNOWN'/method;teacher_ref=saved(folder/'TEACHER',dict(state=teacher.state_dict(),targets=target,detail=detail),binding)
    result=mlp_run(dataset,state['D0'],seed,folder/'TRAINING',binding);first=result['baseline_MSE'];last=result['final_MSE']
    assert first['weighted_MSE']>0 and all(v>0 for v in first['role_weighted_contribution'].values()),'Known target must be nonzero in every trained role'
    absorption=dict(all=1-last['weighted_MSE']/first['weighted_MSE'],**{k:1-last['role_weighted_contribution'][k]/first['role_weighted_contribution'][k] for k in first['role_MSE']})
    assert all(np.isfinite(x) for x in absorption.values());passed=all(x>=.9 for x in absorption.values())
    status='PASS_ACTUAL_L1_MLP_KNOWN2000_REQUIRES_ROOT' if passed else 'FAIL_ACTUAL_L1_MLP_KNOWN2000_NO_REAL_MLP_RELEASE'
    return atomic_result(folder/'RESULT.json',dict(status=status,passed=passed,plan=ref(plan),source_files=sources(),method=method,endpoint=2000,teacher=teacher_ref,training=ref(folder/'TRAINING/RESULT.json'),
        absorption=absorption,all_roles_absorption_ge90=passed,replay1999_to2000_exact=True,reference_zero_override=False,known_targets_are_not_real_GT=True))

def fit(plan,release,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,r,root=context(plan,release,'fit',unit);spec=next(v for v in p['units'] if v['id']==unit);route,order,K=spec['route'],spec['order'],spec['K']
    path=root/'BANKS'/unit/'RESULT.json'
    if path.exists():v=read(path);assert v['plan']==ref(plan) and v['source_files']==sources();return ref(path)
    sr=read(root/'SPACES'/route/'RESULT.json');assert sr['plan']==ref(plan);state=loaded(sr['artifact']);d=asset_data(checked(p['data_proof']));v,z,aux=inputs_for(state,d) if state['space']['status']=='VALID_SPACE' else (None,None,None)
    binding=dict(plan=ref(plan),unit=unit,source_files=sources());cluster=None;models=[]
    if state['space']['status']=='VALID_SPACE':cluster=geo.cluster(state['geometry'],order,K,state['tokens'],state['sessions'],state['fit_identity'])
    valid=cluster is not None and cluster['status']=='VALID_DIRECTION_CLUSTER'
    parent=None;pr=None
    if valid and K==1 and order=='C_to_P':
        parent=root/'BANKS'/f'{route}__P_to_C__K1'/'RESULT.json';pr=read(parent);assert pr['plan']==ref(plan) and pr['spec']['route']==route and pr['space']==ref(root/'SPACES'/route/'RESULT.json')
    if pr is not None and loaded(pr['clustering'])['status']=='VALID_DIRECTION_CLUSTER':
        assert np.all(cluster['labels']==0);models=pr['clusters']
        reuse=dict(source=ref(parent),same_Q_rows=True,same_B_center_D0_and_private_initialization=True,logical_order_retained=order)
    else:
        reuse=None
        for k in range(K):
            rows=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64);folder=path.parent/f'cluster{k:02d}';methods={};n=len(rows)
            if n:
                B=state['space']['basis'];c=state['space']['center'];f=coordinates(state['fail_view'][rows],c,B);s=coordinates(state['success_view'][rows],c,B)
                assert f.tobytes()==z[[d['pos'][int(state['Q_indices'][j])] for j in rows]].tobytes()
                common=ops.algebra(f,s,state['geometry']['coefficients'][rows],state['geometry']['L'],state['D0'],state['geometry']['orders']['P_to_C']['no_direction'][rows])
                common_ref=saved(folder/'COMMON',common,binding);high=z[[d['pos'][i] for i in d['high']]]
            else:common_ref=None
            paired_fit=None
            for method in ops.MENU:
                support=ops.support(method,n,len(d['high']),len(d['normal']))
                if not valid:support=dict(status='INVALID_SPACE_OR_CLUSTER',Base_fallback_required=True)
                if support['Base_fallback_required']:
                    methods[method]=dict(status=support['status'],Base_fallback_required=True);continue
                out=folder/method/'RESULT.json'
                if out.exists():entry=read(out);assert entry['binding']==binding;methods[method]=entry;continue
                model_binding=dict(**binding,method=method,cluster=k)
                if method.startswith('mlp'):
                    ds=ops.paired_rows(f,s,high if method=='mlp_replay' else None);seed=initialization(state,K,method)
                    result=mlp_run(ds,state['D0'],seed,folder/method/'TRAINING',model_binding)
                    entry=dict(status='VALID',kind='mlp',state=result['endpoint'],training=ref(folder/method/'TRAINING/RESULT.json'),initial_seed=int(seed),Base_fallback_required=False)
                elif method in ('paired_RRR','direction_amplitude'):
                    if paired_fit is None:
                        pp=[d['pp'][int(state['Q_indices'][j])] for j in rows];normal=[d['pos'][i] for i in d['normal']]
                        target,proof=ops.paired_targets(B,state['success_view'][rows]-state['fail_view'][rows],state['view'],state['geometry']['fractions'][rows],state['actual_fulltoken_RMS'][rows],state['D0'])
                        paired_fit=ops.fit_paired(state['fail_view'][rows],d['pairs']['raw_aux24_float32'][pp].astype(np.float64),v[normal],aux[normal],target,state['D0'])
                        paired_ref=saved(folder/'PAIRED_COMBINED',dict(model=paired_fit,target=target,proof=proof),binding)
                    else:paired_ref=ref(folder/'PAIRED_COMBINED/MANIFEST.json')
                    entry=dict(status='VALID',kind='paired',state=paired_ref,Base_fallback_required=False)
                else:
                    model=ops.fit_original(method,f,s,common,high)
                    model.pop('common',None);ar=saved(folder/method/'MODEL',model,model_binding)
                    entry=dict(status='VALID',kind='analytic',state=ar,Base_fallback_required=False)
                entry.update(binding=binding);atomic(out,entry,exclusive=True);methods[method]=entry
            assert set(methods)==set(ops.MENU)
            negative={m:dict(method=m,seed=int(geo.seeded('NEGATIVE_CONTROL',state['fit_identity'],state['recipe'],K,m)%(2**63-1)),control=True,noncompetitive=True,coefficient_L2_only_not_actual_physical_matched=True) for m in ops.CONTROLS}
            models.append(dict(cluster=k,Q_indices=state['Q_indices'][rows].tolist(),common=common_ref,methods=methods,negative_controls=negative,NO_DIRECTION_rows_retained=True))
    cluster_ref=saved(path.parent/'CLUSTER',cluster if cluster is not None else dict(status='INVALID_SPACE'),binding)
    return atomic_result(path,dict(status='COMPLETE_L1_BANK_ALL13_FIXED_ENDPOINTS_NOT_PDMS_SELECTED',passed=True,time=stamp(),source_files=sources(),plan=ref(plan),actual_release=ref(release),spec=spec,
        space=ref(root/'SPACES'/route/'RESULT.json'),clustering=cluster_ref,clusters=models,common_initialization_excludes_order_cluster=True,K1_exact_reuse=reuse,
        all_menu_slots_present=True,logical_invalid_slots_execute_Base=True,any_weak_MLP_fit_not_excluded=True,no_native_or_scores=True))

def make_predictor(state,common,method,entry,cache=None):
    """Compiled ordinary shape1 function; load one physical model once.

    The closure consumes only current ordinary v/aux/fraction, never a case ID,
    current teacher state/membership/score. Training-local statistics are state.
    """
    B=state['space']['basis'];center=state['space']['center'];cache={} if cache is None else cache
    if entry['Base_fallback_required']:return lambda v,aux,fraction:np.zeros(B.shape[1],np.float64)
    key=(entry['state']['path'],entry['state']['sha256'],hashlib.sha256(json.dumps(sources(),sort_keys=True).encode()).hexdigest())
    if key not in cache:cache[key]=loaded(entry['state'])
    fit=cache[key]
    if entry['kind']=='analytic':
        model=dict(**fit,common=common)
        return lambda v,aux,fraction:ops.predict_original(coordinates(v,center,B),model)
    if entry['kind']=='paired':return lambda v,aux,fraction:ops.predict_paired(v,aux,fit['model'],method,B,state['view'],np.array([fraction]))[0]
    import torch
    model=ops.old.make_mlp(B.shape[1],fit['seed'],'cpu');model.load_state_dict(fit['model']);model.eval().requires_grad_(False)
    def predict(v,aux,fraction):
        z=coordinates(v,center,B)
        with torch.no_grad():return model(torch.from_numpy(((z-fit['mean'])/fit['scale'])[None].astype(np.float32))).numpy()[0].astype(np.float64)*fit['D0']
    return predict

def pack(plan,release,unit,negative=False):
    p,r,root=context(plan,release,'pack-controls' if negative else 'pack',unit);bank=read(root/'BANKS'/unit/'RESULT.json');assert bank['passed'] and bank['plan']==ref(plan) and bank['source_files']==sources()
    state=loaded(meta(bank['space'])['artifact']);d=asset_data(checked(p['data_proof']));panel=p['panel_indices'];menu=list(ops.CONTROLS if negative else ops.MENU);K=bank['spec']['K'];rank=48
    coefficients=np.zeros((len(panel),K,len(menu),rank),np.float64);valid=np.zeros((K,len(menu)),bool)
    if state['space']['status']=='VALID_SPACE':
        v,z,aux=inputs_for(state,d)
        for model in bank['clusters']:
            k=model['cluster'];common=None if model['common'] is None else loaded(model['common'])
            cache={}
            for j,method in enumerate(menu):
                if not negative:
                    entry=model['methods'][method];valid[k,j]=not entry['Base_fallback_required']
                    predictor=make_predictor(state,common,method,entry,cache) if valid[k,j] else None
                for n,i in enumerate(panel):
                    pos=d['pos'][i];fraction=float(d['ordinary']['token_counts_int64'][pos,0]/d['ordinary']['token_counts_int64'][pos].sum())
                    if negative:
                        if common is not None:coefficients[n,k,j]=ops.negative_control(z[pos],method,common,model['negative_controls'][method]['seed'])['coefficients'];valid[k,j]=common is not None
                    else:
                        if valid[k,j]:coefficients[n,k,j]=predictor(v[pos],aux[pos],fraction)
    actions=[]
    for k in range(K):
        for j,m in enumerate(menu):
            for dose in p['doses']:
                spec=dict(unit=unit,cluster=k,method=m,alpha=dose['alpha'],cap_D0=dose['cap_D0'],control=negative,
                    plan_sha256=ref(plan)['sha256'],physical_bank_sha256=ref(root/'BANKS'/unit/'RESULT.json')['sha256'])
                actions.append(dict(action_id=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),**spec,method_position=j,valid=bool(valid[k,j]),invalid_executes_literal_Base=not bool(valid[k,j])))
    folder=root/('CONTROL_REQUESTS' if negative else 'REQUESTS')/unit
    take=np.array([d['pos'][i] for i in panel]);ordinary=d['ordinary'];ov=ordinary_view(ordinary['Base_pooled_float64'][take],ordinary['token_counts_int64'][take],state['view'])
    ar=saved(folder/'COMPACT',dict(indices_int64=np.array(panel,np.int64),raw_coefficients_float64=coefficients,valid_bool=valid,
        ordinary_role_means_float64=ordinary['Base_pooled_float64'][take],ordinary_view_float64=ov,raw_aux24_float32=ordinary['raw_aux24_float32'][take],token_counts_int64=ordinary['token_counts_int64'][take]),
        dict(plan=ref(plan),unit=unit,control=negative,source_files=sources()))
    decoder=meta(bank['space'])['decoder']
    return atomic_result(folder/'RESULT.json',dict(status='COMPLETE_CPU_ORDINARY_PANEL_REQUESTS_REQUIRES_NATIVE_VALIDATION',passed=True,plan=ref(plan),source_files=sources(),bank=ref(root/'BANKS'/unit/'RESULT.json'),
        compact=ar,space=bank['space'],decoder=decoder,actions=actions,panel_indices=panel,noise_indices=p['noises'],input_scope='ordinary_Base_only_session_held_prediction',
        logical_expert_outputs=len(panel)*len(actions)*len(p['noises']),control=negative,noncompetitive=negative,raw_prediction_preserves_magnitude=True,no_native_or_scoring=True))

def case_arrays(coefficients,actions,state,positions=None):
    """Pure raw-coefficient -> alpha -> direct FP32 request; never normalize."""
    positions=list(range(len(actions))) if positions is None else list(positions);assert len(positions)==len(set(positions)) and all(0<=j<len(actions) for j in positions)
    shifts=[];caps=[];identifiers=[]
    for j in positions:
        a=actions[j];q=coefficients[a['cluster'],a['method_position']]
        if a['valid']:
            v=geo.request(q,state['space']['basis'],state['view'],state['D0'],a['alpha'],a['cap_D0']);shift=v['requested_shifts_float64'][0]
        else:shift=np.zeros((2,384),np.float64)
        shifts.append(shift);caps.append(a['cap_D0']*state['D0']);identifiers.append(a['action_id'])
    shifts=np.asarray(shifts,np.float64).reshape(-1,2,384)
    return dict(action_ids=np.array(identifiers,dtype='U64'),requested_shifts_float64=shifts,requested_shifts_float32=shifts.astype(np.float32),cap_RMS_float64=np.array(caps,np.float64))

def materialize_case(request,index,positions=None):
    """Return canonical native CASE arrays, applying alpha before FP32 cast.

    No fitted model, teacher or current-scene GT is accessed here. Native must
    still perform the frozen actual FP32 Base-add downward cap.
    """
    r=read(request);assert r['passed'] and r['status']=='COMPLETE_CPU_ORDINARY_PANEL_REQUESTS_REQUIRES_NATIVE_VALIDATION' and r['source_files']==sources()
    raw=loaded(r['compact']);state=loaded(r['decoder']);n=r['panel_indices'].index(index)
    return case_arrays(raw['raw_coefficients_float64'][n],r['actions'],state,positions)

def selftest(output,proof=None):
    import torch
    torch.set_num_threads(1);assert not torch.cuda.is_initialized()
    rng=np.random.default_rng(2026102536);torch_rng=torch.get_rng_state().clone();B=np.eye(384,48,dtype=np.float64);D0=.2
    f=rng.normal(size=(16,48));s=f+rng.normal(size=f.shape)*.1;high=rng.normal(size=(9,48))
    count=np.tile(np.array([7,4],np.int64),(16,1));roles=rng.normal(size=(16,2,384));vf,vs,view=geo.old_geometry.view_vectors(roles,roles,count,'valid_shared384',np.arange(16))
    ov=ordinary_view(roles,count,view);assert ov.tobytes()==vf.tobytes()
    center=np.arange(384,dtype=np.float64)/1000;z=coordinates(vf,center,B);assert all(z[j].tobytes()==coordinates(ov[j],center,B).tobytes() for j in range(16))
    common=ops.algebra(f,s,s-f,np.eye(48)/np.sqrt(384),D0,np.zeros(16,bool));fixtures={}
    for method in ops.MENU:
        if method.startswith('mlp') or method in ('paired_RRR','direction_amplitude'):continue
        fitted=ops.fit_original(method,f,s,common,high);pred=ops.predict_original(f[0],fitted);assert pred.shape==(48,) and np.isfinite(pred).all();fixtures[method]=pred
    fv=f@B.T;sv=s@B.T;fractions=np.full(len(f),7/11,np.float64)
    target,target_proof=ops.paired_targets(B,sv-fv,view,fractions,np.full(len(f),D0,np.float64),D0)
    aux=rng.normal(size=(len(f),24));normal=rng.normal(size=(9,384));normalaux=rng.normal(size=(9,24))
    paired_model=ops.fit_paired(fv,aux,normal,normalaux,target,D0)
    for method in ('paired_RRR','direction_amplitude'):
        q0=ops.predict_paired(fv[0],aux[0],paired_model,method,B,view,fractions[:1]);assert q0.shape==(1,48) and np.isfinite(q0).all();fixtures[method]=q0[0]
    for method in ops.CONTROLS:
        negative=ops.negative_control(f[0],method,common,2026102539);assert negative['control'] and negative['noncompetitive'] and negative['coefficient_L2_only_not_actual_physical_dose_matched']
    sessions={};teacher_details={}
    for method in ('mlp','mlp_replay'):
        ds=ops.paired_rows(f,s,high if method=='mlp_replay' else None);norm,_,_=ops.standardize(ds['x'],ds['weights']);seed=2026102537
        teacher,target,detail=ops.old.control_teacher(norm.astype(np.float32),48,seed,2026102538,.25*D0);ds['y']=target.astype(np.float64)
        assert all(np.square(target[np.asarray(ds['row_roles'])==role]).mean()>0 for role in set(ds['row_roles']))
        session=ops.MLPTrainingSession(ds,D0,seed);first=session.step();before=session.snapshot();second=session.step();after=session.snapshot()
        clone=ops.MLPTrainingSession(ds,D0,seed);clone.restore(before);repeat=clone.step();assert equal(second,repeat) and equal(after,clone.snapshot())
        assert all(int(v['step'])==2 for v in after['optimizer']['state'].values());sessions[method]=after;teacher_details[method]=detail
    q=np.full((1,1,48),1e-6,np.float64);actions=[dict(action_id='a'*64,cluster=0,method_position=0,alpha=.5,cap_D0=2.,valid=True),dict(action_id='b'*64,cluster=0,method_position=0,alpha=2.,cap_D0=2.,valid=False)]
    state=dict(space=dict(basis=B),view=view,D0=D0);request=case_arrays(q,actions,state)
    assert np.count_nonzero(request['requested_shifts_float32'][1])==0 and np.max(np.abs(request['requested_shifts_float32'][0]))<D0*.001
    assert request['requested_shifts_float32'].tobytes()==request['requested_shifts_float64'].astype(np.float32).tobytes()
    binding=dict(synthetic_only=True,source_files=sources());savedref=saved(Path(output).parent/(Path(output).stem+'_NUMERIC'),dict(sessions=sessions,fixtures=fixtures,request=request),binding)
    assert equal(loaded(savedref)['sessions'],sessions) and torch.equal(torch_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    actual=None
    if proof is not None:
        d=asset_data(proof);take=np.array([d['pp'][i] for i in d['Q']]);pairs=d['pairs'];actual_f,_,actual_view=geo.old_geometry.view_vectors(pairs['Base_pooled_float64'][take],pairs['teacher_pooled_float64'][take],pairs['token_counts_int64'][take],'valid_shared384',np.arange(len(take)))
        actual_v=ordinary_view(d['ordinary']['Base_pooled_float64'],d['ordinary']['token_counts_int64'],actual_view);other=actual_v[[d['pos'][i] for i in d['Q']]]
        assert actual_f.tobytes()==other.tobytes();proj=coordinates(actual_f,center,B);online=np.stack([coordinates(row,center,B) for row in other]);assert proj.tobytes()==online.tobytes()
        actual=dict(data_proof=ref(proof),fit=len(d['fit']),Q=len(d['Q']),normal=len(d['normal']),high=len(d['high']),Base_view_bytes_exact=True,Base_view_max_error=float(np.max(np.abs(actual_f-other))),
            shape1_fit_vs_ordinary_coordinate_bytes_exact=True,synthetic_basis_used_only_for_arithmetic_fixture_not_real_space_fit=True)
    result=dict(status='PASS_CPU_L1_FITTER_SYNTHETIC_AND_SMALL_ASSET_IDENTITY_NO_REAL_FIT',passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),
        synthetic_two_step_ordinary_and_replay=teacher_details,synthetic_model_Adam_private_RNG_restore_exact=True,synthetic_state=savedref,actual_small_asset_check=actual,
        invalid_request_zero=True,small_prediction_not_normalized=True,direct_FP32_cast_exact=True,real_fits_clusters_or_2000_controls=0,GPU_native_scores_or_queue=0,
        actual_1999_to2000_control_not_yet_run=True,BLAS_thread_env={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
    return atomic_result(output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','space','known','fit','pack','selftest']);p.add_argument('--plan',type=Path);p.add_argument('--release',type=Path)
    p.add_argument('--data-proof',type=Path);p.add_argument('--output',type=Path);p.add_argument('--route',choices=['R','D']);p.add_argument('--method',choices=['mlp','mlp_replay']);p.add_argument('--unit');p.add_argument('--threads',type=int,default=1);p.add_argument('--negative',action='store_true');a=p.parse_args()
    assert 1<=a.threads<=8
    try:
        if a.action=='selftest':result=selftest(a.output,a.data_proof)
        elif a.action=='prepare':result=prepare(a.data_proof,a.output or ROOT/'PLAN.json')
        elif a.action=='space':result=space(a.plan,a.release,a.route)
        elif a.action=='known':result=known(a.plan,a.release,a.method,a.threads)
        elif a.action=='fit':result=fit(a.plan,a.release,a.unit,a.threads)
        else:result=pack(a.plan,a.release,a.unit,a.negative)
    except BaseException:
        atomic(ROOT/'FAILURES'/(str(time.time_ns())+'.json'),dict(time=stamp(),action=a.action,unit=a.unit,route=a.route,method=a.method,traceback=traceback.format_exc(),source_files=sources()),exclusive=True);raise
    print(json.dumps(result),flush=True)
