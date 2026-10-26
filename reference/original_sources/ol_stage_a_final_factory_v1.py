"""Stage A locked shared48 families: fresh final-inner and alltrain factory.

No native/model/score/queue. Actual numerical stages require ROOT release.
Only ordinary live Base post means/counts/aux enter the final predict_case.
"""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,json,time,hashlib,ast
from collections import OrderedDict
import numpy as np
from b5_common import read,atomic,stamp,sha
import ol_l3_outer_refit_v1 as math
import ol_cv_shared48_cpu_v1 as cv
inner,adapter,data,ops,geo=math.inner,math.cal,math.data,math.ops,math.geo
ref,meta,checked=data.ref,data.meta,data.checked
ROOT=OL/'preparation/stage_a_final_factory_v1'
WINNERS=OL/'analysis/stage_a_selection_v1/FROZEN_WINNERS.json'
WINNER_SELECTOR=OL/'code/ol_stage_a_selection_v1.py'
WINNER_REGISTRATION=OL/'preparation/stage_a_selection_v1/SOURCE_REGISTRATION.json'
WIN_STATUS='FROZEN_STAGE_A_TWO_CORE_LIBRARY_WINNERS_BEFORE_FINAL_FIT_V1'
PLAN_STATUS='READY_STAGE_A_LOCKED_SHARED48_CPU_FACTORY_NOT_EXECUTION_V1'
RELEASE_STATUS='PASS_ROOT_STAGE_A_FINAL_CPU_FACTORY_EXECUTION_V1'
INNER_BANK_STATUS='COMPLETE_STAGE_A_LOCKED_FAMILY_INNER_ALL13_FIXED2000_REQUIRES_ROOT_V1'
FINAL_BANK_STATUS='COMPLETE_STAGE_A_LOCKED_FAMILY_ALLTRAIN_SELECTED_BANK_REQUIRES_ROOT_V1'
SPACE_STATUS='COMPLETE_STAGE_A_OWN_TRAIN_SHARED48_SPACE_REQUIRES_ROOT_V1'
REQUEST_STATUS='COMPLETE_STAGE_A_FINAL_CAL_ORDINARY_REQUESTS_REQUIRES_NATIVE_V1'
COARSE_STATUS='FROZEN_STAGE_A_LOCKED_FAMILY_COARSE_METHODS_FOR_FULL_FINAL_CAL_V1'
FINAL_FREEZE_STATUS='FROZEN_STAGE_A_FINAL_INNERCAL_SLOTS_AND_WHOLE_VARIANTS_BEFORE_ALLTRAIN_REFIT_V1'
VARIANTS={'original':(1.,1.),'cap_x2':(1.,2.),'dose_and_cap_x2':(2.,2.)}
_CACHE=OrderedDict()


def sources():
    s=math.sources();s.update(cv.sources())
    for path in (__file__,WINNER_SELECTOR,WINNER_REGISTRATION):s[str(Path(path).resolve())]=sha(path)
    return s
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def result(path,value):atomic(path,value,exclusive=True);return ref(path)
def save(path,value,binding):return inner.saved(path,value,binding)


def metadata():
    proof=meta(ref(cv.PROOF));split=meta(proof['session_split']);rows={r['index']:r for r in meta(proof['inputs'])['cases']};pair=read(cv.PAIR_PACK)
    qualified=set(pair['canonical_qualified_indices']);case_sources={r['index']:rr for rr in pair['cases'] for r in [meta(rr)]}
    assert pair['passed'] and pair['full1850']==proof['full1850'] and len(qualified)==1807 and len(rows)==10042
    train=split['final_locked_family_inner_fit'];cal=split['final_locked_family_inner_calibration'];ids=train['selected_indices'];calids=cal['selected_indices']
    assert len(ids)==8021 and len(calids)==2021 and cal['selected_counts']==dict(failure=363,mid=444,high=1214)
    assert len(train['sessions'])==104 and len(cal['sessions'])==26 and set(ids).isdisjoint(calids) and set(train['sessions']).isdisjoint(cal['sessions'])
    assert set(ids)|set(calids)==set(rows) and len({r['session'] for r in rows.values()})==130
    def scope(indices):
        q=[i for i in indices if i in qualified];normal=[i for i in indices if rows[i]['audit_category'] in ('mid','high')];high=[i for i in indices if rows[i]['audit_category']=='high']
        return dict(indices=indices,Q=q,normal=normal,high=high,sessions=sorted({rows[i]['session'] for i in indices}))
    scopes=dict(INNER=scope(ids),ALLTRAIN=scope(sorted(rows)));assert len(scopes['ALLTRAIN']['Q'])==1807 and len(scopes['ALLTRAIN']['normal'])==8192 and len(scopes['ALLTRAIN']['high'])==6144
    assert set(scopes['INNER']['Q'])<=set(scopes['ALLTRAIN']['Q'])
    return dict(proof=proof,split=split,rows=rows,pair=pair,pair_sources=case_sources,scopes=scopes,cal=cal,panel=cv.pilot_panel(0,cal,rows))


def inventory(output):
    d=metadata();value=dict(status='SOURCE_ONLY_STAGE_A_POPULATION_AND_FUTURE_WINNER_INVENTORY_NO_FIT_V1',producer=ref(__file__),source_files=sources(),data_proof=ref(cv.PROOF),session_split=d['proof']['session_split'],pair_pack=ref(cv.PAIR_PACK),
        scopes={k:dict(N=len(s['indices']),Q=len(s['Q']),normal=len(s['normal']),high=len(s['high']),sessions=len(s['sessions'])) for k,s in d['scopes'].items()},
        cal=dict(N=2021,counts=d['cal']['selected_counts'],sessions=26,panel32=d['panel']),future_winners=str(WINNERS),new_pair_payload_reads=0,actual_fit_or_score=0)
    return result(output,value)


def identity_registration(value,producer):
    reg=meta(value['source_registration']);assert reg['passed'] and reg.get('source',reg.get('producer'))==producer
    return reg


def complete_request_coverage(acceptance,request_refs,indices,noises,scope):
    a=meta(acceptance);q=meta(a['queue']);assert a['passed'] and a['source_files']==q['source_files'] and q['scope']==scope and q['noise_indices']==noises
    specs={meta(r)['unit']:(r,meta(r)) for r in request_refs};seen={u:np.zeros((len(indices),len(r['actions'])),bool) for u,(_,r) in specs.items()};where={i:n for n,i in enumerate(indices)}
    for uid,(_,r) in specs.items():assert r['panel_indices']==indices and r['noise_indices']==noises and r['producer']==ref(__file__) and r['source_files']==sources()
    for j in q['jobs']:
        assert j['unit'] in specs and j['request']==specs[j['unit']][0] and j['index'] in where
        pos=j['positions'];mask=seen[j['unit']];n=where[j['index']];assert pos and len(set(pos))==len(pos) and min(pos)>=0 and max(pos)<mask.shape[1] and not mask[n,pos].any();mask[n,pos]=True
    assert all(v.all() for v in seen.values())


def winner_guard(path,selector_source):
    import ol_stage_a_selection_v1 as producer
    selector_source=selector_source or WINNER_SELECTOR
    w=read(path);assert ref(selector_source)==ref(WINNER_SELECTOR) and w['passed'] and w['status']==WIN_STATUS and w['producer']==ref(selector_source) and w['source_files']==producer.sources()
    reg=identity_registration(w,w['producer']);assert w['source_registration']==ref(WINNER_REGISTRATION) and reg['source_files']==producer.sources() and reg['source_complete_OOF_matrices_read']==0
    assert w['stage_registration']==reg['stage_registration'] and w['all_candidates_full10042_130sessions'] and w['all_1850_failures_8192_references'] and w['population_total']==85813 and w['whole_dose_group_not_individually_cherry_picked']
    assert len(w['winners'])==2 and {r['order'] for r in w['winners']}=={'P_to_C','C_to_P'}
    for row in w['winners']:
        assert row['route'] in ('R','D') and 2<=row['K']<=8 and row['view']=='valid_shared384' and row['rank']==48 and row['dose_rule'] in ('original','whole_variant_selection_rule_v1')
        o=meta(row['source_oof']);assert o['passed']
    return w


def known_guard():
    k=meta(ref(cv.KNOWN));assert k['passed'] and k['status']=='PASS_ROOT_OFFLINE_LIBRARY_L1_MLP_KNOWN2000_ACCEPTANCE_V1'
    for method in ('mlp','mlp_replay'):
        c=meta(k['controls'][method]);assert c['passed'] and c['endpoint']==2000 and c['all_roles_absorption_ge90'] and c['replay1999_to2000_exact']


def prepare_inner(winners,selector_source,output):
    selector_source=selector_source or WINNER_SELECTOR
    w=winner_guard(winners,selector_source);d=metadata();s=d['scopes']['INNER'];known_guard();units=[]
    for v in w['winners']:
        uid='stageA__'+digest(v['family_id'])[:20];spec={k:v[k] for k in ('route','view','rank','order','K','dose_rule')};spec['id']=uid
        units.append(dict(id=uid,family_id=v['family_id'],spec=spec,winner=v,dependencies=[]))
    assert len({u['id'] for u in units})==2
    p=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),stage='INNER',winners=ref(winners),selector_source=ref(selector_source),selector_registration=w['source_registration'],
        units=units,routes=sorted({u['spec']['route'] for u in units}),view_name='valid_shared384',rank=48,data_proof=ref(cv.PROOF),ordinary_inputs=d['proof']['inputs'],session_split=d['proof']['session_split'],pair_pack=ref(cv.PAIR_PACK),
        train_indices=s['indices'],Q_indices=s['Q'],normal_indices=s['normal'],high_indices=s['high'],train_sessions=s['sessions'],train_pair_sources=[d['pair_sources'][i] for i in s['Q']],
        inner_Q_indices=s['Q'],cal_indices=d['cal']['selected_indices'],calibration_group=d['cal'],panel=d['panel'],panel_indices=[r['index'] for r in d['panel']],known_kernel_root=ref(cv.KNOWN),
        output_root=str(ROOT/'ACTUAL/INNER'),menu=list(ops.MENU),doses=read(inner.ROOT/'PLAN_FINAL.json')['doses'],
        original_parent=dict(winners=ref(winners),data_proof=ref(cv.PROOF)),no_cal_teacher_fit=True,locked_family_only_no_K_view_route_search=True)
    return result(output,p)


def final_guard(path,inner_plan,selector_source):
    f=read(path);p=read(inner_plan);assert f['passed'] and f['status']==FINAL_FREEZE_STATUS and f['producer']==ref(selector_source) and f['plan']==ref(inner_plan) and f['winners']==p['winners'];identity_registration(f,f['producer'])
    assert len(f['libraries'])==2 and {l['unit'] for l in f['libraries']}=={u['id'] for u in p['units']}
    for l in f['libraries']:
        u=next(u for u in p['units'] if u['id']==l['unit']);assert l['spec']==u['spec'] and l['dose_rule']==u['spec']['dose_rule']
        assert len(l['slots'])==u['spec']['K'] and [s['cluster'] for s in l['slots']]==list(range(u['spec']['K']))
        factor=VARIANTS[l['selected_whole_variant']];assert (l['raw_shift_factor'],l['cap_factor'])==factor
        if l['dose_rule']=='original':assert l['selected_whole_variant']=='original' and l['whole_variant_freeze'] is None
        else:
            whole=meta(l['whole_variant_freeze']);assert whole['passed'] and whole['plan']==ref(inner_plan) and whole['winners']==p['winners'];identity_registration(whole,whole['producer'])
            row=next(r for r in whole['libraries'] if r['unit']==l['unit'])
            assert row['selected_variant']==l['selected_whole_variant'] and (row['raw_shift_factor'],row['cap_factor'])==factor
        r=meta(l['inner_request']);b=meta(l['inner_bank']);assert r['status']==REQUEST_STATUS and b['status']==INNER_BANK_STATUS and r['source_files']==b['source_files']==sources() and r['plan']==b['plan']==ref(inner_plan)
        assert r['scope']=='innercal' and r['bank']==l['inner_bank'] and r['panel_indices']==p['cal_indices'] and r['noise_indices']==[0,1,2]
        for s in l['slots']:
            if s['Base']:assert s['selected_position']==-1 and s['method']=='Base' and s['alpha']==s['cap_D0']==0 and s['source_action_id'] is None
            else:
                a=r['actions'][s['selected_position']];assert a['action_id']==s['source_action_id'] and all(a[k]==s[k] for k in ('cluster','method','alpha','cap_D0'))
    complete_request_coverage(f['source_native_acceptance'],[l['inner_request'] for l in f['libraries']],p['cal_indices'],[0,1,2],'innercal')
    return f


def prepare_alltrain(inner_plan,selection,selector_source,output):
    p=read(inner_plan);assert p['status']==PLAN_STATUS and p['stage']=='INNER' and p['source_files']==sources() and p['producer']==ref(__file__)
    f=final_guard(selection,inner_plan,selector_source);d=metadata();s=d['scopes']['ALLTRAIN'];units=[]
    for l in f['libraries']:
        units.append(dict(id=l['unit'],spec=l['spec'],inner_bank=l['inner_bank'],inner_request=l['inner_request'],slots=l['slots'],dependencies=[],dose_rule=l['dose_rule'],
            selected_whole_variant=l['selected_whole_variant'],raw_shift_factor=l['raw_shift_factor'],cap_factor=l['cap_factor'],whole_variant_freeze=l['whole_variant_freeze']))
    q=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),stage='ALLTRAIN',units=units,routes=sorted({u['spec']['route'] for u in units}),view_name='valid_shared384',rank=48,
        winners=p['winners'],formal_selection=ref(selection),inner_plan=ref(inner_plan),selector_source=ref(selector_source),selector_registration=f['source_registration'],known_kernel_root=p['known_kernel_root'],
        data_proof=p['data_proof'],ordinary_inputs=p['ordinary_inputs'],session_split=p['session_split'],pair_pack=p['pair_pack'],train_indices=s['indices'],Q_indices=s['Q'],normal_indices=s['normal'],high_indices=s['high'],train_sessions=s['sessions'],
        train_pair_sources=[d['pair_sources'][i] for i in s['Q']],inner_Q_indices=p['Q_indices'],original_parent=dict(winners=p['winners'],inner_plan=ref(inner_plan),final_selection=ref(selection)),output_root=str(ROOT/'ACTUAL/ALLTRAIN'),
        selected_only=True,never_fit_or_match_on_val_test=True)
    return result(output,q)


def context(plan,release,action,unit=None):
    p=read(plan);r=read(release);assert p['status']==PLAN_STATUS and p['producer']==ref(__file__) and p['source_files']==sources() and p['stage'] in ('INNER','ALLTRAIN')
    assert r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan) and r['source_files']==sources() and r['stage']==p['stage'] and r['winners']==p['winners'] and action in r['allowed_actions']
    if p['stage']=='ALLTRAIN':assert r['formal_selection']==p['formal_selection']
    if unit is not None:assert unit in r['allowed_units'] and unit in {u['id'] for u in p['units']}
    known_guard();return p,Path(p['output_root'])


def data_stage(plan,release):
    p,root=context(plan,release,'data');path=root/'DATA/RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};parts=[]
    for i,rr in zip(p['Q_indices'],p['train_pair_sources']):
        r=meta(rr);assert r['index']==i and r['canonical_qualified_pair'] and r['full1850']==meta(p['data_proof'])['full1850'];a=data.small_arrays(r['arrays'])
        assert bool(a['canonical_qualified_pair_bool']) and float(a['canonical_Base_S0_score_float64'])<.7 and float(a['canonical_teacher_S0_score_float64'])>.9
        for k in ('Base_pooled_float64','token_counts_int64','raw_aux24_float32'):assert a[k].tobytes()==o[k][where[i]].tobytes()
        parts.append(a)
    d=dict(Q_indices=np.array(p['Q_indices'],np.int64),Base=np.stack([a['Base_pooled_float64'] for a in parts]),teacher=np.stack([a['teacher_pooled_float64'] for a in parts]),counts=np.stack([a['token_counts_int64'] for a in parts]),aux=np.stack([a['raw_aux24_float32'] for a in parts]),rms=np.array([float(a['actual_fulltoken_teacher_RMS_float64']) for a in parts]))
    ar=save(path.parent/'STATE',d,dict(plan=ref(plan),stage=p['stage'],source_files=sources(),scope='own_train_Q_only'))
    return result(path,dict(status='COMPLETE_STAGE_A_OWN_TRAIN_Q_ONLY_DATA_NO_AGGREGATE_TEACHER_V1',passed=True,producer=ref(__file__),source_files=sources(),plan=ref(plan),stage=p['stage'],artifact=ar,actual_Q=len(parts),aggregate_PAIRS_arrays_opened=0,cal_val_test_teacher_arrays_opened=0))


def space(plan,release,route):
    p,root=context(plan,release,'space');assert route in p['routes'];path=root/'SPACES'/route/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    d=inner.loaded(read(root/'DATA/RESULT.json')['artifact']);f,s,view=geo.old_geometry.view_vectors(d['Base'],d['teacher'],d['counts'],'valid_shared384',np.arange(len(d['Q_indices'])))
    fit=geo.fit_space(f,s,d['rms'],route,'valid_shared384',48);D0=float(np.median(d['rms']));rows=metadata()['rows']
    identity=digest(dict(data_proof=p['data_proof'],Q=p['Q_indices'],normal=p['normal_indices'],high=p['high_indices'],fit_sessions=p['train_sessions'],view='valid_shared384',scope='stage_a_'+p['stage']))
    state=dict(space=fit,view=view,D0=D0,Q_indices=d['Q_indices'],fail_view=f,success_view=s,actual_fulltoken_RMS=d['rms'],raw_aux=d['aux'],fit_identity=identity,recipe=dict(route=route,view='valid_shared384',rank=48),tokens=[rows[i]['token'] for i in p['Q_indices']],sessions=[rows[i]['session'] for i in p['Q_indices']])
    if fit['status']=='VALID_SPACE':state['geometry']=geo.geometry(fit['basis'],view,d['counts'][:,0]/d['counts'].sum(1),s-f,d['rms'])
    ar=save(path.parent/'STATE',state,dict(plan=ref(plan),source_files=sources(),stage=p['stage'],route=route));dec=data.arrays(path.parent/'DECODER.npz',dict(basis_float64=fit['basis'],center_float64=fit['center'],D0_float64=np.array(D0)))
    return result(path,dict(status=SPACE_STATUS,passed=True,producer=ref(__file__),source_files=sources(),plan=ref(plan),stage=p['stage'],route=route,artifact=ar,decoder=dec,view=view,space_status=fit['status'],fit_identity=identity,D0=D0,Q=len(f)))


def fit_inner(plan,release,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,root=context(plan,release,'fit-inner',unit);assert p['stage']=='INNER';f=p;spec=next(s['spec'] for s in p['units'] if s['id']==unit);K=spec['K'];path=root/'BANKS'/unit/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    sr=read(root/'SPACES'/spec['route']/'RESULT.json');assert sr['plan']==ref(plan) and sr['source_files']==sources();state=inner.loaded(sr['artifact'])
    valid=state['space']['status']=='VALID_SPACE';cluster=geo.cluster(state['geometry'],spec['order'],K,state['tokens'],state['sessions'],state['fit_identity']) if valid else dict(status='INVALID_SPACE')
    valid=valid and cluster['status']=='VALID_DIRECTION_CLUSTER';binding=dict(plan=ref(plan),stage=p['stage'],unit=unit,source_files=sources());models=[];reuse=None
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};v=inner.ordinary_view(o['Base_pooled_float64'],o['token_counts_int64'],state['view']) if valid else None
    B,center=state['space']['basis'],state['space']['center'];z=inner.coordinates(v,center,B) if valid else None
    if valid and next(u for u in p['units'] if u['id']==unit)['dependencies']:
        pr=read(root/'BANKS'/next(u for u in p['units'] if u['id']==unit)['dependencies'][0]/'RESULT.json');assert pr['plan']==ref(plan) and pr['space']==ref(root/'SPACES'/spec['route']/'RESULT.json')
        if pr['valid_cluster']:
            assert K==1 and (cluster['labels']==0).all();models=pr['clusters'];reuse=ref(root/'BANKS'/next(u for u in p['units'] if u['id']==unit)['dependencies'][0]/'RESULT.json')
    if reuse is None:
        for k in range(K):
            rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64);folder=path.parent/f'cluster{k:02d}';methods={};n=len(rr);common=None;cr=None
            if n:
                failed=inner.coordinates(state['fail_view'][rr],center,B);success=inner.coordinates(state['success_view'][rr],center,B)
                assert failed.tobytes()==z[[where[int(state['Q_indices'][j])] for j in rr]].tobytes()
                common=ops.algebra(failed,success,state['geometry']['coefficients'][rr],state['geometry']['L'],state['D0'],state['geometry']['orders']['P_to_C']['no_direction'][rr])
                cr=save(folder/'COMMON',common,binding);high=z[[where[i] for i in f['high_indices']]]
            paired_ref=None
            for method in ops.MENU:
                support=ops.support(method,n,len(f['high_indices']),len(f['normal_indices']))
                if not valid:support=dict(status='INVALID_SPACE_OR_CLUSTER',Base_fallback_required=True)
                if support['Base_fallback_required']:methods[method]=dict(status=support['status'],Base_fallback_required=True);continue
                out=folder/method/'RESULT.json'
                if out.exists():entry=read(out);assert entry['binding']==binding;methods[method]=entry;continue
                mb=dict(**binding,method=method,cluster=k)
                if method.startswith('mlp'):
                    ds=ops.paired_rows(failed,success,high if method=='mlp_replay' else None);sd=inner.initialization(state,K,method)
                    run=inner.mlp_run(ds,state['D0'],sd,folder/method/'TRAINING',mb)
                    entry=dict(status='VALID',kind='mlp',state=run['endpoint'],training=ref(folder/method/'TRAINING/RESULT.json'),initial_seed=int(sd),Base_fallback_required=False,
                        reused_numeric_engine=ref(inner.__file__),known_kernel_root=p['known_kernel_root'])
                elif method in ('paired_RRR','direction_amplitude'):
                    if paired_ref is None:
                        target,proof=ops.paired_targets(B,state['success_view'][rr]-state['fail_view'][rr],state['view'],state['geometry']['fractions'][rr],state['actual_fulltoken_RMS'][rr],state['D0'])
                        normal=[where[i] for i in f['normal_indices']];model=ops.fit_paired(state['fail_view'][rr],state['raw_aux'][rr].astype(np.float64),v[normal],o['raw_aux24_float32'][normal].astype(np.float64),target,state['D0'])
                        paired_ref=save(folder/'PAIRED_COMBINED',dict(model=model,target=target,proof=proof),binding)
                    entry=dict(status='VALID',kind='paired',state=paired_ref,Base_fallback_required=False)
                else:
                    model=ops.fit_original(method,failed,success,common,high);model.pop('common',None);ar=save(folder/method/'MODEL',model,mb)
                    entry=dict(status='VALID',kind='analytic',state=ar,Base_fallback_required=False)
                entry['binding']=binding;atomic(out,entry,exclusive=True);methods[method]=entry
            assert set(methods)==set(ops.MENU)
            models.append(dict(cluster=k,Q_indices=state['Q_indices'][rr].tolist(),common=cr,methods=methods,NO_DIRECTION_rows_retained=True,
                negative_controls_not_counted_as_repairs={m:dict(control=True,noncompetitive=True,coefficient_L2_only_not_actual_matched=True) for m in ops.CONTROLS}))
    ar=save(path.parent/'CLUSTER',cluster,binding)
    return result(path,dict(status=INNER_BANK_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),actual_release=ref(release),stage=p['stage'],spec=spec,
        space=ref(root/'SPACES'/spec['route']/'RESULT.json'),clustering=ar,clusters=models,valid_cluster=valid,K1_exact_reuse=reuse,all13_menu_slots_present=True,
        weak_MLP_fit_not_PDMS_veto=True,no_innercal_or_held_teacher_fit=True,MLP_endpoint=2000,replay1999_to2000_from_frozen_engine=True))

def refit_alltrain(plan,release,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,root=context(plan,release,'refit-alltrain',unit);assert p['stage']=='ALLTRAIN';u=next(v for v in p['units'] if v['id']==unit);K=u['spec']['K'];path=root/'BANKS'/unit/'RESULT.json'
    if path.exists():r=read(path);assert r['plan']==ref(plan) and r['source_files']==sources();return ref(path)
    sr=read(root/'SPACES'/u['spec']['route']/'RESULT.json');assert sr['plan']==ref(plan) and sr['source_files']==sources();state=inner.loaded(sr['artifact']);valid=state['space']['status']=='VALID_SPACE'
    cluster=geo.cluster(state['geometry'],u['spec']['order'],K,state['tokens'],state['sessions'],state['fit_identity']) if valid else dict(status='INVALID_SPACE');valid=valid and cluster['status']=='VALID_DIRECTION_CLUSTER'
    old=meta(u['inner_bank']);old_valid=bool(old.get('valid_cluster',meta(old['space'])['space_status']=='VALID_SPACE')) and old['clustering'] is not None
    if old_valid:old_valid=adapter.fitted_fields(old['clustering'],[('status',)])[0]=='VALID_DIRECTION_CLUSTER'
    valid=valid and old_valid
    B,center=state['space']['basis'],state['space']['center'];common=[];hashes=[];directions=[];binding=dict(plan=ref(plan),stage=p['stage'],view_name=p['view_name'],rank=p['rank'],unit=unit,source_files=sources())
    for k in range(K):
        rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64)
        c=ops.algebra(inner.coordinates(state['fail_view'][rr],center,B),inner.coordinates(state['success_view'][rr],center,B),state['geometry']['coefficients'][rr],state['geometry']['L'],state['D0'],state['geometry']['orders']['P_to_C']['no_direction'][rr]) if len(rr) else None
        h,v=math.physical_hash(c,state);common.append(c);hashes.append(h);directions.append(v)
    old_labels=adapter.fitted_fields(old['clustering'],[('labels',)])[0] if old_valid else None
    old_q=adapter.fitted_fields(meta(old['space'])['artifact'],[('Q_indices',)])[0]
    assert old_q.tolist()==p['inner_Q_indices'] and set(old_q)<=set(state['Q_indices']);whereQ={int(i):n for n,i in enumerate(state['Q_indices'])};C=np.zeros((K,K),np.int64)
    if valid:
        newlabels=cluster['labels'][[whereQ[int(i)] for i in old_q]];np.add.at(C,(old_labels,newlabels),1);mapping=math.exact_matching(C,hashes)
        mapping.update(ARI=math.adjusted_rand(C),common_indices=old_q,old_labels=old_labels,new_labels=newlabels,contingency=C,new_physical_centers=np.asarray(directions),new_physical_center_hashes=hashes)
    else:mapping=dict(old_to_new=list(range(K)),new_to_old=list(range(K)),contingency=C,status='INVALID_INNER_OR_OUTER_CLUSTER_EXPLICIT_BASE_NO_MATCH_CLAIM',ARI=None)
    cr=save(path.parent/'CLUSTER',cluster,binding);mr=save(path.parent/'TRANSFER',mapping,binding);parent=None
    if u['dependencies']:
        parent=read(root/'BANKS'/u['dependencies'][0]/'RESULT.json');assert parent['plan']==ref(plan) and parent['space']==ref(root/'SPACES'/u['spec']['route']/'RESULT.json')
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};ordinary=inner.ordinary_view(o['Base_pooled_float64'],o['token_counts_int64'],state['view']) if valid else None;models=[]
    for k in range(K):
        slot=mapping['new_to_old'][k];selection=u['slots'][slot];rr=np.flatnonzero(cluster['labels']==k) if valid else np.empty(0,np.int64);folder=path.parent/f'cluster{k:02d}';comm=save(folder/'COMMON',common[k],binding) if common[k] is not None else None
        support=ops.support(selection['method'],len(rr),len(p['high_indices']),len(p['normal_indices'])) if not selection['Base'] else dict(Base_fallback_required=True,status='FROZEN_INNER_SLOT_BASE')
        if not valid:support=dict(Base_fallback_required=True,status='INVALID_OUTER_SPACE_OR_CLUSTER')
        if support['Base_fallback_required']:entry=dict(status=support['status'],Base_fallback_required=True)
        elif parent is not None and parent['valid_cluster']:
            assert K==1 and parent['selected_slots'][0]['Q_indices']==state['Q_indices'][rr].tolist();entry=parent['selected_slots'][0]['entry']
        else:
            method=selection['method'];f=inner.coordinates(state['fail_view'][rr],center,B);s=inner.coordinates(state['success_view'][rr],center,B)
            if method.startswith('mlp'):
                high=inner.coordinates(ordinary[[where[i] for i in p['high_indices']]],center,B);ds=ops.paired_rows(f,s,high if method=='mlp_replay' else None);sd=inner.initialization(state,K,method)
                run=inner.mlp_run(ds,state['D0'],sd,folder/'MLP2000',dict(**binding,method=method,cluster=k));entry=dict(status='VALID',kind='mlp',state=run['endpoint'],training=ref(folder/'MLP2000/RESULT.json'),Base_fallback_required=False,reused_numeric_engine=ref(inner.__file__),known_kernel_root=p['known_kernel_root'],initial_seed=int(sd))
            elif method in ('paired_RRR','direction_amplitude'):
                target,proof=ops.paired_targets(B,state['success_view'][rr]-state['fail_view'][rr],state['view'],state['geometry']['fractions'][rr],state['actual_fulltoken_RMS'][rr],state['D0']);normal=[where[i] for i in p['normal_indices']]
                fit=math.fit_selected_paired(state['fail_view'][rr],state['raw_aux'][rr].astype(np.float64),ordinary[normal],o['raw_aux24_float32'][normal].astype(np.float64),target,state['D0'],method)
                ar=save(folder/'PAIRED_SELECTED',dict(model=fit,target=target,proof=proof),binding);entry=dict(status='VALID',kind='paired_single',state=ar,Base_fallback_required=False)
            else:
                high=inner.coordinates(ordinary[[where[i] for i in p['high_indices']]],center,B);fit=ops.fit_original(method,f,s,common[k],high);fit.pop('common',None)
                ar=save(folder/'MODEL',fit,binding);entry=dict(status='VALID',kind='analytic',state=ar,Base_fallback_required=False)
        uid=digest(dict(plan=ref(plan),space=ref(root/'SPACES'/u['spec']['route']/'RESULT.json'),cluster=cr,new_cluster=k,selection=selection,model=entry.get('state'),Q=state['Q_indices'][rr].tolist()))
        models.append(dict(new_cluster=k,old_slot=slot,selection=selection,Q_indices=state['Q_indices'][rr].tolist(),common=comm,entry=entry,physical_UID=uid))
    return result(path,dict(status=FINAL_BANK_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),deployment_scope='alltrain130',stage=p['stage'],view_name=p['view_name'],rank=p['rank'],original_parent=p['original_parent'],formal_selection=p['formal_selection'],spec=u['spec'],inner_bank=u['inner_bank'],space=ref(root/'SPACES'/u['spec']['route']/'RESULT.json'),clustering=cr,transfer=mr,selected_slots=models,valid_cluster=valid,
        K1_exact_reuse=None if parent is None or not valid or not parent['valid_cluster'] else ref(root/'BANKS'/u['dependencies'][0]/'RESULT.json'),inner_cluster_valid=old_valid,fresh_cluster_status=cluster['status'],inner_model_weights_or_responses_reused=False,held_GT_or_PDMS_read=False))



def coarse_guard(selection,selector_source,plan):
    s=read(selection);p=read(plan);assert s['passed'] and s['status']==COARSE_STATUS and s['producer']==ref(selector_source) and s['plan']==ref(plan) and s['winners']==p['winners'];reg=identity_registration(s,s['producer']);assert reg['native_scores_read_at_registration']==0
    a=meta(s['source_native_acceptance']);q=meta(a['queue']);assert a['passed'] and q['scope']=='pilot32' and q['noise_indices']==[0,1] and set(q['indices'])==set(p['panel_indices']) and a['source_files']==q['source_files']
    by={v['unit']:v for v in s['results']};assert len(by)==len(s['results'])==2 and set(by)=={u['id'] for u in p['units']}
    for u in p['units']:
        r=by[u['id']];assert r['spec']==u['spec'] and r['inner_bank']==ref(Path(p['output_root'])/'BANKS'/u['id']/'RESULT.json')
        req=meta(r['request']);assert req['status']==REQUEST_STATUS and req['producer']==ref(__file__) and req['source_files']==sources() and req['plan']==ref(plan) and req['scope']=='pilot32' and req['bank']==r['inner_bank']
        assert len(r['retained_per_slot'])==u['spec']['K'] and {v['cluster'] for v in r['retained_per_slot']}==set(range(u['spec']['K']))
        for v in r['retained_per_slot']:assert len(v['retained_methods'])==len(set(v['retained_methods']))==3 and v['retained_methods'][0]=='centroid_D0' and set(v['retained_methods'])<=set(ops.MENU) and v['all_seven_doses_retained']
    complete_request_coverage(s['source_native_acceptance'],[r['request'] for r in by.values()],p['panel_indices'],[0,1],'pilot32')
    return by


def pack_inner(plan,release,unit,scope,selection=None,selector_source=None):
    p,root=context(plan,release,'pack-pilot' if scope=='pilot32' else 'pack-cal',unit);assert p['stage']=='INNER' and scope in ('pilot32','innercal');u=next(u for u in p['units'] if u['id']==unit);K=u['spec']['K']
    menus=[list(ops.MENU) for _ in range(K)] if scope=='pilot32' else [v['retained_methods'] for v in sorted(coarse_guard(selection,selector_source,plan)[unit]['retained_per_slot'],key=lambda v:v['cluster'])]
    ids=p['panel_indices'] if scope=='pilot32' else p['cal_indices'];folder=root/('PILOT_REQUESTS' if scope=='pilot32' else 'CAL_REQUESTS')/unit;path=folder/'RESULT.json'
    bank=read(root/'BANKS'/unit/'RESULT.json');sp=meta(bank['space']);dec=data.small_arrays(sp['decoder']);state=dict(space=dict(basis=dec['basis_float64'],center=dec['center_float64']),D0=float(dec['D0_float64']),view=sp['view'])
    o=data.small_arrays(meta(p['data_proof'])['ordinary_arrays']);where={int(i):n for n,i in enumerate(o['indices_int64'])};take=[where[i] for i in ids];roles=o['Base_pooled_float64'][take];counts=o['token_counts_int64'][take];aux=o['raw_aux24_float32'][take];v=inner.ordinary_view(roles,counts,state['view']);M=len(menus[0]);raw=np.zeros((len(ids),K,M,48));valid=np.zeros((K,M),bool);cache={}
    for c in bank['clusters']:
        k=c['cluster'];common=None
        for j,method in enumerate(menus[k]):
            e=c['methods'][method];valid[k,j]=bank['valid_cluster'] and not e['Base_fallback_required']
            if not valid[k,j]:continue
            if common is None and e['kind']=='analytic':common=inner.loaded(c['common'])
            predict=adapter.predictor(e,common,state,method,cache)
            for n in range(len(ids)):raw[n,k,j]=predict(v[n],aux[n].astype(np.float64),float(counts[n,0]/counts[n].sum()))
    actions=[]
    for k in range(K):
        for j,method in enumerate(menus[k]):
            for dose in p['doses']:
                a=dict(unit=unit,cluster=k,method=method,alpha=dose['alpha'],cap_D0=dose['cap_D0'],method_position=j,valid=bool(valid[k,j]),invalid_executes_literal_Base=not bool(valid[k,j]),control=False)
                actions.append(dict(action_id=digest(dict(plan=ref(plan),action=a)),**a))
    rows=metadata()['rows'];weights=np.array([r['natural_mean_weight'] for r in p['panel']]) if scope=='pilot32' else np.array([p['calibration_group']['conditional_natural_mean_weight_per_sample'][rows[i]['audit_category']] for i in ids])
    ar=data.arrays(folder/'COMPACT.npz',dict(indices_int64=np.array(ids,np.int64),raw_coefficients_float64=raw,natural_mean_weights_float64=weights,ordinary_role_means_float64=roles,token_counts_int64=counts,raw_aux24_float32=aux))
    return result(path,dict(status=REQUEST_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),scope=scope,unit=unit,spec=u['spec'],bank=ref(root/'BANKS'/unit/'RESULT.json'),space=bank['space'],compact=ar,decoder=sp['decoder'],view=state['view'],actions=actions,methods_per_slot=menus,panel_indices=ids,noise_indices=[0,1] if scope=='pilot32' else [0,1,2],input_scope='ordinary_Base_only_stage_a_final_cal_prediction',teacher_payloads_opened_for_prediction=0,logical_expert_outputs=len(ids)*len(actions)*(2 if scope=='pilot32' else 3)))


def materialize_case(request,index,positions=None):
    rr=request if isinstance(request,dict) else ref(request);key=(rr['path'],rr['sha256'],digest(sources()))
    if key not in _CACHE:
        r=meta(rr);assert r['status']==REQUEST_STATUS and r['passed'] and r['producer']==ref(__file__) and r['source_files']==sources() and r['teacher_payloads_opened_for_prediction']==0
        a=data.small_arrays(r['compact']);d=data.small_arrays(r['decoder']);assert a['indices_int64'].tolist()==r['panel_indices'];_CACHE[key]=(r,a,d,{int(i):n for n,i in enumerate(a['indices_int64'])})
        if len(_CACHE)>4:_CACHE.popitem(last=False)
    r,a,d,where=_CACHE[key];return adapter.case_arrays(a['raw_coefficients_float64'][where[index]],r['actions'],d,r['view'],positions)


def predict_case(bank,role_means_float64,counts_int64,rawaux24_float32):
    rr=bank if isinstance(bank,dict) else ref(bank);key=('live',rr['path'],rr['sha256'],digest(sources()))
    if key not in _CACHE:
        b=meta(rr);assert b['passed'] and b['status']==FINAL_BANK_STATUS and b['producer']==ref(__file__) and b['source_files']==sources() and b['stage']=='ALLTRAIN'
        p=meta(b['plan']);sp=meta(b['space']);d=data.small_arrays(sp['decoder']);state=dict(space=dict(basis=d['basis_float64'],center=d['center_float64']),D0=float(d['D0_float64']),view=sp['view']);u=next(u for u in p['units'] if u['id']==b['spec']['id'])
        predictors=[];cache={}
        for m in sorted(b['selected_slots'],key=lambda v:v['old_slot']):
            e=m['entry'];method=m['selection']['method'];common=inner.loaded(m['common']) if not e['Base_fallback_required'] and e['kind']=='analytic' else None
            predict=None if e['Base_fallback_required'] else math.paired_predict(e,state,method) if e['kind']=='paired_single' else adapter.predictor(e,common,state,method,cache)
            predictors.append((m,predict))
        _CACHE[key]=(b,p,state,u,predictors)
        if len(_CACHE)>4:_CACHE.popitem(last=False)
    b,p,state,u,predictors=_CACHE[key];roles=np.asarray(role_means_float64);counts=np.asarray(counts_int64);aux=np.asarray(rawaux24_float32)
    assert roles.dtype==np.float64 and roles.shape==(2,384) and counts.dtype==np.int64 and counts.shape==(2,) and (counts>0).all() and aux.dtype==np.float32 and aux.shape==(24,) and np.isfinite(roles).all() and np.isfinite(aux).all()
    v=inner.ordinary_view(roles[None],counts[None],state['view'])[0];shifts=[];caps=[];ids=[]
    for m,predict in predictors:
        s=m['selection'];q=np.zeros(48) if predict is None else predict(v,aux.astype(np.float64),float(counts[0]/counts.sum()))
        physical,cap=decode_main(q,state,s['alpha'],s['cap_D0'],u['raw_shift_factor'],u['cap_factor'],predict is not None)
        shifts.append(physical);caps.append(cap);ids.append(digest(dict(bank=rr,physical_UID=m['physical_UID'],old_slot=m['old_slot'],whole_variant=u['selected_whole_variant'])))
    x=np.asarray(shifts,np.float64);return dict(action_ids=np.array(ids,dtype='U64'),requested_shifts_float64=x,requested_shifts_float32=x.astype(np.float32),cap_RMS_float64=np.asarray(caps,np.float64))


def decode_main(q,state,alpha,cap_D0,raw_factor,cap_factor,valid=True):
    assert np.asarray(q).dtype==np.float64 and np.asarray(q).shape==(48,) and np.isfinite(q).all()
    physical=np.zeros((2,384),np.float64) if not valid else geo.request(q,state['space']['basis'],state['view'],state['D0'],alpha,cap_D0)['requested_shifts_float64'][0]
    return physical*np.float64(raw_factor),np.float64(cap_D0)*state['D0']*np.float64(cap_factor)


def selftest(output):
    began=time.monotonic();d=metadata();rng=np.random.default_rng(2026102539);roles=rng.normal(size=(512,2,384));counts=rng.integers(1,30,size=(512,2),dtype=np.int64);teacher=roles+rng.normal(size=roles.shape)*.1;rms=np.full(512,.2)
    f,s,view=geo.old_geometry.view_vectors(roles,teacher,counts,'valid_shared384',np.arange(512));assert f.tobytes()==inner.ordinary_view(roles,counts,view).tobytes()
    checks=[]
    for route in ('R','D'):
        sp=geo.fit_space(f,s,rms,route,'valid_shared384',48);assert sp['status']=='VALID_SPACE' and sp['basis'].shape==(384,48)
        g=geo.geometry(sp['basis'],view,counts[:,0]/counts.sum(1),s-f,rms);assert np.isfinite(g['coefficients']).all();checks.append(dict(route=route,WLS=True,rank=48))
    C=np.array([[0,4],[5,0]],np.int64);m=math.exact_matching(C,['b','a']);assert m['old_to_new']==[1,0] and m['maximum_integer_overlap']==9
    state=dict(space=sp,view=view,D0=.2);q=rng.normal(size=48);shift,cap=decode_main(q,state,1.5,2.,2.,2.)
    expected=geo.request(q,sp['basis'],view,.2,1.5,2.)['requested_shifts_float64'][0]*2.
    assert shift.tobytes()==expected.tobytes() and shift.astype(np.float32).tobytes()==expected.astype(np.float32).tobytes() and cap==.8
    noop,_=decode_main(np.zeros(48),state,1.,2.,2.,2.,False);assert not noop.any()
    z=np.zeros((2,2,384),np.float64);z[1]=.123456789;raw=z*2.;assert raw.astype(np.float32).tobytes()==(z*np.float64(2)).astype(np.float32).tobytes()
    payload=dict(roles=roles[:4],counts=counts[:4],mapping=m);saved=save(Path(output).parent/'SYNTHETIC_STORAGE',payload,dict(synthetic_only=True,source_files=sources()));assert inner.equal(payload,inner.loaded(saved))
    for name in ('fit_inner','refit_alltrain','predict_case','materialize_case'):assert any(isinstance(n,ast.FunctionDef) and n.name==name for n in ast.parse(Path(__file__).read_text()).body)
    return result(output,dict(status='PASS_CPU_STAGE_A_METADATA_SHARED48_WLS_MATCH_AND_STORAGE_NO_ACTUAL_FIT_V1',passed=True,producer=ref(__file__),source_files=sources(),actual_scope_counts={k:dict(N=len(v['indices']),Q=len(v['Q']),normal=len(v['normal']),high=len(v['high']),sessions=len(v['sessions'])) for k,v in d['scopes'].items()},cal_counts=d['cal']['selected_counts'],panel_indices=[r['index'] for r in d['panel']],synthetic_spaces=checks,oldslot_integer_matching=True,FP64_whole_factor_before_single_cast=True,storage_roundtrip=True,new_pair_payload_or_actual_fit_model_score_or_queue_calls=0,active_seconds=time.monotonic()-began))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','selftest','prepare-inner','prepare-alltrain','data','space','fit-inner','refit-alltrain','pack-pilot','pack-cal']);p.add_argument('--winners',type=Path,default=WINNERS);p.add_argument('--selector-source',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--release',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--inner-plan',type=Path);p.add_argument('--route',choices=['R','D']);p.add_argument('--unit');p.add_argument('--threads',type=int,default=1);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='inventory':r=inventory(a.output or ROOT/'INVENTORY.json')
    elif a.action=='selftest':r=selftest(a.output or ROOT/'CPU_SELFTEST.json')
    elif a.action=='prepare-inner':r=prepare_inner(a.winners,a.selector_source,a.output or ROOT/'INNER_PLAN.json')
    elif a.action=='prepare-alltrain':r=prepare_alltrain(a.inner_plan,a.selection,a.selector_source,a.output or ROOT/'ALLTRAIN_PLAN.json')
    elif a.action=='data':r=data_stage(a.plan,a.release)
    elif a.action=='space':r=space(a.plan,a.release,a.route)
    elif a.action=='fit-inner':r=fit_inner(a.plan,a.release,a.unit,a.threads)
    elif a.action=='refit-alltrain':r=refit_alltrain(a.plan,a.release,a.unit,a.threads)
    else:r=pack_inner(a.plan,a.release,a.unit,'pilot32' if a.action=='pack-pilot' else 'innercal',a.selection,a.selector_source)
    print(json.dumps(r),flush=True)
