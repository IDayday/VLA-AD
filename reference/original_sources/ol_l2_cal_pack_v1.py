"""ROOT-gated full inner-calibration ordinary requests; no fitting or native calls.

L2 reuses L1 physical fitted models and the preregistered coarse selection.
Only inventory/selftest may run before an actual complete pilot summary exists.
"""
from pathlib import Path
import sys,os
sys.dont_write_bytecode=True
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
OL=Path(__file__).resolve().parents[1];P2=OL.parent
sys.path[:0]=[str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse,hashlib,json,time,traceback,uuid
from collections import Counter,OrderedDict
import numpy as np
from b5_common import read,atomic,stamp,sha
import ol_l1_pilot_fitter_v1 as fitter
data,ops,storage=fitter.data,fitter.ops,fitter.storage
ref,meta,checked=data.ref,data.meta,data.checked
ROOT=OL/'preparation/l2_cal_pack_v1'
PILOT=fitter.PILOT
FIT_PLAN=fitter.ROOT/'PLAN_FINAL.json'
DATA_PROOF=OL/'preparation/l1_complete_small_assets_v1/ROOT_ACCEPTANCE.json'
SELECTION_SOURCE=OL/'code/ol_l1_pilot_selection_v1.py'
SELECTION_REG=OL/'analysis/l1_pilot_selection_v1/SELECTION_SOURCE_REGISTRATION.json'
REVIEW=P2/'provenance/ASTRA_FIXED_EXTERNAL_VAL_AND_L0_L1_EXECUTION_REVIEW_V1.md'
SUMMARY_STATUS='COMPLETE_L1_FULL_MENU_COARSE_PILOT_FOR_FORMAL_CALIBRATION_NOT_FINAL_LIBRARIES'
PLAN_STATUS='READY_L2_FULL_INNERCAL1681_RETAINED_ORDINARY_REQUEST_PLAN_NOT_EXECUTION'
RELEASE_STATUS='PASS_ROOT_OFFLINE_LIBRARY_L2_CAL1681_CPU_PACK_EXECUTION_RELEASE_V1'
REQUEST_STATUS='COMPLETE_L2_CAL1681_ORDINARY_REQUESTS_REQUIRES_NATIVE_VALIDATION_V1'
_REQUEST_CACHE=OrderedDict()


def sources():
    value=fitter.sources()
    for p in (__file__,SELECTION_SOURCE,SELECTION_REG,REVIEW):value[str(Path(p).resolve())]=sha(p)
    return value

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def calibration_metadata():
    proof=read(DATA_PROOF);p=read(FIT_PLAN);pilot=read(PILOT);split=meta(proof['session_split']);inputs=meta(proof['inputs'])
    assert proof['passed'] and proof['status']==fitter.DATA_STATUS and proof['inputs']==ref(data.ROOT/'INPUTS10042.json')
    assert p['source_files']==fitter.sources() and p['data_proof']==ref(DATA_PROOF) and p['pilot']==ref(PILOT)
    assert pilot['source_session_split']==proof['session_split'] and pilot['outer_fold']==0
    fold=next(f for f in split['folds'] if f['fold']==0);group=fold['inner_calibration'];ids=list(group['selected_indices'])
    rows={r['index']:r for r in inputs['cases']};selected={r['index']:r for r in split['selected']}
    assert len(ids)==len(set(ids))==1681 and ids==pilot['complete_innercal_indices']
    assert set(ids)<=set(rows) and set(ids).isdisjoint(p['fit_indices']) and set(p['panel_indices'])<=set(ids)
    sessions=set(group['sessions']);assert len(sessions)==21 and sessions.isdisjoint(p['fit_sessions']) and sessions.isdisjoint(fold['outer_held']['sessions'])
    assert all(rows[i]['token']==selected[i]['token'] and rows[i]['session']==selected[i]['session'] and rows[i]['session'] in sessions for i in ids)
    categories=[rows[i]['audit_category'] for i in ids];counts=Counter(categories)
    assert counts==dict(failure=294,mid=360,high=1027)==group['selected_counts']
    population=group['full_population_counts'];total=group['full_population_scenes']
    assert population==dict(failure=294,mid=1491,high=11891) and total==13676==sum(population.values())
    per=group['conditional_natural_mean_weight_per_sample'];weights=np.array([per[c] for c in categories],np.float64)
    for c in counts:
        assert per[c]==population[c]/total/counts[c]
        assert abs(weights[np.array(categories)==c].sum()-population[c]/total)<1e-12
    assert abs(weights.sum()-1)<1e-12
    return dict(proof=proof,fit_plan=p,pilot=pilot,split=split,group=group,indices=ids,categories=categories,weights=weights,rows=rows)

def selection_units(summary,plan):
    """Validate reported pruning only. Never rank/reselect from pilot scores."""
    assert summary['passed'] and summary['status']==SUMMARY_STATUS and summary['all32_cross_behavior_matrices_saved']
    specs={u['id']:u for u in plan['units']};results={u['unit']:u for u in summary['results']}
    assert len(results)==len(summary['results'])==32 and set(results)==set(specs)
    keep=summary['retained_units'];assert len(keep)==len(set(keep)) and set(keep)<=set(results)
    for route in ('R','D'):
        chosen={specs[u]['K'] for u in keep if specs[u]['route']==route}
        assert 1 in chosen and len(chosen-{1})<=2
        assert {u for u in keep if specs[u]['route']==route}=={u for u,s in specs.items() if s['route']==route and s['K'] in chosen}
    assert sum(specs[u]['K'] for u in keep)<=68
    for unit,r in results.items():
        assert r['spec']==specs[unit]
        slots=r['retained_per_slot'];assert len(slots)==specs[unit]['K'] and {s['cluster'] for s in slots}==set(range(specs[unit]['K']))
        for s in slots:
            m=s['retained_methods'];assert len(m)==len(set(m))==3 and m[0]=='centroid_D0' and set(m)<=set(plan['menu'])
            assert s['all_seven_doses_retained']
    return [(u,results[u]) for u in keep]

def prepare(summary_path,output):
    d=calibration_metadata();reg=read(SELECTION_REG);summary=read(summary_path)
    assert reg['passed'] and reg['source']==ref(SELECTION_SOURCE) and reg['actual_candidate_scores_read']==0
    assert summary['producer']==ref(SELECTION_SOURCE) and summary['source_registration']==ref(SELECTION_REG) and summary['panel']==ref(PILOT)
    assert reg['scientific_registration']==d['pilot']['scientific_registration']
    selected=selection_units(summary,d['fit_plan']);accepted=meta(summary['native_acceptance'])
    assert accepted['passed'] and accepted['status']=='PASS_CPU_L1_NATIVE_ALL_EXPERT_CROSS_APPLICATION_REQUIRES_ROOT_V1'
    queue=meta(accepted['queue']);assert queue['scope']=='main32'
    assert accepted['source_files']==queue['source_files'] and accepted['jobs']==len(queue['jobs']) and accepted['scenes']==32
    assert queue['noise_indices']==[0,1] and sum(len(j['positions'])*2 for j in queue['jobs'])==838656
    banks=meta(queue['complete_bank_root']);assert banks['passed'] and banks['status']=='PASS_ROOT_OFFLINE_LIBRARY_L1_COMPLETE_PILOT_BANK_AND_REQUESTS_V1' and banks['plan']==ref(FIT_PLAN)
    assert banks['fitter_source_files']==fitter.sources() and banks['logical_native_expert_requests']==838656
    request_by_unit={meta(r)['bank']['path']:r for r in banks['requests']};units=[]
    for name,r in selected:
        original=meta(r['request']);bank=meta(original['bank']);spec=bank['spec']
        assert r['request']==request_by_unit[original['bank']['path']] and bank['source_files']==original['source_files']==fitter.sources()
        assert bank['plan']==original['plan']==ref(FIT_PLAN) and spec==r['spec'] and spec['id']==name
        slots=sorted(r['retained_per_slot'],key=lambda s:s['cluster'])
        methods=[s['retained_methods'] for s in slots];positions=[j for j,a in enumerate(original['actions']) if a['method'] in methods[a['cluster']]]
        assert len(positions)==spec['K']*3*7
        units.append(dict(id=name,spec=spec,bank=original['bank'],pilot_request=r['request'],methods_per_slot=methods,pilot_positions=positions,
            logical_actions=spec['K']*21,logical_expert_outputs=1681*spec['K']*21*3))
    result=dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),data_proof=ref(DATA_PROOF),
        source_selection_summary=ref(summary_path),source_selection_registration=ref(SELECTION_REG),source_pilot_acceptance=summary['native_acceptance'],
        complete_bank_root=queue['complete_bank_root'],fitter_plan=ref(FIT_PLAN),pilot=ref(PILOT),session_split=d['proof']['session_split'],ordinary_inputs=d['proof']['inputs'],
        units=units,indices=d['indices'],rows=[dict(index=i,token=d['rows'][i]['token'],session=d['rows'][i]['session'],category=c,natural_mean_weight=float(w)) for i,c,w in zip(d['indices'],d['categories'],d['weights'])],
        population_counts=d['group']['full_population_counts'],population_total=13676,selected_counts=d['group']['selected_counts'],sessions=d['group']['sessions'],
        weight_source='SPLITS fold0.inner_calibration conditional_natural_mean_weight_per_sample, copied float64; conditional13676 population, not global85813 or panel32 counts',
        doses=d['fit_plan']['doses'],noise_indices=[0,1,2],logical_expert_outputs=sum(u['logical_expert_outputs'] for u in units),
        output_root=str(ROOT/'ACTUAL'),no_new_fit_or_teacher_target_read=True,calibration_role='complete_innercal_development_including_pilot32_not_independent_confirmation')
    assert result['logical_expert_outputs']<=7201404
    atomic(output,result,exclusive=True);return ref(output)

def _tree_at(node,keys):
    for key in keys:
        assert node['kind']=='dict';items={k['value']:v for k,v in node['items'] if k['kind']=='scalar'};node=items[key]
    return node

def fitted_fields(record,paths):
    """Lazy selective NPZ members. Never decode paired scene tables or Adam."""
    m=meta(record);assert m['schema']=='ol_l1_lossless_numeric_artifact_v1' and m['producer']==ref(storage.__file__) and m['source_files']==storage.sources()
    with np.load(checked(m['arrays']),allow_pickle=False) as z:
        def decode(n):
            if n['kind']=='scalar':return n['value']
            if n['kind']=='dict':return {decode(k):decode(v) for k,v in n['items']}
            if n['kind'] in ('list','tuple'):
                values=[decode(v) for v in n['items']];return tuple(values) if n['kind']=='tuple' else values
            v=np.array(z[n['member']]);assert v.dtype.str==n['dtype'] and list(v.shape)==n['shape'] and fitter.geo.digest(v)==n['sha256']
            if n['kind']=='torch_tensor':
                import torch
                return torch.from_numpy(v)
            assert n['kind']=='ndarray';return v
        return [decode(_tree_at(m['tree'],p)) for p in paths]

def space_for_prediction(record):
    r=meta(record);assert r['source_files']==fitter.sources() and r['plan']==ref(FIT_PLAN)
    B,center,view,D0,status=fitted_fields(r['artifact'],[('space','basis'),('space','center'),('view',),('D0',),('space','status')])
    return dict(space=dict(basis=B,center=center,status=status),view=view,D0=D0)

def predictor(entry,common,state,method,cache):
    B=state['space']['basis'];center=state['space']['center'];kind=entry['kind'];record=entry['state']
    key=(kind,record['path'],record['sha256'])
    if key not in cache:
        if kind=='mlp':
            names=['model','mean','scale','D0','seed'];cache[key]=dict(zip(names,fitted_fields(record,[(n,) for n in names])))
        elif kind=='paired':
            names=['input_mean','input_scale','target_mean','target_scale','weight','bias'];values=fitted_fields(record,[('model','rank'),('model','D0'),*[('model','learned',n) for n in names]])
            cache[key]=dict(rank=values[0],D0=values[1],learned=dict(zip(names,values[2:])))
        else:
            assert kind=='analytic';cache[key]=fitter.loaded(record)
    model=cache[key]
    if kind=='analytic':
        fit=dict(**model,common=common)
        return lambda v,aux,fraction:ops.predict_original(fitter.coordinates(v,center,B),fit)
    if kind=='paired':return lambda v,aux,fraction:ops.predict_paired(v,aux,model,method,B,state['view'],np.array([fraction],np.float64))[0]
    import torch
    network=ops.old.make_mlp(B.shape[1],model['seed'],'cpu');network.load_state_dict(model['model']);network.eval().requires_grad_(False)
    def predict(v,aux,fraction):
        z=fitter.coordinates(v,center,B)
        with torch.no_grad():return network(torch.from_numpy(((z-model['mean'])/model['scale'])[None].astype(np.float32))).numpy()[0].astype(np.float64)*model['D0']
    return predict

def context(plan_path,release_path,unit):
    p=read(plan_path);r=read(release_path)
    assert p['status']==PLAN_STATUS and p['producer']==ref(__file__) and p['source_files']==sources()
    assert r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan_path) and r['source_files']==sources()
    assert r['source_selection_summary']==p['source_selection_summary'] and 'pack' in r['allowed_actions'] and unit in r['allowed_units']
    assert ref(checked(p['source_selection_summary']))==p['source_selection_summary']
    return p,next(u for u in p['units'] if u['id']==unit)

def pack(plan_path,release_path,unit,threads=1):
    import torch
    torch.set_num_threads(threads);p,u=context(plan_path,release_path,unit);folder=Path(p['output_root'])/'REQUESTS'/unit;result_path=folder/'RESULT.json'
    if result_path.exists():
        r=read(result_path);assert r['plan']==ref(plan_path) and r['source_files']==sources() and r['bank']==u['bank'];return ref(result_path)
    bank=meta(u['bank']);assert bank['plan']==p['fitter_plan'] and bank['source_files']==fitter.sources() and bank['spec']==u['spec']
    proof=meta(p['data_proof']);ordinary=data.small_arrays(proof['ordinary_arrays']);indices=ordinary['indices_int64'];where={int(i):n for n,i in enumerate(indices)};take=np.array([where[i] for i in p['indices']])
    assert len(indices)==len(set(indices))==10042 and indices.tolist()==meta(proof['inputs'])['indices']
    roles=ordinary['Base_pooled_float64'][take];counts=ordinary['token_counts_int64'][take];aux=ordinary['raw_aux24_float32'][take]
    assert roles.shape==(1681,2,384) and roles.dtype==np.float64 and counts.shape==(1681,2) and counts.dtype==np.int64 and np.all(counts>0)
    assert aux.shape==(1681,24) and aux.dtype==np.float32 and np.isfinite(roles).all() and np.isfinite(aux).all()
    state=space_for_prediction(bank['space']);view=fitter.ordinary_view(roles,counts,state['view']);K=u['spec']['K'];raw=np.zeros((1681,K,3,48),np.float64);valid=np.zeros((K,3),bool);method_records=[]
    assert view.shape==(1681,384) and state['view']['view']=='valid_shared384'
    if state['space']['status']=='VALID_SPACE':
        cache={}
        for cluster in bank['clusters']:
            k=cluster['cluster'];common=None
            for j,method in enumerate(u['methods_per_slot'][k]):
                entry=cluster['methods'][method];valid[k,j]=not entry['Base_fallback_required'];method_records.append(dict(cluster=k,method=method,model_entry=entry))
                if not valid[k,j]:continue
                if entry['kind']=='analytic' and common is None:common=fitter.loaded(cluster['common'])
                predict=predictor(entry,common,state,method,cache)
                for n in range(1681):raw[n,k,j]=predict(view[n],aux[n].astype(np.float64),float(counts[n,0]/counts[n].sum()))
    assert np.isfinite(raw).all()
    original=meta(u['pilot_request']);actions=[]
    for k in range(K):
        for j,method in enumerate(u['methods_per_slot'][k]):
            for dose in p['doses']:
                old=[(pos,a) for pos,a in enumerate(original['actions']) if a['cluster']==k and a['method']==method and a['alpha']==dose['alpha'] and a['cap_D0']==dose['cap_D0']]
                assert len(old)==1;position,a=old[0];spec=dict(unit=unit,cluster=k,method=method,alpha=dose['alpha'],cap_D0=dose['cap_D0'],plan_SHA=ref(plan_path)['sha256'],bank_SHA=u['bank']['sha256'])
                actions.append(dict(action_id=digest(spec),**spec,method_position=j,valid=bool(valid[k,j]),invalid_executes_literal_Base=not bool(valid[k,j]),pilot_position=position,pilot_action_id=a['action_id'],control=False))
    decoder=data.arrays(folder/'DECODER.npz',dict(basis_float64=state['space']['basis'],center_float64=state['space']['center'],D0_float64=np.array(state['D0'],np.float64)))
    compact=data.arrays(folder/'COMPACT.npz',dict(indices_int64=np.array(p['indices'],np.int64),raw_coefficients_float64=raw,valid_bool=valid,
        ordinary_role_means_float64=roles,ordinary_view_float64=view,raw_aux24_float32=aux,token_counts_int64=counts,natural_mean_weights_float64=np.array([r['natural_mean_weight'] for r in p['rows']],np.float64)))
    r=dict(status=REQUEST_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan_path),actual_release=ref(release_path),bank=u['bank'],
        source_selection_summary=p['source_selection_summary'],pilot_request=u['pilot_request'],space=bank['space'],decoder=decoder,view=state['view'],compact=compact,actions=actions,
        panel_indices=p['indices'],noise_indices=[0,1,2],method_records=method_records,methods_per_slot=u['methods_per_slot'],synthetic_only=False,
        input_scope='ordinary_Base_only_session_held_prediction',population_counts=p['population_counts'],selected_counts=p['selected_counts'],logical_expert_outputs=1681*len(actions)*3,
        calibration_role=p['calibration_role'],raw_prediction_preserves_magnitude=True,only_downward_native_cap_required=True,refitted_models=0,
        ordinary_response_or_teacher_payloads_opened=0,current_cal_teacher_arrays_opened=0,fit_space_selective_fields=['basis','center','view','D0','status'],MLP_Adam_or_paired_target_arrays_opened=0)
    assert len(actions)==u['logical_actions'] and r['logical_expert_outputs']==u['logical_expert_outputs']
    atomic(result_path,r,exclusive=True);return ref(result_path)

def case_arrays(raw,actions,decoder,view,positions=None):
    positions=list(range(len(actions))) if positions is None else list(positions)
    assert len(positions)==len(set(positions)) and all(0<=j<len(actions) for j in positions)
    shifts=[]
    for j in positions:
        a=actions[j]
        if a['valid']:
            q=raw[a['cluster'],a['method_position']];v=fitter.geo.request(q,decoder['basis_float64'],view,float(decoder['D0_float64']),a['alpha'],a['cap_D0']);shifts.append(v['requested_shifts_float64'][0])
        else:shifts.append(np.zeros((2,384),np.float64))
    shifts=np.asarray(shifts,np.float64).reshape(-1,2,384)
    return dict(action_ids=np.array([actions[j]['action_id'] for j in positions],dtype='U64'),requested_shifts_float64=shifts,requested_shifts_float32=shifts.astype(np.float32),
        cap_RMS_float64=np.array([actions[j]['cap_D0']*float(decoder['D0_float64']) for j in positions],np.float64))

def materialize_case(request,index,positions=None):
    """Own L2 producer/status guard; compact ordinary coefficients + decoder only."""
    record=ref(request) if not isinstance(request,dict) else request;key=(record['path'],record['sha256'],digest(sources()))
    if key not in _REQUEST_CACHE:
        r=meta(record);assert r['status']==REQUEST_STATUS and r['passed'] and not r['synthetic_only'] and r['producer']==ref(__file__) and r['source_files']==sources()
        assert r['input_scope']=='ordinary_Base_only_session_held_prediction' and r['raw_prediction_preserves_magnitude'] and r['refitted_models']==0
        compact=data.small_arrays(r['compact']);decoder=data.small_arrays(r['decoder']);assert compact['indices_int64'].tolist()==r['panel_indices']
        _REQUEST_CACHE[key]=(r,compact,decoder,{int(i):n for n,i in enumerate(compact['indices_int64'])})
        if len(_REQUEST_CACHE)>2:_REQUEST_CACHE.popitem(last=False)
    _REQUEST_CACHE.move_to_end(key);r,compact,decoder,pos=_REQUEST_CACHE[key]
    return case_arrays(compact['raw_coefficients_float64'][pos[index]],r['actions'],decoder,r['view'],positions)

def inventory(output):
    d=calibration_metadata();r=dict(status='CPU_L2_CAL1681_METADATA_AND_FUTURE_SELECTION_CONTRACT_NOT_ACTUAL_PACK',ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),
        data_proof=ref(DATA_PROOF),fitter_plan=ref(FIT_PLAN),pilot=ref(PILOT),selection_registration=ref(SELECTION_REG),session_split=d['proof']['session_split'],
        indices=d['indices'],rows=[dict(index=i,token=d['rows'][i]['token'],session=d['rows'][i]['session'],category=c,natural_mean_weight=float(w)) for i,c,w in zip(d['indices'],d['categories'],d['weights'])],
        selected_counts=d['group']['selected_counts'],population_counts=d['group']['full_population_counts'],population_total=13676,sessions=d['group']['sessions'],
        source_selection_summary_read=False,new_pack_executed=False,real_model_fit_native_or_scores=0,conditional_development_not_independent_confirmation=True)
    atomic(output,r,exclusive=True);return ref(output)

def selftest(output):
    d=calibration_metadata();p=d['fit_plan'];results=[]
    for u in p['units']:results.append(dict(unit=u['id'],spec=u,retained_per_slot=[dict(cluster=k,retained_methods=['centroid_D0','mean','local'],all_seven_doses_retained=True) for k in range(u['K'])]))
    keep=[u['id'] for u in p['units'] if u['K'] in (1,7,8)]
    fixture=dict(status=SUMMARY_STATUS,passed=True,all32_cross_behavior_matrices_saved=True,results=results,retained_units=keep)
    picked=selection_units(fixture,p);budget=sum(r['spec']['K']*21*1681*3 for _,r in picked);assert budget<=7201404
    B=np.eye(384,48);decoder=dict(basis_float64=B,center_float64=np.zeros(384),D0_float64=np.array(.2));raw=np.full((1,3,48),1e-6)
    actions=[dict(action_id=str(j)*64,cluster=0,method_position=j,alpha=a,cap_D0=c,valid=v) for j,(a,c,v) in enumerate([(.25,2.,True),(.5,1.,True),(1.,2.,False)])]
    arrays=case_arrays(raw,actions,decoder,dict(view='valid_shared384'));assert np.count_nonzero(arrays['requested_shifts_float32'][2])==0
    assert arrays['requested_shifts_float32'].tobytes()==arrays['requested_shifts_float64'].astype(np.float32).tobytes()
    assert np.max(np.abs(arrays['requested_shifts_float32'][0]))<1e-5 and arrays['cap_RMS_float64'].tolist()==[.4,.2,.4]
    numeric=data.arrays(Path(output).parent/'CPU_SELFTEST_ARRAYS.npz',arrays);again=data.small_arrays(numeric)
    assert fitter.equal(arrays,again)
    invalid={**fixture,'results':[dict(v,retained_per_slot=v['retained_per_slot']) for v in results]};invalid['results'][0]=dict(invalid['results'][0],retained_per_slot=[dict(cluster=0,retained_methods=['mean','mean','local'],all_seven_doses_retained=True)])
    try:selection_units(invalid,p)
    except AssertionError:pass
    else:raise AssertionError('Invalid retained menu was accepted')
    result=dict(status='PASS_CPU_L2_CAL_PACK_SYNTHETIC_SELECTION_WEIGHT_AND_REQUEST_ABI_NO_ACTUAL_PACK',passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),
        synthetic_selection_only=True,actual_selection_summary_read=False,selected_fixture_units=len(picked),fixture_expert_requests=budget,maximum_registered_expert_requests=7201404,
        calibration_rows=1681,counts=d['group']['selected_counts'],population_counts=d['group']['full_population_counts'],weights_sum=float(d['weights'].sum()),fixed_Splits_weight_values_exact=True,
        zero_invalid_exact=True,raw_small_prediction_preserved=True,alpha_before_direct_FP32=True,serialization_bytes_exact=True,duplicate_method_fixture_rejected=True,numeric_fixture=numeric,
        new_actual_packs_fits_native_scores_queue=0)
    atomic(output,result,exclusive=True);return ref(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inventory','prepare','pack','selftest']);p.add_argument('--output',type=Path)
    p.add_argument('--selection-summary',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--release',type=Path);p.add_argument('--unit');p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    assert 1<=a.threads<=8
    try:
        if a.action=='inventory':result=inventory(a.output or ROOT/'INVENTORY.json')
        elif a.action=='selftest':result=selftest(a.output or ROOT/'CPU_SELFTEST.json')
        elif a.action=='prepare':result=prepare(a.selection_summary,a.output or ROOT/'PLAN.json')
        else:result=pack(a.plan.resolve(),a.release.resolve(),a.unit,a.threads)
    except BaseException:
        atomic(ROOT/'FAILURES'/(str(time.time_ns())+'.json'),dict(time=stamp(),action=a.action,unit=a.unit,traceback=traceback.format_exc()),exclusive=True);raise
    print(json.dumps(result),flush=True)
