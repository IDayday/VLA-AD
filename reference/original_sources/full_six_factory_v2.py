"""User-fixed six libraries: original final arithmetic, explicit new metadata.

No queue or GPU calls. ROOT releases gate real fitting/collection. Legacy
numeric function code executes in private namespaces; old module globals and
frozen artifacts remain unchanged. Actual old producers are retained alongside
the new guard/callback execution closure. No old winner/ROOT proof is invented.
"""
from pathlib import Path
import os, sys
sys.dont_write_bytecode = True
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
ROOT = Path(__file__).resolve().parents[1]
OL = ROOT.parent
P2 = OL.parent
sys.path[:0] = [str(ROOT/'code'),str(OL/'code'),str(P2/'code'),str(P2.parent/'code')]
import argparse, copy, hashlib, json, types
from collections import OrderedDict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from b5_common import read, atomic, stamp, sha
import ol_stage_a_final_factory_v1 as base
import ol_stage_b_final_factory_v1 as generalized
import ol_l1_pilot_selection_v1 as coarse
import ol_l2_formal_selection_v2 as full
ref, meta, checked = base.ref, base.meta, base.checked
inner, adapter, data, ops, geo, math = base.inner, base.adapter, base.data, base.ops, base.geo, base.math
PREP = ROOT/'preparation/v2'
ANALYSIS = ROOT/'analysis/v2'
CATALOG = PREP/'USER_FIXED_CATALOG.json'
INNER_PLAN = PREP/'INNER_PLAN.json'
ALLTRAIN_PLAN = PREP/'ALLTRAIN_PLAN.json'
FINAL_CATALOG = PREP/'FINAL_CATALOG.json'
A_PROOF = OL/'preparation/stage_a_final_pipeline_v1/ALLTRAIN/ROOT_ACCEPTANCE.json'
A_FREEZE = OL/'analysis/stage_a_final_selection_v1/FINAL_FREEZE.json'
CATALOG_STATUS = 'USER_FIXED_SIX_DESCRIPTORS_NO_FIT_NO_EXTERNAL_SELECTION_V1'
PLAN_STATUS = 'READY_USER_FIXED_FULL_SIX_FINAL_CPU_PLAN_V1'
RELEASE_STATUS = 'PASS_ROOT_USER_FIXED_FULL_SIX_FACTORY_EXECUTION_V1'
INNER_BANK_STATUS = 'COMPLETE_FULL_SIX_FRESH_FINAL_INNER_ALL13_FIXED2000_V1'
FINAL_BANK_STATUS = 'COMPLETE_FULL_SIX_FRESH_FINAL_ALLTRAIN_SELECTED_BANK_V1'
REQUEST_STATUS = 'COMPLETE_FULL_SIX_FINAL_CAL_ORDINARY_REQUESTS_V1'
COARSE_STATUS = 'FROZEN_FULL_SIX_FRESH3_FINAL_COARSE_METHODS_V1'
SLOTS_STATUS = 'FROZEN_FULL_SIX_FRESH3_FINAL_ORIGINAL_SLOTS_V1'
FREEZE_STATUS = 'FROZEN_FULL_SIX_FRESH3_ORIGINAL_SLOTS_WITH_USER_FIXED_DOUBLE_V1'
PROGRAM_STATUS = 'FROZEN_USER_FIXED_FULL_SIX_RUNTIME_PROGRAM_V1'
FIELD_KEYS = ('route','view','rank','order','K','dose_rule')
FAMILIES = (
    dict(id='D_P7_original',route='D',order='P_to_C',K=7,dose_rule='original',raw_shift_factor=1.,cap_factor=1.,kind='reuse_A'),
    dict(id='D_C8_original',route='D',order='C_to_P',K=8,dose_rule='original',raw_shift_factor=1.,cap_factor=1.,kind='reuse_A'),
    dict(id='R_P8_double',route='R',order='P_to_C',K=8,dose_rule='user_fixed_raw_x2_cap_x2',raw_shift_factor=2.,cap_factor=2.,kind='fresh'),
    dict(id='R_C8_double',route='R',order='C_to_P',K=8,dose_rule='user_fixed_raw_x2_cap_x2',raw_shift_factor=2.,cap_factor=2.,kind='fresh'),
    dict(id='D_P8_double',route='D',order='P_to_C',K=8,dose_rule='user_fixed_raw_x2_cap_x2',raw_shift_factor=2.,cap_factor=2.,kind='fresh'),
    dict(id='D_C8_double',route='D',order='C_to_P',K=8,dose_rule='user_fixed_raw_x2_cap_x2',raw_shift_factor=2.,cap_factor=2.,kind='reuse_A'),
)
_SOURCE_CACHE = None
_NUMERIC = {}
_PROGRAM_CACHE = OrderedDict()


def require(v, label):
    if not v: raise AssertionError(label)


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sources():
    global _SOURCE_CACHE
    if _SOURCE_CACHE is None:
        s = base.sources(); s.update(generalized.sources())
        for p in (__file__,coarse.__file__,full.__file__): s[str(Path(p).resolve())] = sha(p)
        _SOURCE_CACHE = s
    return _SOURCE_CACHE.copy()


def verify_sources():
    for p,h in sources().items(): require(sha(p)==h,'frozen source changed: '+p)


def closure(entry):
    return dict(wrapper=ref(__file__),original_numeric_source=ref(base.__file__),original_function=entry,
                isolated_function_globals=True,legacy_module_globals_modified=False,
                original_numeric_code_sha256=hashlib.sha256(getattr(base,entry).__code__.co_code).hexdigest())


@lru_cache(maxsize=1)
def storage_writer():
    """Original lossless writer, with a real task-local preparation root.

    The code object/encoder/producer are genuinely the original storage
    function. Only its private OL root and manifest callback differ. Its
    original source_files remain exact for unmodified inner.loaded and
    adapter.fitted_fields; the additional actual execution closure is explicit.
    """
    original=inner.storage
    g=original.__dict__.copy()
    def write_manifest(path,value,*args,**kwargs):
        value=dict(value,actual_execution_sources=dict(writer_source=ref(original.__file__),writer_function='save_artifact',
            task_storage_adapter=ref(__file__),actual_preparation_root=str(ROOT/'preparation'),
            original_module_root_unchanged=str(original.OL/'preparation'),unchanged_numeric_encoder_code=True),
            execution_compatibility='Original code object; isolated OL root confines writes to full_six_evaluation/preparation; original readers unchanged.')
        return atomic(private_output(path),value,*args,**kwargs)
    g.update(OL=ROOT,atomic=write_manifest)
    fn=types.FunctionType(original.save_artifact.__code__,g,original.save_artifact.__name__,original.save_artifact.__defaults__,original.save_artifact.__closure__)
    require(fn.__code__ is original.save_artifact.__code__ and original.OL==OL,'original storage code and globals retained')
    return fn


def saved(folder,payload,binding):
    folder=private_output(folder)
    require(folder.is_relative_to(ROOT/'preparation'),'all V2 numeric artifacts use the actual task preparation subtree')
    path=folder/'MANIFEST.json'
    if path.exists():
        require(inner.equal(inner.loaded(ref(path)),payload),'existing numeric payload byte identity')
        require(meta(ref(path))['binding']==binding,'existing numeric binding identity')
        return ref(path)
    return storage_writer()(folder,payload,binding)


@lru_cache(maxsize=1)
def storage_inner():
    """Private inner facade: only persistence callbacks change, never Adam/RNG."""
    proxy=types.SimpleNamespace(**inner.__dict__)
    g=inner.__dict__.copy()
    def engine_atomic(path,value,*args,**kwargs):
        v=dict(value,actual_execution_sources=dict(numeric_engine=ref(inner.__file__),original_function='mlp_run',
            task_storage_adapter=ref(__file__),storage_writer=ref(inner.storage.__file__),only_persistence_callbacks_changed=True))
        return atomic(private_output(path),v,*args,**kwargs)
    def engine_result(path,value):
        engine_atomic(path,value,exclusive=True)
        return ref(path)
    g.update(saved=saved,atomic=engine_atomic,atomic_result=engine_result)
    proxy.saved=saved
    proxy.mlp_run=types.FunctionType(inner.mlp_run.__code__,g,inner.mlp_run.__name__,inner.mlp_run.__defaults__,inner.mlp_run.__closure__)
    require(proxy.mlp_run.__code__ is inner.mlp_run.__code__ and proxy.loaded is inner.loaded,'exact MLP code and original reader')
    return proxy


@lru_cache(maxsize=1)
def original_A_predictor():
    # Native processes validate their complete frozen source closure at startup.
    # Cache that exact identity here, so 85813 scenes do not rehash identical
    # source files per library. The original prediction code/math is unchanged.
    original_sources=base.sources();source_digest=digest(original_sources)
    g=base.__dict__.copy()
    g.update(sources=lambda:original_sources.copy(),digest=lambda v:source_digest if v==original_sources else base.digest(v),_CACHE=OrderedDict())
    return types.FunctionType(base.predict_case.__code__,g,base.predict_case.__name__,base.predict_case.__defaults__,base.predict_case.__closure__)


def private_output(path):
    p = Path(path).resolve()
    require(p.is_relative_to(ROOT) and p != ROOT,'new outputs must stay inside full_six_evaluation')
    return p


def result(path, value):
    atomic(private_output(path),value,exclusive=True)
    return ref(path)


def load_ref(value):
    return meta(value) if isinstance(value,dict) else read(value)


def as_ref(value):
    return value if isinstance(value,dict) else ref(value)


def expected_families():
    return [dict(**v,view='valid_shared384',rank=48) for v in FAMILIES]


@lru_cache(maxsize=1)
def metadata():
    return base.metadata()


def a_entities():
    proof = read(A_PROOF); f = read(A_FREEZE); p = meta(proof['alltrain_plan']); ip = meta(f['plan'])
    require(proof['passed'] and proof['alltrain130'] and proof['common_inner_Q_integer_matching_checked'],'actual A final proof')
    require(proof['factory_source_files']==base.sources() and proof['final_selection']==ref(A_FREEZE),'actual A factory lineage')
    require(p['stage']=='ALLTRAIN' and len(p['train_indices'])==10042 and len(p['train_sessions'])==130,'A must be final alltrain, never outer fold')
    require(ip['stage']=='INNER' and len(ip['train_indices'])==8021 and len(ip['train_sessions'])==104 and len(ip['cal_indices'])==2021,'actual A final INNER/cal split')
    md = metadata()
    for stage, plan in (('INNER',ip),('ALLTRAIN',p)):
        for k, mk in (('train_indices','indices'),('Q_indices','Q'),('normal_indices','normal'),('high_indices','high'),('train_sessions','sessions')):
            require(plan[k]==md['scopes'][stage][mk],'A reused scope exact '+stage+'/'+k)
    banks = {}
    for rr in proof['banks']:
        b = meta(rr)
        require(b['passed'] and b['producer']==ref(base.__file__) and b['source_files']==base.sources() and b['status']==base.FINAL_BANK_STATUS and b['plan']==proof['alltrain_plan'],'actual A bank')
        require(b['stage']=='ALLTRAIN' and b['deployment_scope']=='alltrain130','no fivefold bank substituted')
        key = (b['spec']['route'],b['spec']['order'],b['spec']['K'])
        lib = next(x for x in f['libraries'] if x['unit']==b['spec']['id'])
        require(lib['raw_shift_factor']==lib['cap_factor']==1. and lib['selected_whole_variant']=='original','A base bank must be original')
        banks[key] = rr
    require(set(banks)=={('D','P_to_C',7),('D','C_to_P',8)},'actual A two exact banks')
    return proof,f,p,ip,banks


def prepare_catalog(output=CATALOG):
    verify_sources(); proof,f,p,ip,banks = a_entities()
    programs = []
    for v in expected_families():
        row = dict(v)
        if v['kind']=='reuse_A': row['source_bank']=banks[v['route'],v['order'],v['K']]
        programs.append(row)
    return result(output,dict(status=CATALOG_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),
        authorization='Explicit current user instruction: only these six fixed libraries, full train85813 + TEST12146, S0..S4.',
        programs=programs,source_A_final_acceptance=ref(A_PROOF),source_A_final_freeze=ref(A_FREEZE),
        source_A_inner_plan=f['plan'],source_A_alltrain_plan=proof['alltrain_plan'],
        final_evaluation=dict(train_scenes=85813,test_scenes=12146,noise_indices=list(range(5)),
            selection='one Base-or-expert per scene after five-noise mean PDMS; same frozen expert/dose for all five noises; equal scene weights',
            cross_library_union=False,online_routing=False,alpha_search=False,random_controls=False,B_search_or_promotion=False),
        final_calibration=dict(pilot_scenes=32,pilot_noises=[0,1],cal_scenes=2021,cal_noises=[0,1,2],
            original_slot_method_and_alpha_only=True,whole_factors_fixed_by_user_not_calibration=True),
        no_old_winner_source_or_ROOT_signature_fabricated=True,no_fit_native_or_external_scores_read=True,
        storage_version=2,v1_failure_and_metadata_preserved=True,
        superseded_v1_metadata=[ref(x) for x in (ROOT/'preparation/USER_FIXED_CATALOG.json',ROOT/'preparation/INNER_PLAN.json',ROOT/'preparation/INNER_ROOT_RELEASE.json') if x.exists()]))


def program_record(row,catalog,bank,final_plan):
    v=dict(row);v['bank']=bank
    if row['kind']=='fresh':v['actual_execution_sources']=meta(bank)['actual_execution_sources']
    else:v['actual_execution_sources']=dict(original_factory=ref(base.__file__),reuse_root=meta(catalog)['source_A_final_acceptance'],new_dose_adapter=ref(__file__))
    v.update(status=PROGRAM_STATUS,passed=True,producer=ref(__file__),source_files=sources(),catalog=catalog,final_plan=final_plan,noise_indices=list(range(5)),original_bank_producer=meta(bank)['producer'],
        fixed_scene_expert_for_all_five_noises=True,root_final_acceptance_not_signed=True)
    return v


def save_program(value):
    path=PREP/'programs'/(value['id']+'.json')
    if path.exists():
        require(read(path)==value,'existing immutable program must be byte-identical metadata')
        return ref(path)
    return result(path,value)


def prepare_reused(catalog=CATALOG,output=PREP/'REUSED_A_PROGRAMS.json'):
    c=catalog_guard(catalog);programs=[]
    for row in c['programs']:
        if row['kind']=='reuse_A':programs.append(save_program(program_record(row,ref(catalog),row['source_bank'],c['source_A_alltrain_plan'])))
    require(len(programs)==3,'actual A originals and same-bank C8 double only')
    return result(output,dict(status='COMPLETE_THREE_ACTUAL_A_REUSE_RUNTIME_PROGRAMS_FRESH3_STILL_PENDING_V1',passed=True,producer=ref(__file__),source_files=sources(),catalog=ref(catalog),programs=programs,
        all_six_complete=False,fresh_programs_pending=[v['id'] for v in FAMILIES if v['kind']=='fresh'],new_fit_or_native_calls=0))


def catalog_guard(path):
    c = read(path)
    require(c['status']==CATALOG_STATUS and c['producer']==ref(__file__) and c['source_files']==sources(),'new user catalog source')
    require([{k:v[k] for k in expected_families()[0]} for v in c['programs']]==expected_families(),'exact six fixed descriptors/order/factors')
    require(c['final_evaluation']['noise_indices']==list(range(5)) and not c['final_evaluation']['cross_library_union'],'five noises and no union')
    require(c['source_A_final_acceptance']==ref(A_PROOF) and c['source_A_final_freeze']==ref(A_FREEZE),'actual A reuse refs')
    return c


def plan_guard(path):
    p = read(path); require(p['status']==PLAN_STATUS and p['producer']==ref(__file__) and p['source_files']==sources(),'full six actual plan')
    catalog_guard(p['catalog']['path'])
    require(p['catalog']==ref(p['catalog']['path']) and p['stage'] in ('INNER','ALLTRAIN'),'actual catalog/stage')
    md = metadata(); s = md['scopes'][p['stage']]
    for k,mk in (('train_indices','indices'),('Q_indices','Q'),('normal_indices','normal'),('high_indices','high'),('train_sessions','sessions')):
        require(p[k]==s[mk],'only prescribed final split '+k)
    require({u['id'] for u in p['units']}=={v['id'] for v in FAMILIES if v['kind']=='fresh'},'only three fresh units')
    for u in p['units']:
        wanted=next(v for v in expected_families() if v['id']==u['id'])
        require(all(u['spec'][k]==wanted[k] for k in FIELD_KEYS),'fresh descriptor exact')
    return p


def context(plan, release, action, unit=None):
    p = plan_guard(plan); r = read(release)
    require(r['passed'] and r['status']==RELEASE_STATUS and r['plan']==ref(plan) and r['source_files']==sources() and r['catalog']==p['catalog'] and r['stage']==p['stage'],'actual ROOT factory release')
    require(action in r['allowed_actions'],'ROOT action gate '+action)
    if unit is not None: require(unit in r['allowed_units'] and unit in {u['id'] for u in p['units']},'ROOT unit gate')
    if p['stage']=='ALLTRAIN': require(r['formal_selection']==p['formal_selection'],'ROOT frozen original slots')
    private_output(p['output_root']); base.known_guard()
    return p,Path(p['output_root'])


def prepare_inner(catalog=CATALOG, output=INNER_PLAN):
    verify_sources(); base.known_guard(); c = catalog_guard(catalog); md=metadata(); s=md['scopes']['INNER']
    units=[]
    for v in c['programs']:
        if v['kind']!='fresh': continue
        spec={k:v[k] for k in FIELD_KEYS};spec['id']=v['id']
        units.append(dict(id=v['id'],spec=spec,dependencies=[]))
    return result(output,dict(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),stage='INNER',catalog=ref(catalog),units=units,routes=['D','R'],view_name='valid_shared384',rank=48,
        data_proof=ref(base.cv.PROOF),ordinary_inputs=md['proof']['inputs'],session_split=md['proof']['session_split'],pair_pack=ref(base.cv.PAIR_PACK),
        train_indices=s['indices'],Q_indices=s['Q'],normal_indices=s['normal'],high_indices=s['high'],train_sessions=s['sessions'],train_pair_sources=[md['pair_sources'][i] for i in s['Q']],
        inner_Q_indices=s['Q'],cal_indices=md['cal']['selected_indices'],calibration_group=md['cal'],panel=md['panel'],panel_indices=[r['index'] for r in md['panel']],
        known_kernel_root=ref(base.cv.KNOWN),output_root=str(ROOT/'preparation/ACTUAL_V2/INNER'),menu=list(ops.MENU),doses=read(inner.ROOT/'PLAN_FINAL.json')['doses'],
        original_parent=dict(user_catalog=ref(catalog),source_A_data_plan=c['source_A_inner_plan']),numeric_seed_identity_scope='stage_a_INNER',
        original_slot_selection_only=True,whole_variant_selection=False,external_feedback_used=False))


def numeric(entry):
    """Execute original code object with explicit isolated metadata callbacks."""
    if entry in _NUMERIC: return _NUMERIC[entry]
    g=base.__dict__.copy(); source_digest=digest(sources())
    def fast_digest(v):
        return source_digest if v==sources() else base.digest(v)
    def numeric_result(path,value):
        value=dict(value,actual_execution_sources=closure(entry),factory_wrapper=ref(__file__))
        return result(path,value)
    g.update(context=context,sources=sources,metadata=metadata,coarse_guard=coarse_guard,result=numeric_result,digest=fast_digest,
        INNER_BANK_STATUS=INNER_BANK_STATUS,FINAL_BANK_STATUS=FINAL_BANK_STATUS,REQUEST_STATUS=REQUEST_STATUS,COARSE_STATUS=COARSE_STATUS,
        save=saved,inner=storage_inner(),_CACHE=OrderedDict())
    fn=types.FunctionType(getattr(base,entry).__code__,g,getattr(base,entry).__name__,getattr(base,entry).__defaults__,getattr(base,entry).__closure__)
    require(fn.__code__ is getattr(base,entry).__code__ and fn.__globals__ is not base.__dict__,'unchanged code, isolated globals')
    _NUMERIC[entry]=fn
    return fn


def reuse_stage(plan,release,kind):
    p,root=context(plan,release,'data' if kind=='DATA' else 'space')
    c=meta(p['catalog']); oldplan=meta(c['source_A_inner_plan'] if p['stage']=='INNER' else c['source_A_alltrain_plan'])
    suffix='DATA/RESULT.json' if kind=='DATA' else 'SPACES/D/RESULT.json'
    original=ref(Path(oldplan['output_root'])/suffix); old=meta(original)
    require(old['passed'] and old['plan']==(c['source_A_inner_plan'] if p['stage']=='INNER' else c['source_A_alltrain_plan']) and old['source_files']==base.sources(),'reuse exact actual final DATA/Dspace')
    out=root/suffix
    if out.exists():
        v=read(out); require(v['plan']==ref(plan) and v['source_files']==sources() and v['reused_actual_source']==original,'existing reuse sidecar');return ref(out)
    value={k:old[k] for k in ('artifact','decoder','view','space_status','fit_identity','D0','Q','route','actual_Q') if k in old}
    return result(out,dict(**value,status='REUSED_ACTUAL_FINAL_'+kind+'_ASSET_NO_NEW_FIT_V1',passed=True,producer=ref(__file__),source_files=sources(),plan=ref(plan),stage=p['stage'],
        reused_actual_source=original,original_producer=old['producer'],source_artifact_unchanged=True,new_space_or_DATA_fit=False,actual_execution_sources=dict(metadata_adapter=ref(__file__),original_source=original)))


def data_stage(plan,release): return reuse_stage(plan,release,'DATA')
def space(plan,release,route):
    require(route in ('D','R'),'only fixed shared48 routes')
    return reuse_stage(plan,release,'D_SPACE') if route=='D' else numeric('space')(plan,release,route)
def fit_inner(plan,release,unit,threads=1): return numeric('fit_inner')(plan,release,unit,threads)
def refit_alltrain(plan,release,unit,threads=1): return numeric('refit_alltrain')(plan,release,unit,threads)
def pack_inner(plan,release,unit,scope,selection=None):
    return numeric('pack_inner')(plan,release,unit,scope,selection,Path(__file__).resolve())
def materialize_case(request,index,positions=None): return numeric('materialize_case')(request,index,positions)


def coarse_guard(selection,selector_source,plan):
    p=plan_guard(plan);s=read(selection)
    require(s['passed'] and s['status']==COARSE_STATUS and s['producer']==ref(__file__) and s['source_files']==sources() and s['plan']==ref(plan),'actual fresh3 coarse selection')
    by={r['unit']:r for r in s['results']};require(set(by)=={u['id'] for u in p['units']},'all three coarse programs')
    for uid,r in by.items():
        require(len(r['retained_per_slot'])==8,'eight coarse slots')
        for row in r['retained_per_slot']:
            require(len(row['retained_methods'])==len(set(row['retained_methods']))==3 and row['retained_methods'][0]=='centroid_D0' and set(row['retained_methods'])<=set(ops.MENU) and row['all_seven_doses_retained'],'same centroid plus2/all7')
    return by


def request_proof(plan,release,scope,output):
    p,root=context(plan,release,'request-proof');require(p['stage']=='INNER' and scope in ('pilot32','innercal'),'cal scopes only')
    ids=p['panel_indices'] if scope=='pilot32' else p['cal_indices']; noise=[0,1] if scope=='pilot32' else [0,1,2]
    req=[]
    for u in p['units']:
        rr=ref(root/('PILOT_REQUESTS' if scope=='pilot32' else 'CAL_REQUESTS')/u['id']/'RESULT.json');r=meta(rr)
        require(r['passed'] and r['producer']==ref(base.__file__) and r['factory_wrapper']==ref(__file__) and r['source_files']==sources() and r['plan']==ref(plan) and r['spec']==u['spec'],'actual numeric request')
        require(r['panel_indices']==ids and r['noise_indices']==noise and r['teacher_payloads_opened_for_prediction']==0,'ordinary-only request population')
        z=materialize_case(rr,ids[0],[0,len(r['actions'])-1]);require(z['requested_shifts_float32'].tobytes()==z['requested_shifts_float64'].astype(np.float32).tobytes(),'direct FP32 request conversion')
        req.append(rr)
    return result(output,dict(status='COMPLETE_FULL_SIX_FRESH3_CAL_REQUESTS_CPU_PROOF_REQUIRES_ROOT_NATIVE_V1',passed=True,producer=ref(__file__),source_files=sources(),factory_plan=ref(plan),factory_source_files=sources(),
        scope=scope,requests=req,indices=ids,noise_indices=noise,logical_expert_outputs=sum(meta(r)['logical_expert_outputs'] for r in req),ROOT_acceptance_not_signed=True))


def collect(plan,release,acceptance,scope,workers=4):
    p,_=context(plan,release,'collect-'+scope);require(scope in ('pilot32','innercal') and 1<=workers<=4,'bounded saved-only collection')
    a=read(acceptance);q=meta(a['queue']);require(a['passed'] and a['source_files']==q['source_files'] and q['factory_plan']==ref(plan) and q['factory_source_files']==sources() and q['panel_scope']==scope and q['scope']==scope,'actual native main acceptance')
    for path,h in q['source_files'].items():require(sha(path)==h,'actual complete native execution source')
    require(all(q['source_files'].get(k)==v for k,v in sources().items()),'native actually binds new factory closure')
    ids=p['panel_indices'] if scope=='pilot32' else p['cal_indices']; noises=[0,1] if scope=='pilot32' else [0,1,2]
    require(q['noise_indices']==noises and set(q['indices'])==set(ids) and len(q['indices'])==len(ids),'same exact cal population/noises')
    req={};refs={};values={};seen={};where={i:n for n,i in enumerate(ids)};base_scores=np.full((len(ids),len(noises),7),np.nan);base_bytes={}
    jobs={j['id']:j for j in q['jobs']};used=set();require(len(jobs)==len(q['jobs'])==len(a['items'])==a['jobs'],'complete native items')
    for j in jobs.values():
        uid=j['unit']
        if uid in req:require(refs[uid]==j['request'],'one immutable request/unit');continue
        r=meta(j['request']);require(r['plan']==ref(plan) and r['source_files']==sources() and r['scope']==scope and r['panel_indices']==ids and r['noise_indices']==noises,'new ordinary request exact')
        req[uid]=r;refs[uid]=j['request'];values[uid]=np.full((len(ids),len(r['actions']),len(noises),7),np.nan);seen[uid]=np.zeros((len(ids),len(r['actions'])),bool)
    require(set(req)=={u['id'] for u in p['units']},'all three final fresh families')
    def item(rr):
        v=meta(rr)
        with np.load(checked(v['arrays']),allow_pickle=False) as z:
            return v,{k:np.array(z[k]) for k in ('scores_float64','Base_scores_float64','Base_trajectories_float32','positions_int64','action_ids')}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for v,z in ex.map(item,a['items']):
            j=v['job'];uid=j['unit'];n=where[j['index']];pos=j['positions']
            require(v['passed'] and j==jobs[j['id']] and j['id'] not in used and v['queue']==a['queue'] and v['source_files']==q['source_files'],'native item immutable job/source');used.add(j['id'])
            require(z['positions_int64'].tolist()==pos and z['action_ids'].tolist()==[req[uid]['actions'][k]['action_id'] for k in pos] and not seen[uid][n,pos].any(),'complete exact action coverage')
            require(v['noise_indices']==noises and z['scores_float64'].dtype==np.float64 and z['scores_float64'].shape==(len(pos),len(noises),7),'score dimensions')
            bb=(z['Base_scores_float64'].tobytes(),z['Base_trajectories_float32'].tobytes(),v['input_receipt'])
            if n in base_bytes:require(base_bytes[n]==bb,'literal RAW Base scores/trajectory/receipt repeated exact')
            else:base_bytes[n]=bb;base_scores[n]=z['Base_scores_float64']
            values[uid][n,pos]=z['scores_float64'];seen[uid][n,pos]=True
    require(used==set(jobs) and all(x.all() for x in seen.values()) and np.isfinite(base_scores).all() and all(np.isfinite(x).all() for x in values.values()),'all calibration outputs complete')
    rows=metadata()['rows'];weights=np.array([r['natural_mean_weight'] for r in p['panel']],np.float64) if scope=='pilot32' else np.array([p['calibration_group']['conditional_natural_mean_weight_per_sample'][rows[i]['audit_category']] for i in ids],np.float64)
    results=[];bm=base_scores[:,:,0].mean(1)
    for uid,t in sorted(values.items()):
        r=req[uid];acts=r['actions'];m=t[:,:,:,0].mean(2);matrix=data.arrays(ANALYSIS/scope/'matrices'/(uid+'.npz'),dict(indices_int64=np.array(ids,np.int64),scores_float64=t,Base_scores_float64=base_scores,natural_weights_float64=weights))
        row=dict(unit=uid,spec=r['spec'],inner_bank=r['bank'],inner_request=refs[uid],request=refs[uid],matrix=matrix)
        if scope=='pilot32':
            ci=next(k for k,x in enumerate(acts) if x['cluster']==0 and x['method']=='centroid_D0' and x['alpha']==1 and x['cap_D0']==2)
            row.update(selection=coarse.optimize(bm,m,weights,acts,8,ci),retained_per_slot=coarse.marginal_methods(bm,m,weights,acts,8,ci))
        else:
            opt=full.optimize(bm,m,weights,acts,8);row['selection']=opt
            row['slots']=[dict(cluster=k,Base=True,method='Base',alpha=0,cap_D0=0,selected_position=-1,source_action_id=None) if pos<0 else dict(cluster=k,Base=False,method=acts[pos]['method'],alpha=acts[pos]['alpha'],cap_D0=acts[pos]['cap_D0'],selected_position=pos,source_action_id=acts[pos]['action_id']) for k,pos in enumerate(opt['positions'])]
        results.append(row)
    output=ANALYSIS/('COARSE_SELECTION.json' if scope=='pilot32' else 'SLOTS.json')
    value=dict(status=COARSE_STATUS if scope=='pilot32' else SLOTS_STATUS,passed=True,producer=ref(__file__),source_files=sources(),plan=ref(plan),catalog=p['catalog'],source_native_acceptance=ref(acceptance),
        selection_source_functions=dict(coarse=ref(coarse.__file__),full=ref(full.__file__)),whole_factor_selection_performed=False)
    value['results' if scope=='pilot32' else 'libraries']=results
    return result(output,value)


def freeze(plan,release,slots,output=ANALYSIS/'FINAL_SLOT_FREEZE.json'):
    p,_=context(plan,release,'freeze');s=read(slots)
    require(s['passed'] and s['status']==SLOTS_STATUS and s['source_files']==sources() and s['producer']==ref(__file__) and s['plan']==ref(plan),'actual complete final-cal original slots')
    require({v['unit'] for v in s['libraries']}=={u['id'] for u in p['units']},'all three slot libraries')
    rows=[]
    for row in s['libraries']:
        v=dict(row);v.update(dose_rule='user_fixed_raw_x2_cap_x2',selected_whole_variant='user_fixed_double_no_variant_selection',raw_shift_factor=2.,cap_factor=2.,whole_variant_freeze=None)
        request=meta(row['inner_request'])
        for sl in row['slots']:
            if sl['Base']:require((sl['method'],sl['alpha'],sl['cap_D0'],sl['selected_position'],sl['source_action_id'])==('Base',0,0,-1,None),'literal Base slot')
            else:
                a=request['actions'][sl['selected_position']];require(sl['source_action_id']==a['action_id'] and all(sl[k]==a[k] for k in ('cluster','method','alpha','cap_D0')),'exact original slot and dose')
        rows.append(v)
    return result(output,dict(status=FREEZE_STATUS,passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),plan=ref(plan),catalog=p['catalog'],original_slots=ref(slots),source_native_acceptance=s['source_native_acceptance'],
        libraries=rows,whole_factor_fixed_by_user=True,original_not_a_whole_variant_candidate=True,no_final_population_score_feedback=True))


def prepare_alltrain(plan,selection,output=ALLTRAIN_PLAN):
    p=plan_guard(plan);f=read(selection);require(f['passed'] and f['status']==FREEZE_STATUS and f['plan']==ref(plan) and f['source_files']==sources() and f['catalog']==p['catalog'],'actual user-fixed final slot freeze')
    md=metadata();s=md['scopes']['ALLTRAIN'];units=[]
    for l in f['libraries']:
        require(l['raw_shift_factor']==l['cap_factor']==2. and l['whole_variant_freeze'] is None,'fixed double, no variant selection')
        units.append({k:l[k] for k in ('spec','inner_bank','inner_request','slots','dose_rule','selected_whole_variant','raw_shift_factor','cap_factor','whole_variant_freeze')}|dict(id=l['unit'],dependencies=[]))
    value={k:p[k] for k in ('catalog','routes','view_name','rank','data_proof','ordinary_inputs','session_split','pair_pack','known_kernel_root')}
    value.update(status=PLAN_STATUS,ready=True,time=stamp(),producer=ref(__file__),source_files=sources(),stage='ALLTRAIN',units=units,formal_selection=ref(selection),inner_plan=ref(plan),
        train_indices=s['indices'],Q_indices=s['Q'],normal_indices=s['normal'],high_indices=s['high'],train_sessions=s['sessions'],train_pair_sources=[md['pair_sources'][i] for i in s['Q']],inner_Q_indices=p['Q_indices'],
        original_parent=dict(user_catalog=p['catalog'],inner_plan=ref(plan),original_slot_freeze=ref(selection)),output_root=str(ROOT/'preparation/ACTUAL_V2/ALLTRAIN'),numeric_seed_identity_scope='stage_a_ALLTRAIN',selected_only=True,never_fit_or_match_on_val_test=True)
    return result(output,value)


def finalize_catalog(plan,release,output=FINAL_CATALOG):
    p,root=context(plan,release,'finalize-catalog');require(p['stage']=='ALLTRAIN','only final 130session banks')
    c=meta(p['catalog']);programs=[]
    for v in c['programs']:
        row=dict(v)
        if v['kind']=='fresh':
            br=ref(root/'BANKS'/v['id']/'RESULT.json');b=meta(br)
            require(b['passed'] and b['status']==FINAL_BANK_STATUS and b['producer']==ref(base.__file__) and b['factory_wrapper']==ref(__file__) and b['source_files']==sources() and b['plan']==ref(plan) and b['deployment_scope']=='alltrain130','actual fresh final bank')
            require(b['spec']==next(u['spec'] for u in p['units'] if u['id']==v['id']) and not b['inner_model_weights_or_responses_reused'],'fresh selected operator fitting')
            row=program_record(v,p['catalog'],br,ref(plan))
        else:
            row=program_record(v,p['catalog'],v['source_bank'],c['source_A_alltrain_plan'])
        programs.append(save_program(row))
    return result(output,dict(status='COMPLETE_USER_FIXED_SIX_FINAL_RUNTIME_PROGRAMS_REQUIRES_ROOT_V1',passed=True,time=stamp(),producer=ref(__file__),source_files=sources(),catalog=p['catalog'],alltrain_plan=ref(plan),
        formal_selection=p['formal_selection'],programs=programs,all_six_complete=True,ROOT_acceptance_not_signed=True,no_external_scores_opened=True))


def program_guard(value):
    rr=as_ref(value);key=(rr['path'],rr['sha256'])
    if key not in _PROGRAM_CACHE:
        p=meta(rr);require(p['status']==PROGRAM_STATUS and p['passed'] and p['producer']==ref(__file__) and p['source_files']==sources() and p['noise_indices']==list(range(5)),'actual frozen full-six program')
        wanted=next(x for x in expected_families() if x['id']==p['id'])
        require(all(p[k]==v for k,v in wanted.items()),'only six exact fixed program/factors')
        catalog_guard(p['catalog']['path']);require(p['catalog']==ref(p['catalog']['path']),'program actual catalog')
        b=meta(p['bank']);require(b['passed'] and b['stage']=='ALLTRAIN' and b['deployment_scope']=='alltrain130','runtime never uses a single outer-fold model')
        require(all(b['spec'][k]==p[k] for k in ('route','view','rank','order','K')),'program exact bank descriptor')
        if p['kind']=='reuse_A':require(b['source_files']==base.sources() and b['producer']==ref(base.__file__) and b['status']==base.FINAL_BANK_STATUS and b['plan']==p['final_plan'] and p['final_plan']==meta(p['catalog'])['source_A_alltrain_plan'],'actual original A bank/actual A final plan')
        else:
            require(b['source_files']==sources() and b['producer']==ref(base.__file__) and b['factory_wrapper']==ref(__file__) and b['status']==FINAL_BANK_STATUS and b['plan']==p['final_plan'],'actual fresh new bank')
            fp=meta(p['final_plan']);u=next(x for x in fp['units'] if x['id']==p['id'])
            require(len(fp['train_indices'])==10042 and len(fp['train_sessions'])==130 and u['raw_shift_factor']==u['cap_factor']==2.,'actual final fresh plan and fixed double')
        _PROGRAM_CACHE[key]=p
    return _PROGRAM_CACHE[key]


def predict_case(program,role_means_float64,counts_int64,rawaux24_float32):
    p=program_guard(program)
    if p['kind']=='fresh':return numeric('predict_case')(p['bank'],role_means_float64,counts_int64,rawaux24_float32)
    z=original_A_predictor()(p['bank'],role_means_float64,counts_int64,rawaux24_float32)
    if p['raw_shift_factor']==p['cap_factor']==1.:return z
    x=z['requested_shifts_float64']*np.float64(2.)
    return dict(action_ids=np.array([digest(dict(program=p['id'],bank=p['bank'],original_action_id=str(a),user_fixed_factors=[2.,2.])) for a in z['action_ids']],dtype='U64'),
        requested_shifts_float64=x,requested_shifts_float32=x.astype(np.float32),cap_RMS_float64=z['cap_RMS_float64']*np.float64(2.))


def selftest(output):
    before=base.sources();tests=[];rng=np.random.default_rng(2026102601)
    for name in ('space','fit_inner','refit_alltrain','pack_inner','materialize_case','predict_case'):
        f=numeric(name);require(f.__code__ is getattr(base,name).__code__,'true unchanged numeric code');tests.append('unchanged_'+name)
    B=np.linalg.qr(rng.normal(size=(384,48)))[0];state=dict(space=dict(basis=B,center=np.zeros(384)),view=dict(view='valid_shared384'),D0=.2)
    q=rng.normal(size=48).astype(np.float64)
    for valid in (True,False):
        a,cap=base.decode_main(q,state,1.,2.,1.,1.,valid);b,bc=base.decode_main(q,state,1.,2.,2.,2.,valid)
        require(b.tobytes()==(a*2.).tobytes() and bc==cap*2. and b.astype(np.float32).tobytes()==np.asarray(a*2.,np.float32).tobytes(),'float64 doubling before direct FP32, cap doubled')
    # Exercise the full unchanged live-prediction function with an isolated,
    # labelled synthetic predictor cache; no entity or release is fabricated.
    fn=numeric('predict_case');rr=dict(path='synthetic_only_no_disk_bank',sha256='f'*64);p=dict(units=[dict(id='synthetic',raw_shift_factor=2.,cap_factor=2.,selected_whole_variant='user_fixed_double_no_variant_selection')]);u=p['units'][0]
    ms=[(dict(selection=dict(alpha=1.,cap_D0=2.),physical_UID=str(k),old_slot=k),lambda v,a,f,k=k:np.full(48,.01*(k+1),np.float64)) for k in range(8)]
    fn.__globals__['_CACHE'][('live',rr['path'],rr['sha256'],digest(sources()))]=(dict(),p,state,u,ms)
    out=fn(rr,np.zeros((2,384),np.float64),np.array([3,2],np.int64),np.zeros(24,np.float32))
    require(out['requested_shifts_float64'].shape==(8,2,384) and out['requested_shifts_float32'].tobytes()==out['requested_shifts_float64'].astype(np.float32).tobytes() and out['cap_RMS_float64'].tolist()==[.8]*8,'full legacy predict thin call')
    require(math.exact_matching(np.array([[1,9],[8,2]],np.int64),['z','a'])['old_to_new']==[1,0],'integer common-member matching')
    # Actual A banks, ordinary small pooled inputs only: verify the reuse and
    # enhanced-C8 runtime branches without creating any new program/model.
    _,_,ap,_,banks=a_entities();o=data.small_arrays(meta(ap['data_proof'])['ordinary_arrays'])
    c8=banks['D','C_to_P',8];cached_keys=[]
    for descriptor in ('D_C8_original','D_C8_double'):
        rr=dict(path='selftest_cached_actual_A_'+descriptor,sha256='e'*64)
        pp=next(copy.deepcopy(v) for v in expected_families() if v['id']==descriptor);pp['bank']=c8
        key=(rr['path'],rr['sha256']);_PROGRAM_CACHE[key]=pp;cached_keys.append(key)
        for n in (0,len(o['indices_int64'])//2,len(o['indices_int64'])-1):
            args=(o['Base_pooled_float64'][n],o['token_counts_int64'][n],o['raw_aux24_float32'][n]);original=base.predict_case(c8,*args);actual=predict_case(rr,*args)
            factor=pp['raw_shift_factor'];x=original['requested_shifts_float64']*np.float64(factor)
            require(actual['requested_shifts_float64'].tobytes()==x.tobytes() and actual['requested_shifts_float32'].tobytes()==x.astype(np.float32).tobytes() and actual['cap_RMS_float64'].tobytes()==(original['cap_RMS_float64']*factor).tobytes(),'actual C8 reuse/double CPU byte control')
            if factor==1.:require(actual['action_ids'].tobytes()==original['action_ids'].tobytes(),'cached original source identity leaves action IDs exact')
        tests.append('actual_A_'+descriptor+'_three_ordinary_cases_exact')
    for k in cached_keys:_PROGRAM_CACHE.pop(k)
    require(base.sources()==before and base.context is not context and base.predict_case.__globals__ is base.__dict__,'legacy module globals untouched')
    md=metadata();require(len(md['scopes']['INNER']['indices'])==8021 and len(md['cal']['selected_indices'])==2021 and len(md['scopes']['ALLTRAIN']['indices'])==10042,'actual final split metadata')
    return result(output,dict(status='PASS_SOURCE_ONLY_FULL_SIX_FACTORY_NUMERIC_AND_FINAL_SPLIT_CONTROLS_V1',passed=True,producer=ref(__file__),source_files=sources(),tests=tests,
        actual_fit_or_native_calls=0,synthetic_predictor_thin_call=True,actual_A_bank_CPU_predictions_only=True,integer_matching=True,raw64_double_then_FP32=True,cap_double=True,legacy_globals_unchanged=True,
        final_scopes={k:dict(scenes=len(v['indices']),Q=len(v['Q']),sessions=len(v['sessions'])) for k,v in md['scopes'].items()},cal_scenes=2021))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare-catalog','prepare-reused','prepare-inner','data','space','fit-inner','pack-pilot','pack-cal','request-proof','collect','freeze','prepare-alltrain','refit-alltrain','finalize-catalog','selftest'])
    p.add_argument('--catalog',type=Path,default=CATALOG);p.add_argument('--plan',type=Path);p.add_argument('--release',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--acceptance',type=Path)
    p.add_argument('--scope',choices=['pilot32','innercal']);p.add_argument('--unit');p.add_argument('--route',choices=['R','D']);p.add_argument('--threads',type=int,default=1);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path)
    a=p.parse_args();require(1<=a.threads<=8,'bounded CPU threads')
    if a.action=='prepare-catalog':r=prepare_catalog(a.output or CATALOG)
    elif a.action=='prepare-reused':r=prepare_reused(a.catalog,a.output or PREP/'REUSED_A_PROGRAMS.json')
    elif a.action=='prepare-inner':r=prepare_inner(a.catalog,a.output or INNER_PLAN)
    elif a.action=='data':r=data_stage(a.plan,a.release)
    elif a.action=='space':r=space(a.plan,a.release,a.route)
    elif a.action=='fit-inner':r=fit_inner(a.plan,a.release,a.unit,a.threads)
    elif a.action in ('pack-pilot','pack-cal'):r=pack_inner(a.plan,a.release,a.unit,'pilot32' if a.action=='pack-pilot' else 'innercal',a.selection)
    elif a.action=='request-proof':r=request_proof(a.plan,a.release,a.scope,a.output)
    elif a.action=='collect':r=collect(a.plan,a.release,a.acceptance,a.scope,a.workers)
    elif a.action=='freeze':r=freeze(a.plan,a.release,a.selection,a.output or ANALYSIS/'FINAL_SLOT_FREEZE.json')
    elif a.action=='prepare-alltrain':r=prepare_alltrain(a.plan,a.selection,a.output or ALLTRAIN_PLAN)
    elif a.action=='refit-alltrain':r=refit_alltrain(a.plan,a.release,a.unit,a.threads)
    elif a.action=='finalize-catalog':r=finalize_catalog(a.plan,a.release,a.output or FINAL_CATALOG)
    else:r=selftest(a.output or ROOT/'selftest/v2/FACTORY_SELFTEST.json')
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
