"""Original stratum/hash/panel rules on localS0, preserving registered104/26sessions."""
from pathlib import Path
from collections import Counter
import ast,csv,gzip,json,hashlib,time
import numpy as np
NEW=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026');CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026');OUT=Path(__file__).parent;SRC=NEW/'source_snapshot/ReCogDrive';OL=SRC/'1024-representation_distillation_diagnosis_batch5/stage2/offline_library';start=time.time()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def put(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def bind(path,names,namespace):
 tree=ast.parse(path.read_text())
 for name in names:
  node=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace
cvpath=OL/'preparation/session_cv_metadata_v1/SPLITS.json';cv=json.loads(cvpath.read_text());source_cv=OL/'code/ol_session_cv_metadata_v1.py';source_panel=OL/'code/ol_cv_shared48_cpu_v1.py';ns=bind(source_cv,['key','category'],{'hashlib':hashlib});panelns=bind(source_panel,['hash_panel','pilot_panel'],{'hashlib':hashlib})
with gzip.open(CODE/'manifests/train.csv.gz','rt')as f:train=list(csv.DictReader(f))
with np.load('/mnt/project/recogdrive-dc8-local-refit-20261026/inputs/LOCAL_TRAIN_BASE_POOL.npz',allow_pickle=False)as z:tokens=z['tokens'].astype(str).tolist();scores=np.array(z['Base_quality_float64']);ids=np.array(z['indices_int64']);sessions=z['sessions'].astype(str).tolist()
local={t:dict(index=int(i),global_index=int(i),token=t,session=s,Base_scores=q[:,0].tolist())for t,i,s,q in zip(tokens,ids,sessions,scores)}
for r in local.values():r['audit_category']=ns['category'](r)
assert len(local)==len(train)==85813 and set(local)=={r['token']for r in train}
failures=[r for r in local.values()if r['audit_category']=='failure'];references=[]
for group,count in [('mid',2048),('high',6144)]:
 available=sorted([r for r in local.values()if r['audit_category']==group],key=lambda r:ns['key']('reference8192|'+group,r['token']));assert len(available)>=count;references.extend(available[:count])
selected=sorted(failures+references,key=lambda r:r['global_index']);rows={r['index']:r for r in selected};oldrows={r['index']:r for r in cv['selected']};oldset=set(oldrows);newset=set(rows);new_gt={r['index']for r in failures};old_gt={r['index']for r in cv['selected']if r['historical_Base_S0_category']=='failure'};newonly=new_gt-old_gt
fit_sessions=set(cv['final_locked_family_inner_fit']['sessions']);cal_sessions=set(cv['final_locked_family_inner_calibration']['sessions']);assert len(fit_sessions)==104 and len(cal_sessions)==26 and not fit_sessions&cal_sessions;assert fit_sessions|cal_sessions=={r['session']for r in local.values()}
def describe(ss):
 ss=set(ss);population=Counter(r['audit_category']for r in local.values()if r['session']in ss);examples=[r for r in selected if r['session']in ss];sample=Counter(r['audit_category']for r in examples);n=sum(population.values());weights={g:population[g]/(n*sample[g])for g in population};assert abs(sum(weights[r['audit_category']]for r in examples)-1)<1e-12
 return {'sessions':sorted(ss),'full_population_counts':dict(population),'full_population_scenes':n,'selected_indices':[r['global_index']for r in examples],'selected_counts':dict(sample),'conditional_natural_mean_weight_per_sample':weights}
fit,cal=describe(fit_sessions),describe(cal_sessions);panel=panelns['pilot_panel'](0,cal,rows);assert len(panel)==32 and Counter(r['category']for r in panel)=={'failure':16,'mid':4,'high':12};assert not {r['session']for r in panel}&fit_sessions
pops={}
for pop in ['train','validation','test']:
 with gzip.open(CODE/'manifests'/f'{pop}.csv.gz','rt')as f:pops[pop]=list(csv.DictReader(f))
sets={p:{r['token']for r in rs}for p,rs in pops.items()};assert not sets['train']&sets['validation']and not sets['train']&sets['test']and not sets['validation']&sets['test'];assert new_gt<={int(r['index'])for r in pops['train']};assert not {r['session']for r in pops['train']}&{r['session']for r in pops['validation']}
with np.load(NEW/'fitting/ORDINARY10042.npz',allow_pickle=False)as z:
 q=np.array(z['Base_quality_float64']);oi=z['indices_int64'];where={int(v):j for j,v in enumerate(ids)};assert np.array_equal(q,scores[[where[int(i)]for i in oi]])
transitions=Counter((oldrows[i]['historical_Base_S0_category'],rows2['audit_category'])for i,rows2 in ((i,local[next(r['token']for r in cv['selected']if r['index']==i)])for i in oldset))
# Output metadata only; no fitting/payload/production plan is changed.
put(OUT/'LOCAL_ORDINARY_PANEL.json',{'schema':'nrkd_local_ordinary_stratified_panel_v1','count':len(selected),'population_counts':dict(Counter(r['audit_category']for r in local.values())),'selected_counts':dict(Counter(r['audit_category']for r in selected)),'reference_counts':{'mid':2048,'high':6144},'selected':selected,'source_category_and_hash':ref(source_cv),'original_10042_count_not_forced':True})
put(OUT/'LOCAL_FINAL_SPLIT.json',{'schema':'nrkd_local_reclassified_fixed_sessions_v1','source_original_session_split':ref(cvpath),'session_membership_preserved':True,'original_PPU_strata_and_weights_reused':False,'fit':fit,'cal':cal,'no_teacher_outcome_used':True})
put(OUT/'LOCAL_PILOT32.json',{'schema':'nrkd_local_D_C8_pilot_panel_v1','panel':panel,'panel_count':32,'group_quota':{'failure':16,'mid':4,'high':12},'source':ref(source_panel),'scope':'final_locked_family_inner_calibration; samefold0hash namespace ascurrentfit_library.prepare','session_round_robin':True,'weights_recomputed_from_local_full_calibration_population_strata':True})
report={'schema':'nrkd_local_teacher_panel_recheck_v1','status':'EXACT_LOCAL_CATEGORY_HASH_CONTROL_PANEL_REAPPLIED','original_ordinary_count':10042,'local_ordinary_count':len(selected),'construction':{'all_localS0_failures':len(failures),'mid_hash_controls':2048,'high_hash_controls':6144,'category':'failureS0<.7,mid.7<=S0<=.9,highS0>.9','reference_priority':'sha256(OL_SESSION_CV|2026102531|reference8192|{category}|{token})','selected_order':'global_index ascending','mean5_or_any5_used':False},'new273_GT_candidates_in_original10042':len(newonly&oldset),'new273_GT_candidates_outside_original10042':len(newonly-oldset),'local_GT1689_outside_original10042':len(new_gt-oldset),'ordinary_panel_intersection':len(newset&oldset),'ordinary_old_only':len(oldset-newset),'ordinary_new_only':len(newset-oldset),'fixed_sessions':{'fit':104,'cal':26,'membership_preserved':True},'derived_fit_count':len(fit['selected_indices']),'derived_cal_count':len(cal['selected_indices']),'fit_selected_counts':fit['selected_counts'],'cal_selected_counts':cal['selected_counts'],'fit_full_population_counts':fit['full_population_counts'],'cal_full_population_counts':cal['full_population_counts'],'historical10042_category_transition_counts':{'|'.join(k):v for k,v in transitions.items()},'pilot32':{'failure':16,'mid':4,'high':12,'natural_weights_recomputed':True},'isolation':{'Train_only_selected':True,'Validation_and_Test_tokens_excluded':True,'Train_Validation_sessions_disjoint':True,'Base_scores_for_Validation_or_Test_not_opened':True},'original_ORDINARY10042_all5quality_exact_full_local_pool':True,'artifacts':{n:ref(OUT/n)for n in ['LOCAL_ORDINARY_PANEL.json','LOCAL_FINAL_SPLIT.json','LOCAL_PILOT32.json']},'sources':{'session_metadata':ref(source_cv),'pilot_function':ref(source_panel),'data_pack':ref(OL/'code/ol_l1_data_pack_v1.py'),'source_original_split':ref(cvpath)},'production_files_or_workers_changed':False,'GPU_scorer_or_status_poll_calls':0,'elapsed_seconds':time.time()-start}
put(OUT/'LOCAL_PANEL_RECHECK.json',report);print(json.dumps(report))
