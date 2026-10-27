"""Reapply original Base-S0 screening to already measured local Train scores."""
from pathlib import Path
import csv,gzip,json,hashlib,time,datetime
import numpy as np
NEW=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026');CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026');OUT=Path(__file__).parent;SRC=NEW/'source_snapshot/ReCogDrive';POOL=Path('/mnt/project/recogdrive-dc8-local-refit-20261026/inputs/LOCAL_TRAIN_BASE_POOL.npz');start=time.time()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def put(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
with gzip.open(CODE/'manifests/train.csv.gz','rt')as f:train=list(csv.DictReader(f))
by={r['token']:r for r in train};assert len(by)==85813
with np.load(POOL,allow_pickle=False)as z:
 tokens=z['tokens'].astype(str).tolist();ids=np.array(z['indices_int64']);sessions=z['sessions'].astype(str).tolist();scores=np.array(z['Base_quality_float64'])
assert len(set(tokens))==len(set(ids.tolist()))==85813 and set(tokens)==set(by)
assert scores.dtype==np.float64 and scores.shape==(85813,5,7) and np.isfinite(scores).all()
for i,t,s in zip(ids,tokens,sessions):assert int(by[t]['index'])==int(i)and by[t]['session']==s
old_path=SRC/'1022-gt_assisted_representation_optimization/manifests/SCENES.json';old=json.loads(old_path.read_text());old=[r for r in old if r['subset']=='exploration'];oldids={int(r['global_index'])for r in old};assert len(oldids)==1850 and all(float(r['baseline_pdms_100'])<70 for r in old)
source_gt=SRC/'1022-gt_assisted_representation_optimization/code/g5_prepare.py';source_cv=SRC/'1024-representation_distillation_diagnosis_batch5/stage2/offline_library/code/ol_session_cv_metadata_v1.py';source_pack=source_cv.with_name('ol_l1_data_pack_v1.py')
records=[]
for pos in np.flatnonzero(scores[:,0,0]<.7):
 token=tokens[pos];r=by[token];records.append(dict(r,index=int(ids[pos]),global_index=int(ids[pos]),token=token,session=sessions[pos],population='train',subset='exploration',quality='failure',audit_category='failure',Base_S0=float(scores[pos,0,0]),Base_S0_points=float(100*scores[pos,0,0]),Base_S0_S4_PDMS=scores[pos,:,0].tolist(),old_PPU_GT_candidate=int(ids[pos])in oldids))
records.sort(key=lambda r:r['global_index']);newids={r['global_index']for r in records};candidate_path=OUT/'LOCAL_CANDIDATES.json';poolref=ref(POOL);rules={'failure':'Base_quality_float64[:,0,0] < 0.7','mid':'0.7 <= Base_S0 <= 0.9','high':'Base_S0 > 0.9','population':'original85813Train/exploration only; noValidation/Test','source_gt':{**ref(source_gt),'line':4},'source_categories':{**ref(source_cv),'line':19},'source_data_pack':{**ref(source_pack),'line':56}}
put(candidate_path,{'schema':'nrkd_local_GT_candidate_screen_v1','status':'LOCAL_S0_CANDIDATES_RECOMPUTED','count':len(records),'rule':rules,'local_Base_pool':poolref,'source_train_manifest':ref(CODE/'manifests/train.csv.gz'),'candidates':records,'production_queues_modified':False,'scoring_or_GPU_calls':0})
for name,selection in [('INTERSECTION',newids&oldids),('OLD_ONLY',oldids-newids),('NEW_ONLY',newids-oldids)]:
 data=[]
 posbyid={int(x):j for j,x in enumerate(ids)}
 for i in sorted(selection):
  j=posbyid[i];data.append({'global_index':i,'index':i,'token':tokens[j],'session':sessions[j],'local_Base_S0':float(scores[j,0,0]),'local_Base_S0_S4_PDMS':scores[j,:,0].tolist()})
 put(OUT/(name+'.json'),{'schema':'nrkd_local_GT_candidate_delta_v1','group':name,'count':len(data),'rule':rules['failure'],'records':data})
with (OUT/'LOCAL_CANDIDATES.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=['global_index','token','session','Base_S0','Base_S0_points','old_PPU_GT_candidate']);w.writeheader();w.writerows({k:r[k]for k in w.fieldnames}for r in records)
summary={'schema':'nrkd_local_candidate_recheck_v1','status':'EXACT_ORIGINAL_RULE_REAPPLIED_LOCAL','created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rule':rules,'original_PPU_GT_Train_count':1850,'local_GT_Train_count':len(records),'intersection':len(newids&oldids),'old_only':len(oldids-newids),'new_only':len(newids-oldids),'local_Train_strata':{'failure':int((scores[:,0,0]<.7).sum()),'mid':int(((scores[:,0,0]>=.7)&(scores[:,0,0]<=.9)).sum()),'high':int((scores[:,0,0]>.9).sum())},'wrong_rules_for_comparison_only':{'any_five_below70':int((scores[:,:,0]<.7).any(1).sum()),'five_mean_below70':int((scores[:,:,0].mean(1)<.7).sum())},'candidate_list':ref(candidate_path),'intersection_list':ref(OUT/'INTERSECTION.json'),'old_only_list':ref(OUT/'OLD_ONLY.json'),'new_only_list':ref(OUT/'NEW_ONLY.json'),'local_Base_pool':poolref,'original_PPU_GT_Train_manifest':ref(old_path),'elapsed_seconds':time.time()-start,'no_production_edits_or_worker_changes':True,'GPU_scorer_or_status_poll_calls':0,'derived_reference_panel_followup':'pending separate file; candidate list immutable'}
put(OUT/'LOCAL_CANDIDATE_RECHECK.json',summary);print(json.dumps(summary))
