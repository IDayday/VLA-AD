"""Full CPU target candidate from verified same-execution native actual shifts.

Writes only an independent candidate tree. No torch/Library/head/scorer import.
"""
from __future__ import annotations
import argparse,base64,concurrent.futures as cf,fcntl,hashlib,importlib.util,json,os,sqlite3,threading,time,traceback,uuid
from pathlib import Path
import numpy as np
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
BUNDLE_SHA='ed0a1bef8b27d89e7bb4eb8f32c93134204d66c42184ddecaa1831c5a7188dfc'
COUNTS={'train':85813,'test':12146,'validation':17475}
READER_PATH=CODE/'scripts/student_distillation/read_teacher_cache.py'
spec=importlib.util.spec_from_file_location('cached_actual_target_lossless_reader',READER_PATH);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)

def sha(p):return reader.sha_file(p)
def ref(p):return reader.reference(p)
def read(p):return reader.read_json(p)
def digest(v):return reader.digest(v)
def check(r,root):return reader.resolve(r,root)
def _bundle_metadata(path,private):
 path=Path(path).resolve();b=read(path)
 if not private and sha(path)!=BUNDLE_SHA:raise ValueError('The accepted new ed0 frozen teacher is required')
 if b.get('status')!='FROZEN_LOCAL_TEACHER'or b.get('teacher_origin')!='local_refit'or b.get('fit_population')!='train':raise ValueError('Frozen local Train teacher required')
 lp=check(b['library'],path.parent);config=read(lp);root=reader.resolve(b.get('artifact_root',str(lp.parent)),path.parent)
 if (config.get('route'),config.get('order'),config.get('K'),config.get('rank'),config.get('raw_shift_factor'),config.get('cap_factor'))!=('D','C_to_P',8,48,2.,2.):raise ValueError('D/C_to_P/K8/D48/double is fixed')
 with np.load(check(config['space'],root),allow_pickle=False)as z:basis=np.array(z['basis']);center=np.array(z['center']);d0=float(z['D0'])
 if basis.dtype!=np.float64 or basis.shape!=(384,48)or not np.isfinite(basis).all()or center.shape!=(384,)or not np.isfinite(center).all()or not np.isfinite(d0)or d0<=0:raise ValueError('Invalid actual D48/D0')
 error=float(np.max(np.abs(basis.T@basis-np.eye(48))))
 if error>1e-8:raise ValueError('Nonorthonormal basis')
 experts=config['experts'];caps=[float(s['cap_D0'])*float(config['cap_factor'])for s in experts]
 if len(experts)!=8 or not np.isfinite(caps).all()or min(caps)<0 or not 0<max(caps)<=4.:raise ValueError('Invalid fixed expert caps')
 for s,c in zip(experts,caps):
  check(s['parameters'],root)
  if 'runtime_state'in s:check(s['runtime_state'],root)
  if c==0 and not((s.get('Base')is True or str(s.get('method','')).lower()=='base')and float(s['alpha'])==0):raise ValueError('Only explicit Base can have cap0')
 runtime=b.get('runtime_predictor');rh=sha(check(runtime,path.parent))if runtime else None
 identity={'bundle_path':str(path),'bundle_sha256':sha(path),'library_sha256':sha(lp),'basis_sha256':config['space']['sha256'],'source_checkpoint_sha256':b['source_checkpoint_sha256'],'teacher_origin':'local_refit','runtime_predictor_sha256':rh,'orthogonality_maxabs':error,'per_slot_cap_in_d0':caps,'student_cap_in_d0':max(caps),'teacher_identity':b['teacher_identity']}
 return b,identity

def _selection(ctx):
 p=check(ctx['selection_ref'],ctx['root']);z={}
 with np.load(p,allow_pickle=False)as f:
  for k in ('tokens','indices','choice','q_points','selected_actual_role_shifts','selected_actual_RMS','selected_cap_RMS'):z[k]=np.array(f[k])
 z['tokens']=z['tokens'].astype(str).tolist();n=len(z['tokens']);q=z['q_points'];a=z['selected_actual_role_shifts']
 if len(set(z['tokens']))!=n or q.dtype!=np.float64 or q.shape!=(n,9)or not np.isfinite(q).all()or(q<0).any()or(q>100+1e-8).any():raise ValueError('Invalid selected score coverage/units')
 if z['indices'].dtype!=np.int64 or z['indices'].shape!=(n,)or len(set(z['indices'].tolist()))!=n or not np.array_equal(q.argmax(1),z['choice']):raise ValueError('Selected IDs/point argmax differ')
 if a.dtype!=np.float32 or a.shape!=(n,2,384)or not np.isfinite(a).all()or not np.array_equal(a[:,0],a[:,1]):raise ValueError('Actual FP32 shifts must have identical roles')
 for k in ('selected_actual_RMS','selected_cap_RMS'):
  if z[k].shape!=(n,)or not np.isfinite(z[k]).all()or(z[k]<0).any():raise ValueError('Invalid selected RMS/caps')
 z['gain_points']=q[np.arange(n),z['choice']]-q[:,0];z['gain1_keep']=z['gain_points']>1.;z['by_token']={t:i for i,t in enumerate(z['tokens'])};return z

def _fingerprint(p):
 s=Path(p).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def build(paths,population,out,*,workers=8,private_fixture=False):
 if population not in COUNTS or type(workers)is not int or not 1<=workers<=8:raise ValueError('Fixed population and1..8 CPUworkers required')
 out=Path(out).resolve();values=read(paths)if not isinstance(paths,dict)else dict(paths);values=values.get('paths',values)
 if not private_fixture and out!=RESULTS/'targets_verified_native_cache_stage'/population:raise ValueError('Formal candidate output must remain in targets_verified_native_cache_stage/population')
 feature_path=Path(values['feature_manifest']).resolve();fm=read(feature_path)
 if private_fixture and not fm.get('synthetic_only'):raise ValueError('Private override requires explicit synthetic feature fixture')
 if int(fm['populations'][population]['count'])!=COUNTS[population]and not private_fixture:raise ValueError('Cannot shorten a formal population')
 b,teacher=_bundle_metadata(values['teacher_bundle'],private_fixture)
 if fm.get('source_checkpoint_sha256')!=teacher['source_checkpoint_sha256']:raise ValueError('Teacher/Base feature checkpoint differs')
 ctx=reader.context(values,population,allow_synthetic=private_fixture,verify_selection=True)
 folder,index,lookup,marker_path=reader.index_files(ctx);marker=read(marker_path)
 n=int(ctx['feature_spec']['count'])
 if n!=COUNTS[population]and not private_fixture:raise ValueError('Cannot shorten a formal population')
 if marker.get('status')!='COMPLETE'or marker.get('population')!=population or marker.get('scenes')!=n:raise ValueError('Closed full repaired-cache index required')
 expected={'teacher_bundle':ref(values['teacher_bundle']),'feature_manifest':ref(feature_path),'feature_population_sha256':digest(ctx['feature_spec']),'selection':ctx['selection_ref']}
 if any(marker['inputs'].get(k)!=v for k,v in expected.items()):raise ValueError('Closed index identity differs from current inputs')
 # Verify both immutable index artifacts once; each scene then verifies its
 # indexed record hash and its source NPZ/full H0/layout/reconstruction bytes.
 source_refs={'repaired_cache_complete':ref(marker_path),'repaired_cache_jsonl':marker['artifacts']['jsonl'],'repaired_cache_sqlite':marker['artifacts']['sqlite_lookup'],'teacher_bundle':ref(values['teacher_bundle']),'feature_manifest':ref(feature_path),'selection':ctx['selection_ref'],'population_evaluation':marker['inputs']['population_evaluation'],'native_complete':marker['inputs']['native_complete']}
 for r in source_refs.values():check(r,folder)
 snapshots={r['path']:_fingerprint(r['path'])for r in source_refs.values()}
 feature_index=check(ctx['feature_spec']['index'],feature_path.parent);rows=[json.loads(x)for x in feature_index.read_text().splitlines()if x]
 selection=_selection(ctx)
 if len(rows)!=n or len({r['token']for r in rows})!=n or len({r['global_index']for r in rows})!=n or {r['token']for r in rows}!=set(selection['by_token']):raise ValueError('Exact feature/selection coverage required')
 for r in rows:
  i=selection['by_token'][r['token']]
  if r['population']!=population or int(r['global_index'])!=int(selection['indices'][i]):raise ValueError('Feature/globalindex/selection join differs')
 identity={'teacher':teacher,'selection_sha256':ctx['selection_ref']['sha256'],'feature_manifest_sha256':ctx['features_sha'],'feature_population_identity_sha256':digest(ctx['feature_spec']),'target_versions':['full','gain1'],'population':population,'diagnostic_only':population!='train','execution_device_type':'cpu_verified_native_cache'}
 identity_sha=digest(identity);out.parent.mkdir(parents=True,exist_ok=True)
 lock=(out.parent/('.'+population+'.CANDIDATE_BUILDER.lock')).open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if out.exists():
  m=read(out/'MANIFEST.json');done=read(out/'COMPLETE.json')
  if (m.get('schema')!='nrkd_compact_targets_v1'or m.get('teacher_origin')!='local_refit'or m.get('teacher_identity')!=teacher or m.get('immutable_population')!=population or set(m.get('populations',{}))!={population}):raise ValueError('Existing candidate manifest schema/population differs')
  saved=m['populations'][population]
  if (saved.get('identity')!=identity or saved.get('identity_sha256')!=identity_sha or saved.get('count')!=n or saved.get('diagnostic_only')!=(population!='train') or set(saved.get('versions',{}))!={'full','gain1'} or done.get('status')!='COMPLETE'or done.get('identity_sha256')!=identity_sha or done.get('population_spec')!=saved):raise ValueError('Existing candidate completion/identity differs')
  check(saved['index'],out)
  for version in saved['versions'].values():
   array=np.load(check(version['shifts'],out),mmap_mode='r',allow_pickle=False)
   if array.dtype!=np.float32 or array.shape!=(n,384):raise ValueError('Existing candidate shift layout differs')
  construction=read(check(saved['construction_receipt'],out))
  counts={name:n for name in ['H0_SHA','full_teacher_SHA','padding_SHA','RMS_cap','source_SCORES_SHA','actual_role_shift','layout_SHA','token_global_index_join']}
  if (construction.get('schema')!='nrkd_verified_native_cache_target_construction_v1'or construction.get('status')!='COMPLETE'or construction.get('population')!=population or construction.get('scenes')!=n or construction.get('scenes_verified')!=n or construction.get('execution_device_type')!='cpu_verified_native_cache'or construction.get('private_synthetic_fixture')is not private_fixture or construction.get('per_row_check_counts')!=counts):raise ValueError('Existing construction completion/checkcounts differ')
  if (construction.get('source_refs')!=source_refs or construction.get('cache_index')!={'complete':ref(marker_path),'inputs':marker['inputs'],'artifacts':marker['artifacts']} or construction.get('teacher_identity')!=teacher or construction.get('feature_manifest_sha256')!=ctx['features_sha'] or construction.get('feature_population_sha256')!=digest(ctx['feature_spec']) or construction.get('selection_sha256')!=ctx['selection_ref']['sha256'] or construction.get('builder_source',{}).get('sha256')!=sha(__file__) or construction.get('reader_source')!=ref(READER_PATH)):raise ValueError('Existing construction source identity differs')
  if (construction.get('ordered_tokens_sha256')!=digest([r['token']for r in rows])or construction.get('ordered_global_indices_sha256')!=digest([int(r['global_index'])for r in rows])or construction.get('outputs')!={'index':saved['index'],'versions':saved['versions']}):raise ValueError('Existing construction output coverage differs')
  check(construction['verified_rows'],out)
  return m
 stage=out.parent/('.'+population+'.cpu_cache_partial.'+uuid.uuid4().hex);stage.mkdir();start=time.monotonic();readers=[];readers_lock=threading.Lock();local=threading.local();seen=np.zeros(n,bool)
 database=None
 try:
  shifts={v:np.lib.format.open_memmap(stage/(v+'_shifts.npy'),mode='w+',dtype=np.float32,shape=(n,384))for v in ['full','gain1']}
  database=sqlite3.connect(stage/'VERIFIED_ROWS.sqlite3');database.execute('CREATE TABLE rows(idx INTEGER PRIMARY KEY, record TEXT NOT NULL, proof TEXT NOT NULL)')
  zero={'full':0,'gain1':0};completed=0
  def work(item):
   k,row=item
   if not hasattr(local,'cache'):
    local.cache=reader.TeacherCacheReader(values,population,verify_index=False,allow_synthetic=private_fixture)
    with readers_lock:readers.append(local.cache)
   scene=local.cache.read(row['token'],'full');m=scene['metadata'];i=selection['by_token'][row['token']];choice=int(selection['choice'][i]);gain=float(selection['gain_points'][i]);keep=bool(selection['gain1_keep'][i]);delta=scene['actual_shift']
   if m['population']!=population or int(m['global_index'])!=int(row['global_index'])or str(m['log_id'])!=str(row['log_id'])or m['original_full_choice']!=choice or m['gain_points_fp64']!=gain or m['gain1_keep']!=keep:raise ValueError('Cached teacher row differs from selected target metadata')
   if scene['actual_role_shifts'].tobytes()!=selection['selected_actual_role_shifts'][i].tobytes()or delta.tobytes()!=selection['selected_actual_role_shifts'][i,0].tobytes():raise ValueError('Actual scored shift bytes differ')
   if m['Base_immutable_mmap']['Base_interface_sha256']!=row['actual_Base_interface_sha256']or m['Base_immutable_mmap']['token_layout_sha256']!=row['token_layout_sha256']:raise ValueError('Frozen Base/layout differs')
   if not np.isclose(m['actual_RMS'],selection['selected_actual_RMS'][i],rtol=1e-10,atol=1e-12)or m['cap_RMS']!=float(selection['selected_cap_RMS'][i]):raise ValueError('Actual full-selected RMS/cap differs')
   # Only one full read is required. Gain1 is either these exactsame teacher
   # bytes/shift, or literal Base/zero; no second branch selection or API call.
   gain_delta=delta.copy()if keep else np.zeros(384,np.float32)
   record={key:row[key]for key in ('token','log_id','global_index','population','actual_Base_interface_sha256','token_layout_sha256')}
   record.update(original_choice=choice,gain_points=gain,full_is_zero=bool(not np.any(delta)),gain1_is_zero=bool(not keep or not np.any(delta)),gain1_keep=keep,actual_target_RMS=float(m['actual_RMS']),cap_RMS=float(m['cap_RMS']),base_mean_points=float(selection['q_points'][i,0]),full_teacher_mean_points=float(selection['q_points'][i,choice]),gain1_teacher_mean_points=float(selection['q_points'][i,choice if keep else 0]),source_checkpoint_sha256=teacher['source_checkpoint_sha256'],library_sha256=teacher['library_sha256'],basis_sha256=teacher['basis_sha256'])
   proof={'feature_row':k,'token':row['token'],'global_index':row['global_index'],'full_shift_sha256':reader.sha_array(delta),'gain1_shift_sha256':reader.sha_array(gain_delta),'teacher_interface_sha256':m['teacher_interface_sha256'],'Base_interface_sha256':row['actual_Base_interface_sha256'],'padding_sha256':m['padding_bytes_sha256'],'token_layout_sha256':row['token_layout_sha256'],'source_scores':m['source_scores'],'source_result':m['source_result'],'actual_RMS_CPU':m['actual_RMS'],'actual_RMS_native_scored':float(selection['selected_actual_RMS'][i]),'cap_RMS':m['cap_RMS'],'full_same_execution_FP32_SHA':True,'gain1_uses_exact_full_or_Base':True}
   return k,delta.copy(),gain_delta,record,proof
  # At most2*workers pending tasks and at mostworkers full scene objects exist.
  with cf.ThreadPoolExecutor(max_workers=workers)as pool:
   items=iter(enumerate(rows));pending=set()
   for _ in range(min(n,2*workers)):
    try:pending.add(pool.submit(work,next(items)))
    except StopIteration:break
   while pending:
    ready,pending=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
    for future in ready:
     k,full,gain,record,proof=future.result()
     if seen[k]:raise ValueError('Duplicate feature row')
     seen[k]=True;shifts['full'][k]=full;shifts['gain1'][k]=gain;zero['full']+=record['full_is_zero'];zero['gain1']+=record['gain1_is_zero']
     database.execute('INSERT INTO rows VALUES(?,?,?)',(k,json.dumps(record,sort_keys=True,allow_nan=False),json.dumps(proof,sort_keys=True,allow_nan=False)));completed+=1
     if completed%512==0:database.commit();print(json.dumps({'private_candidate_completed':completed,'population':population,'total':n,'seconds':time.monotonic()-start}),flush=True)
     try:pending.add(pool.submit(work,next(items)))
     except StopIteration:pass
  database.commit()
  if completed!=n or not seen.all():raise ValueError('No success for partial coverage')
  for x in shifts.values():x.flush()
  shifts.clear()
  with (stage/'INDEX.jsonl').open('w')as output,(stage/'VERIFIED_ROWS.jsonl').open('w')as proofs:
   for expected_row,(k,record,proof)in enumerate(database.execute('SELECT idx,record,proof FROM rows ORDER BY idx')):
    if k!=expected_row:raise ValueError('Output feature ordering changed')
    output.write(record+'\n');proofs.write(proof+'\n')
   output.flush();os.fsync(output.fileno());proofs.flush();os.fsync(proofs.fileno())
  database.close();database=None
  # All closed source identities must still be unchanged at publication.
  for r in source_refs.values():
   if _fingerprint(r['path'])!=snapshots[r['path']]or sha(r['path'])!=r['sha256']:raise ValueError('Closed construction source changed during build')
  q=selection['q_points'];selected=q[np.arange(n),selection['choice']];gainq=np.where(selection['gain1_keep'],selected,q[:,0])
  stats={'count':n,'original_choice_counts':np.bincount(selection['choice'],minlength=9).tolist(),'gain1_repair_count':int(selection['gain1_keep'].sum()),'base_points':float(q[:,0].mean()),'full_points':float(selected.mean()),'gain1_points':float(gainq.mean()),'identity':'NEW_LOCAL_FROZEN_TEACHER','historical_counts_are_not_constraints':True}
  versions={v:{'shifts':{'path':v+'_shifts.npy','sha256':sha(stage/(v+'_shifts.npy'))},'zero_targets':int(zero[v])}for v in ['full','gain1']}
  population_spec={'count':n,'diagnostic_only':population!='train','identity':identity,'identity_sha256':identity_sha,'index':{'path':'INDEX.jsonl','sha256':sha(stage/'INDEX.jsonl')},'versions':versions,'statistics':stats}
  registry_bytes=base64.b64decode(ctx['registry_snapshot_b64']);reader.atomic_json(stage/'SELECTION_REGISTRY_USED.json',{'registry':json.loads(registry_bytes),'source_registry_sha256':hashlib.sha256(registry_bytes).hexdigest()})
  construction={'schema':'nrkd_verified_native_cache_target_construction_v1','status':'COMPLETE','execution_device_type':'cpu_verified_native_cache','original_teacher_execution':'same saved nativeCUDA Library.targets output used for original9branchS0-S4 scoring; no recomputation','source_refs':source_refs,'feature_population_sha256':digest(ctx['feature_spec']),'population':population,'scenes':n,'scenes_verified':n,'workers':workers,'maximum_pending_tasks':2*workers,'one_full_reader_verification_per_scene':True,'target_source':'saved_scored_actual_role_shifts','full_row_checks':['BaseH0 SHA','token/mask/position layout SHA','sourceSCORES NPZ SHA','identical actualFP32role shifts','fullInterface SHA equal originalLibrary output','padding SHA','actualRMS/cap and selectedshift match','same saved paired trajectories/seven scores'],'verified_rows':{'path':'VERIFIED_ROWS.jsonl','sha256':sha(stage/'VERIFIED_ROWS.jsonl')},'builder_source':ref(__file__),'reader_source':ref(READER_PATH),'private_synthetic_fixture':private_fixture,'elapsed_seconds':time.monotonic()-start,'actual_target_RMS_definition':'Full-selected reconstructedH_T minus H0 valid-token FP64RMS, validated against native recordedRMS with rtol1e-10/atol1e-12; retained as full-selected metadata when gain1zero.','GPU_Library_head_scorer_calls':0}
  construction.update(cache_index={'complete':ref(marker_path),'inputs':marker['inputs'],'artifacts':marker['artifacts']}, teacher_identity=teacher,feature_manifest_sha256=ctx['features_sha'],selection_sha256=ctx['selection_ref']['sha256'],per_row_check_counts={name:n for name in ['H0_SHA','full_teacher_SHA','padding_SHA','RMS_cap','source_SCORES_SHA','actual_role_shift','layout_SHA','token_global_index_join']},ordered_tokens_sha256=digest([r['token']for r in rows]),ordered_global_indices_sha256=digest([int(r['global_index'])for r in rows]),outputs={'index':population_spec['index'],'versions':versions})
  reader.atomic_json(stage/'CONSTRUCTION.json',construction)
  population_spec['construction_receipt']={'path':'CONSTRUCTION.json','sha256':sha(stage/'CONSTRUCTION.json')}
  reader.atomic_json(stage/'COMPLETE.json',{'status':'COMPLETE','identity_sha256':identity_sha,'population_spec':population_spec})
  manifest={'schema':'nrkd_compact_targets_v1','teacher_origin':'local_refit','teacher_identity':teacher,'immutable_population':population,'populations':{population:population_spec}}
  reader.atomic_json(stage/'MANIFEST.json',manifest)
  if out.exists():raise ValueError('Refusing to replace any existing candidate')
  os.rename(stage,out)
  print(json.dumps({'status':'CANDIDATE_FULL_POPULATION_COMPLETE','manifest':ref(out/'MANIFEST.json'),'construction':ref(out/'CONSTRUCTION.json'),'seconds':time.monotonic()-start}),flush=True);return manifest
 except BaseException:
  if database is not None:database.close()
  reader.atomic_json(stage/'FAILURE.json',{'status':'INCOMPLETE_PRIVATE_CANDIDATE','traceback':traceback.format_exc(),'no_canonical_output_changed':True})
  raise
 finally:
  # Reader SQLite handles belong to their exited worker threads; connection
  # objects disappear with this finiteprocess. Never close across threads.
  lock.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths',required=True);p.add_argument('--population',choices=COUNTS,required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,choices=range(1,9),default=8);a=p.parse_args();build(a.paths,a.population,a.out,workers=a.workers)
if __name__=='__main__':main()
