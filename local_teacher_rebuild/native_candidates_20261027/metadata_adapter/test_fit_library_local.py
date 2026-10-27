"""CPU-only transport fixtures; synthetic repair scores never become formal data."""
import importlib.util, json, pickle
from pathlib import Path
import numpy as np
import pytest
PATH=Path(__file__).with_name('fit_library_local.py')
spec=importlib.util.spec_from_file_location('fit_library_local_tested',PATH);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def test_verified_local_metadata_and_fixed_session_weights():
 m=adapter.load_metadata();assert len(m['rows'])==9881 and len(m['candidates']['candidates'])==1689
 assert len(m['fit']['selected_indices'])==7878 and len(m['cal']['selected_indices'])==2003
 assert len(m['fit']['sessions'])==104 and len(m['cal']['sessions'])==26
 assert not set(m['fit']['sessions'])&set(m['cal']['sessions'])
 assert not set(m['fit']['selected_indices'])&set(m['cal']['selected_indices'])
 assert all(x['population']=='train' for x in m['candidates']['candidates'])
 assert sum(x['natural_mean_weight']for x in m['panel'])==pytest.approx(1.)
 assert {x['session']for x in m['panel']}<=set(m['cal']['sessions'])


def test_original_numeric_function_code_and_fork_pickle_unchanged(tmp_path):
 before=adapter.sha(adapter.ORIGINAL);module=adapter.cloned_fitter(tmp_path)
 for name in ['build_space','fit_payload','mlp_fit','_fit_inner_cluster','cpu_cluster_map','pack','collect','refit','freeze']:
  assert getattr(module,name).__code__ is getattr(module._original_source,name).__code__
 assert pickle.loads(pickle.dumps(module._fit_inner_cluster))is module._fit_inner_cluster
 assert module.ROOT==tmp_path and module._original_source.ROOT==adapter.OLD
 assert adapter.sha(adapter.ORIGINAL)==before


def test_actual_plan_and_missing_exports_block_dynamic1689(tmp_path):
 plan=adapter.read(adapter.DEFAULT_ROOT/'fitting/PLAN.json');assert plan['GT_candidate_count']==1689 and len(plan['rows'])==9881
 assert plan['family']=={'route':'D','order':'C_to_P','K':8,'rank':48,'view':'valid_shared384'}
 assert plan['whole_variant']=={'raw_shift_factor':2.,'cap_factor':2.}
 assert plan['isolation']['train_validation_session_disjoint'] and plan['isolation']['counts']['train']==85813
 plan=dict(plan,root=str(tmp_path),gt_root=str(tmp_path/'native_gt'))
 module=adapter.cloned_fitter(tmp_path)
 with pytest.raises(RuntimeError,match='1689/1689 missing'):module.load_pairs(plan)
 pending=adapter.read(tmp_path/'fitting/PAIR_EXPORTS_PENDING.json');assert pending['required']==pending['missing_count']==1689 and pending['present']==0


@pytest.fixture
def pair(tmp_path):
 module=adapter.cloned_fitter(tmp_path);base=np.arange(5*384,dtype=np.float32).reshape(5,384)/1000
 image=np.array([0,1,1,0,0],bool);text=np.array([0,0,0,1,0],bool);teacher=base.copy();teacher[image|text]+=np.float32(.03)
 bq=np.ones((5,7),np.float64);bq[:,0]=.6;tq=np.ones((3,7),np.float64);tq[:,0]=.95
 f=np.stack([base[m].astype(np.float64).mean(0)for m in(image,text)]);counts=np.array([image.sum(),text.sum()],np.int64)
 ordinary=tmp_path/'ordinary.npz';np.savez(ordinary,indices_int64=np.array([12]),Base_pooled_float64=f[None],token_counts_int64=counts[None],Base_quality_float64=bq[None],Base_interface_sha256=np.array([__import__('hashlib').sha256(base.tobytes()).hexdigest()]))
 token='PRIVATE_SYNTHETIC_ONLY';gt=tmp_path/'native_gt';folder=gt/'exports/post'/token;folder.mkdir(parents=True)
 numeric=folder/'TEACHER.npz';values=dict(Base_interface=base,teacher_interface=teacher,image_mask=image,text_mask=text,Base_quality=bq[:3],quality=tq);np.savez(numeric,**values)
 marker=folder/'COMPLETE.json';meta={'status':'PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR','token':token,'global_index':12,'population':'train','engineering_fixture_only':True,'arrays':adapter.reference(numeric)};put(marker,meta)
 plan={'root':str(tmp_path),'gt_root':str(gt),'GT_candidate_count':1,'ordinary':adapter.reference(ordinary),'rows':[{'index':12,'token':token,'session':'PRIVATE_LOG','audit_category':'failure'}],'full_GT_search_required':True}
 return module,plan,marker,numeric,values,meta


def test_full_pair_requires_full_search_and_preserves_canonical_bytes(pair):
 module,plan,marker,numeric,values,meta=pair
 with pytest.raises(RuntimeError,match='mandatory fullGTsearch'):module.load_pairs(plan)
 gt=Path(plan['gt_root']);put(gt/'SEARCH_COMPLETE.json',{'status':'FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE','full_original_finite_search_complete':False});put(gt/'GT_POST_COMPLETE.json',{'train_failures':1,'actual_exports':1})
 with pytest.raises(ValueError,match='full dynamiccandidate'):module.load_pairs(plan)
 put(gt/'SEARCH_COMPLETE.json',{'status':'FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE','full_original_finite_search_complete':True})
 records,pairs=module.load_pairs(plan);assert len(records)==1 and records[0]['qualified'] and pairs[12][0].dtype==np.float64
 with numeric.open('ab')as f:f.write(b'corruption')
 with pytest.raises(ValueError,match='arrays SHA'):module.load_pairs(plan)


def test_padding_or_changed_base_rejected_even_with_new_payload_checksum(pair):
 module,plan,marker,numeric,values,meta=pair
 values['teacher_interface'][0,0]+=1;np.savez(numeric,**values);meta['arrays']=adapter.reference(numeric);put(marker,meta)
 with pytest.raises(ValueError,match='padding'):module.load_pairs(plan)
 values['Base_interface'][0,0]+=1;np.savez(numeric,**values);meta['arrays']=adapter.reference(numeric);put(marker,meta)
 with pytest.raises(ValueError,match='fullFP32mmap SHA'):module.load_pairs(plan)


def test_new_root_entrypoint_supports_teacher_owner_import_contract():
 path=adapter.DEFAULT_ROOT/'code/fit_library.py'
 spec=importlib.util.spec_from_file_location('newroot_fit_owner_contract_fixture',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for name in ['ref','collect_request','emit_validation_diagnostics','emit_teacher_report','source_bindings','save_npz','validate_task']:
  function=getattr(module,name);assert callable(function)
  assert function.__globals__['ROOT']==adapter.DEFAULT_ROOT
 assert module.build_space.__code__ is module._original_source.build_space.__code__
