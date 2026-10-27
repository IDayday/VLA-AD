"""Private CPU tests for semantic reuse/admission; no native score is fabricated."""
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

NEW=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027')
sys.path.insert(0,str(NEW/'code'))
import local_gt_reuse as reuse
import gt_opt_worker as worker

def test_warm_paths_normalize_by_content_and_candidate_index():
    left={'id':'old','block':'interface_pre','cap':.1,'steps':128,'mode':'shared','lr_multiplier':1.,
          'initial':{'path':'/old/arrays.npz','sha256':'a'*64,'candidate_index':1,'source':'audit only'}}
    right=copy.deepcopy(left);right['id']='new';right['initial']['path']='/new/arrays.npz';right['initial']['source']='new audit path'
    assert reuse.semantic_key(42,left)==reuse.semantic_key(42,right)
    for field,value in [('steps',64),('cap',.2),('mode','image_text'),('lr_multiplier',.5)]:
        changed=copy.deepcopy(right);changed[field]=value;assert reuse.semantic_key(42,left)!=reuse.semantic_key(42,changed)
    changed=copy.deepcopy(right);changed['initial']['candidate_index']=0
    assert reuse.semantic_key(42,left)!=reuse.semantic_key(42,changed)
    changed=copy.deepcopy(right);changed['initial']['sha256']='b'*64
    assert reuse.semantic_key(42,left)!=reuse.semantic_key(42,changed)

def test_projection_retains_exact_source_repair_inputs():
    left={'id':'x','block':'interface_post','cap':.2,'steps':64,'mode':'shared','lr_multiplier':1.,
          'project_pre':{'arrays':'old','arrays_sha256':'a'*64,'source':'old','source_sha256':'b'*64,
                         'config':{'block':'interface_pre','cap':.1,'mode':'image_text'}}}
    right=copy.deepcopy(left);right['project_pre']['source']='new';right['project_pre']['source_sha256']='c'*64;right['project_pre']['arrays']='new'
    assert reuse.semantic_config(left)==reuse.semantic_config(right)
    right['project_pre']['config']['cap']=.2
    assert reuse.semantic_config(left)!=reuse.semantic_config(right)

def test_reuse_requires_same_Base_bytes_and_noise(tmp_path):
    protocol={'native_base_root':str(tmp_path/'base'),'reference_noise':{'protocol_sha256':'a','bank_sha256':'b','token_index_sha256':'c'}}
    reuse.atomic(tmp_path/'PROTOCOL.json',protocol)
    reuse.atomic(tmp_path/'base/train/cases/token/COMPLETE.json',{'arrays_sha256':'input','global_index':2,'token':'token'})
    source={'local_input_sha256':'input','reference_noise':dict(protocol['reference_noise'])}
    reuse._validate_new_inputs(tmp_path,2,'token',source)
    source['reference_noise']['bank_sha256']='other'
    with pytest.raises(ValueError,match='noise'):reuse._validate_new_inputs(tmp_path,2,'token',source)
    source['reference_noise']=dict(protocol['reference_noise']);source['local_input_sha256']='other'
    with pytest.raises(ValueError,match='identical'):reuse._validate_new_inputs(tmp_path,2,'token',source)

def test_old_intersection_cannot_claim_before_reuse_finishes(tmp_path):
    root=tmp_path/'gt';base=tmp_path/'base';rows={1:{'token':'old'},2:{'token':'new'}}
    worker.atomic(root/'runs/QUEUE.json',{'active':True,'stages':['baseline']})
    stage=root/'runs/baseline'
    for index in rows:
        worker.atomic(stage/'configuration_tasks'/f'scene_{index:06d}__c0000.json',{'id':f'scene_{index:06d}__c0000','index':index})
        worker.atomic(base/'train/cases'/rows[index]['token']/'COMPLETE.json',{})
    worker.atomic(stage/'REUSE_PENDING.json',{'blocked_old_intersection_indices':[1]})
    task,lock=worker.claim_ready(root,base,rows,'private_worker',3)
    assert task['index']==2;lock.close()
    worker.atomic(stage/'configuration_receipts'/(task['id']+'.json'),{'status':'PRIVATE_TEST_COMPLETE'})
    worker.atomic(stage/'REUSE_IMPORT_COMPLETE.json',{'status':'PRIVATE_TEST_COMPLETE'})
    task,lock=worker.claim_ready(root,base,rows,'private_worker',3)
    assert task['index']==1;lock.close()

def test_actual_manifest_is_exact_local_candidate_set():
    selected=reuse.read(NEW/'inputs/LOCAL_CANDIDATES.json')['candidates']
    actual=reuse.read(NEW/'native_gt/manifests/SCENES.json')
    assert len(actual)==len(selected)==1689
    assert {(r['global_index'],r['token'],r['session']) for r in actual}=={(r['global_index'],r['token'],r['session']) for r in selected}
    assert all(r['Base_S0']<.7 and r['population']=='train' for r in actual)
