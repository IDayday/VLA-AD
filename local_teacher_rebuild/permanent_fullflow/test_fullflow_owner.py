"""Private transport/authority fixtures; no actual native execution is claimed."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import fullflow_teacher_owner as owner

@pytest.fixture
def full_tree(tmp_path):
    gt=tmp_path/'native_gt'
    owner.atomic(gt/'SEARCH_COMPLETE.json',{'status':'FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE','stages':owner.STAGES,'train_failures':1850})
    owner.atomic(gt/'runs/QUEUE.json',{'active':False,'stages':owner.STAGES+['export_post']})
    for stage in owner.STAGES:
        summary=gt/'analysis'/('summary_'+stage)/'SUMMARY.json';owner.atomic(summary,{'synthetic_only':True,'stage':stage})
        owner.atomic(gt/'runs'/stage/'ACCEPTED.json',{'status':'PASS_COMPLETE','scenes':1850,'summary':str(summary.relative_to(gt)),'sha256':owner.sha(summary)})
    final=gt/'analysis/summary_post_progress/SUMMARY.json'
    owner.atomic(gt/'GT_POST_COMPLETE.json',{'status':'ALL1850_LOCAL_GT_POST_PAIRS_EXPORTED','actual_exports':1850,'train_failures':1850,
                  'search_summary':str(final.relative_to(gt)),'search_summary_sha256':owner.sha(final)})
    return tmp_path

def test_all_original_stage_authority(full_tree):
    result=owner.full_gt_acceptance(full_tree)
    assert len(result['all15_original_stages'])==15
    assert not result['shortened_cut_or_FAST_exports_used']

def test_shortened_search_and_missing_stage_are_rejected(full_tree):
    path=full_tree/'native_gt/SEARCH_COMPLETE.json';value=owner.read(path)
    value['status']='SHORTENED_LOCAL_GT_SEARCH_CUT';owner.atomic(path,value)
    with pytest.raises(ValueError,match='Every original'):owner.full_gt_acceptance(full_tree)
    value['status']='FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE';owner.atomic(path,value)
    (full_tree/'native_gt/runs/post_progress/ACCEPTED.json').unlink()
    with pytest.raises(FileNotFoundError):owner.full_gt_acceptance(full_tree)

def test_changed_summary_and_live_queue_fail_closed(full_tree):
    path=full_tree/'native_gt/runs/QUEUE.json';queue=owner.read(path);queue['active']=True;owner.atomic(path,queue)
    with pytest.raises(ValueError,match='queue'):owner.full_gt_acceptance(full_tree)
    queue['active']=False;owner.atomic(path,queue)
    (full_tree/'native_gt/analysis/summary_post_C/SUMMARY.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):owner.full_gt_acceptance(full_tree)

def test_external_partial_status_is_pending_scientific_JSON_remains_strict(tmp_path):
    path=tmp_path/'status.json';path.write_text('{"state":')
    result=owner.original.actor_summary([{'status':str(path)}])
    assert result['pending_or_running']==1 and not result['all_exited']
    with pytest.raises(json.JSONDecodeError):owner.read(path)

def test_withdrawn_host_cannot_enter_permanent_allocation():
    with pytest.raises(ValueError,match='outside'):
        owner.admit([{'host':'training-vlawm-tsinghua-zt6','gpu':1,'uuid':'PRIVATE_INVALID_FIXTURE'}])
