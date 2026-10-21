"""Admission tests use interface fixtures, never substitute for GPU qualification."""
import pytest
from scripts.process_improvement import resume_placement as q
from scripts.process_improvement.common import write


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'ROOT',tmp_path)
    monkeypatch.setattr(q,'sources',lambda:{'unit':'unchanged'})
    run=tmp_path/'runs/paused'
    write(run/'config.json',dict(owner=dict(host='original')))
    write(run/'paused.json',dict(step=977))
    value=dict(status='PASS',world=4,global_batch=32,optimizer_exact=True,
               all_rank_RNG_exact=True,cursor_reference_budget_exact=True,
               validated_initializations=['IL','RL','WARM'],
               qualification_source_hashes={'unit':'unchanged'},
               destination_host='original',source_host='new')
    write(tmp_path/'tests/native_GRPO_cross_host_resume.json',value)
    return run,value


def test_actual_qualified_pair_preserves_world_and_does_not_move_active_job(tmp_path,monkeypatch):
    run,_=setup(tmp_path,monkeypatch)
    assert q.resume_hosts(run,4)==['original','new']
    assert q.resume_hosts(run,8)==['original']
    (run/'paused.json').unlink()
    assert q.resume_hosts(run,4)==['original']


@pytest.mark.parametrize('field,value',[
    ('status','FAILED'),('optimizer_exact',False),('all_rank_RNG_exact',False),
    ('qualification_source_hashes',{'unit':'changed'}),('destination_host','untested'),
    ('validated_initializations',['IL']),('global_batch',8)])
def test_incomplete_stale_or_unrelated_qualification_cannot_move_resume(tmp_path,monkeypatch,field,value):
    run,gate=setup(tmp_path,monkeypatch);gate[field]=value
    write(tmp_path/'tests/native_GRPO_cross_host_resume.json',gate)
    assert q.resume_hosts(run,4)==['original']


def test_world8_requires_its_own_real_resume_and_source_identity(tmp_path,monkeypatch):
    run,value=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(q,'host8_sources',lambda:{'unit8':'unchanged'})
    assert q.resume_hosts(run,8)==['original']
    value.update(world=8,qualification_source_hashes={'unit8':'unchanged'})
    write(tmp_path/'tests/native_GRPO_cross_host_resume8.json',value)
    assert q.resume_hosts(run,8)==['original','new']
    value['all_rank_RNG_exact']=False
    write(tmp_path/'tests/native_GRPO_cross_host_resume8.json',value)
    assert q.resume_hosts(run,8)==['original']
