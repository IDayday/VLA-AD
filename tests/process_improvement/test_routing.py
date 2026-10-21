"""Lease-ordering contracts, independent of scientific outcome fixtures."""
import json

import pytest

from scripts.process_improvement import coordinator


def setup_root(tmp_path, monkeypatch, status, expansion=False):
    monkeypatch.setattr(coordinator, 'ROOT', tmp_path)
    selection=tmp_path/'selection'
    selection.mkdir()
    (selection/'warm_start.json').write_text('{}')
    (selection/'process_gate.json').write_text(json.dumps(dict(status=status,expansion_allowed=expansion)))


@pytest.mark.parametrize('status,expansion,expected', [
    ('SUPPORTED',False,False), ('INCONCLUSIVE',True,False),
    ('NOT_SUPPORTED',False,True), ('INCONCLUSIVE',False,True),
])
def test_local_whole_host_handoff_keeps_independent_process_capacity(tmp_path,monkeypatch,status,expansion,expected):
    setup_root(tmp_path,monkeypatch,status,expansion)
    assert coordinator.local_rl_handoff_ready([]) is expected


def test_short_lease_must_actually_finish_successfully_before_handoff(tmp_path,monkeypatch):
    setup_root(tmp_path,monkeypatch,'NOT_SUPPORTED')
    tasks=[dict(name='audit',gpus=1)]
    assert not coordinator.local_rl_handoff_ready(tasks)
    directory=tmp_path/'jobs/audit'
    directory.mkdir(parents=True)
    assert not coordinator.local_rl_handoff_ready(tasks)
    (directory/'complete.json').write_text('{"status":"FAILED"}')
    assert not coordinator.local_rl_handoff_ready(tasks)
    (directory/'complete.json').write_text('{"status":"COMPLETE"}')
    assert coordinator.local_rl_handoff_ready(tasks)


@pytest.mark.parametrize('name',[
    'auto_eval_cal_confirm_GRPO_IL_42_step977',
    'reviewed_retry1_auto_eval_dev_GRPO_WARM_44_step15625',
])
def test_formal_endpoint_evaluation_can_overlap_qualified_fresh_training(tmp_path,monkeypatch,name):
    setup_root(tmp_path,monkeypatch,'NOT_SUPPORTED')
    tasks=[dict(name=name,gpus=1)]
    assert coordinator.local_rl_handoff_ready(tasks)
    # An independent mechanism prerequisite still prevents the handoff.
    tasks.append(dict(name='qualification_native_GRPO_distributed',gpus=1))
    assert not coordinator.local_rl_handoff_ready(tasks)


@pytest.mark.parametrize('name',[
    'auto_eval_cal_tune_GRPO_IL_42_step977',
    'auto_eval_dev_GRPO_IL_42_step1000',
    'auto_eval_dev_full_OPD_42_epoch1',
])
def test_endpoint_overlap_never_bypasses_other_short_prerequisites(tmp_path,monkeypatch,name):
    setup_root(tmp_path,monkeypatch,'NOT_SUPPORTED')
    assert not coordinator.local_rl_handoff_ready([dict(name=name,gpus=1)])


def owned_job(tmp_path,name,gpus,complete=False,launched=True):
    directory=tmp_path/'jobs'/name;directory.mkdir(parents=True)
    (directory/'spec.json').write_text(json.dumps(dict(host='host',gpus=gpus)))
    if launched:(directory/'launch.json').write_text('{}')
    if complete:(directory/'complete.json').write_text('{"status":"COMPLETE"}')


def test_gang_reservation_only_drains_our_own_leases(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,'ROOT',tmp_path)
    owned_job(tmp_path,'running',list(range(1,8)))
    assert coordinator.fully_research_owned_or_idle('host',[0])
    assert not coordinator.fully_research_owned_or_idle('other_host',[0])


def test_unknown_missing_gpu_is_never_a_gang_reservation(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,'ROOT',tmp_path)
    owned_job(tmp_path,'running',[1,2,3,4,5,6])
    assert not coordinator.fully_research_owned_or_idle('host',[0])


def test_finished_or_unlaunched_jobs_do_not_prove_gpu_ownership(tmp_path,monkeypatch):
    monkeypatch.setattr(coordinator,'ROOT',tmp_path)
    owned_job(tmp_path,'finished',[7],complete=True)
    owned_job(tmp_path,'rejected',[7],launched=False)
    assert not coordinator.fully_research_owned_or_idle('host',list(range(7)))


def qualified_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, 'ROOT', tmp_path)
    monkeypatch.setattr(coordinator, 'sha', lambda path: 'test_source_identity')
    folder=tmp_path/'tests';folder.mkdir()
    gate=dict(status='PASS',world_sizes=[4,8],global_batch=32,actual_optimizer_updates=4,
              exact_resume_optimizer=True,validated_initializations=['IL','RL','WARM'])
    for field in ['train_source_sha256','planner_source_sha256','rng_source_sha256','qualification_source_sha256']:
        gate[field]='test_source_identity'
    (folder/'native_GRPO_partition32.json').write_text(json.dumps(gate))
    return dict(name='formal_GRPO_IL_42_to977',gpus=8,commands=[[
        '-m','torch.distributed.run','--nproc_per_node=8','scripts/process_improvement/train_grpo.py',
        '--name','GRPO_IL_42','--global-batch','32']])


@pytest.mark.parametrize('host',['training-vla-zt','training-rl-zt2'])
def test_new_partition_requires_real_gate_and_short_work_handoff(tmp_path,monkeypatch,host):
    task=qualified_partition(tmp_path,monkeypatch)
    assert coordinator.local_partition_task(task,host,list(range(7)),False)==task
    own=coordinator.local_partition_task(task,host,list(range(7)),True)
    assert own['gpus']==4
    assert '--nproc_per_node=4' in own['commands'][0]
    assert task['gpus']==8 and '--nproc_per_node=8' in task['commands'][0]
    (tmp_path/'tests/native_GRPO_partition32.json').unlink()
    assert coordinator.local_partition_task(task,host,list(range(7)),True)==task


def test_reviewed_prelaunch_retry_keeps_qualified_partition_available(tmp_path,monkeypatch):
    task=qualified_partition(tmp_path,monkeypatch)
    task['name']='reviewed_retry1_formal_GRPO_IL_42_to977'
    assert coordinator.local_partition_task(task,'training-vla-zt',list(range(7)),True)['gpus']==4


@pytest.mark.parametrize('host,idle',[
    ('training-vlawm-tsinghua-zt5',list(range(7))),('training-vla-zt',list(range(8))),
    ('training-vla-zt',list(range(3)))])
def test_partition_preserves_nonlocal_whole_host_and_capacity_rules(tmp_path,monkeypatch,host,idle):
    task=qualified_partition(tmp_path,monkeypatch)
    assert coordinator.local_partition_task(task,host,idle,True)==task


def test_partition_never_changes_an_existing_or_resume_lineage(tmp_path,monkeypatch):
    task=qualified_partition(tmp_path,monkeypatch)
    task['commands'][0].append('--resume')
    assert coordinator.local_partition_task(task,'training-vla-zt',list(range(7)),True)==task
    task['commands'][0].pop()
    run=tmp_path/'runs/GRPO_IL_42';run.mkdir(parents=True);(run/'config.json').write_text('{}')
    assert coordinator.local_partition_task(task,'training-vla-zt',list(range(7)),True)==task


@pytest.mark.parametrize('field,value',[
    ('status','FAILED'),('global_batch',8),('exact_resume_optimizer',False),
    ('validated_initializations',['IL']),
    ('train_source_sha256','changed_source')])
def test_failed_or_stale_partition_gate_cannot_admit_formal_runs(tmp_path,monkeypatch,field,value):
    task=qualified_partition(tmp_path,monkeypatch)
    path=tmp_path/'tests/native_GRPO_partition32.json';gate=json.loads(path.read_text());gate[field]=value
    path.write_text(json.dumps(gate))
    assert coordinator.local_partition_task(task,'training-vla-zt',list(range(7)),True)==task


def test_partition_does_not_apply_global32_receipt_to_another_batch(tmp_path,monkeypatch):
    task=qualified_partition(tmp_path,monkeypatch)
    task['commands'][0][-1]='8'
    assert coordinator.local_partition_task(task,'training-vla-zt',list(range(7)),True)==task


def test_formal_resume_preserves_actual_world4_and_original_host(tmp_path,monkeypatch):
    from scripts.process_improvement import formal_controls
    monkeypatch.setattr(formal_controls,'ROOT',tmp_path)
    monkeypatch.setattr(coordinator,'ROOT',tmp_path)
    for name,value in [
        ('selection/grpo_rates.json',dict(selected={k:dict(lr=1e-5) for k in ['IL','RL']})),
        ('tests/native_GRPO_distributed_resume.json',dict(status='PASS')),
        ('runs/GRPO_IL_42/config.json',dict(world=4,global_batch=32,group_size=8,owner=dict(host='training-vla-zt'))),
        ('runs/GRPO_IL_42/paused.json',dict(step=977)),
        ('runs/GRPO_IL_42/harm_audit_977.json',dict(status='CONTINUE'))]:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    # Cost-interface unit fixtures only; not research artifacts or observations.
    run=tmp_path/'runs/GRPO_IL_42'
    (run/'head_977.pt').write_bytes(b'routing-unit-checkpoint')
    for rank in range(4):
        (run/f'cost_rank_{rank}.json').write_text(json.dumps(dict(
            step=977,compute={},budget=dict(physical_scores=977*256//4),
            allocated_GPU_seconds=100.,peak_history_complete=False)))
    task=next(t for t in formal_controls.tasks() if t['name']=='formal_GRPO_IL_42_to3907')
    assert task['gpus']==4 and task['hosts']==['training-vla-zt']
    assert '--nproc_per_node=4' in task['commands'][0] and '--resume' in task['commands'][0]
