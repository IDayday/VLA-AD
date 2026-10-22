"""Reject unrelated or changed jobs before any maintenance signal or launch."""
import pytest
from scripts.process_improvement.maintenance_native import verify_command,verify_final_world8_command,verify_live_hosts


def command():
    return ['-m','torch.distributed.run','--nproc_per_node=4',
            'scripts/process_improvement/train_grpo.py','--name','GRPO_IL_44',
            '--score-budget','4000000','--stop-after','3907','--resume']


def test_only_registered_unchanged_native_continuation_is_accepted():
    verify_command(command(),'GRPO_IL_44',4)


@pytest.mark.parametrize('field,value',[
    ('--name','unrelated'),('--score-budget','999'),('--stop-after','15625')])
def test_changed_identity_or_budget_is_rejected(field,value):
    cmd=command();cmd[cmd.index(field)+1]=value
    with pytest.raises(ValueError):verify_command(cmd,'GRPO_IL_44',4)


def test_world_changes_and_fresh_initializations_are_rejected():
    with pytest.raises(ValueError):verify_command(command(),'GRPO_IL_44',8)
    cmd=command();cmd.remove('--resume')
    with pytest.raises(ValueError):verify_command(cmd,'GRPO_IL_44',4)


def final8():
    return ['-m','torch.distributed.run','--nproc_per_node=8',
            'scripts/process_improvement/train_grpo.py','--name','GRPO_IL_42',
            '--score-budget','4000000','--resume']


def test_current_world8_target_is_preserved():
    verify_final_world8_command(final8(),'GRPO_IL_42',8)


def test_current_world8_rejects_another_lineage_or_changed_world():
    with pytest.raises(ValueError):verify_final_world8_command(final8(),'GRPO_IL_43',8)
    with pytest.raises(ValueError):verify_final_world8_command(final8(),'GRPO_IL_42',4)


def test_current_world8_rejects_shorter_or_fresh_training():
    cmd=final8()+['--stop-after','12000']
    with pytest.raises(ValueError):verify_final_world8_command(cmd,'GRPO_IL_42',8)
    cmd=final8();cmd.remove('--resume')
    with pytest.raises(ValueError):verify_final_world8_command(cmd,'GRPO_IL_42',8)


@pytest.mark.parametrize('revoked',['source','destination'])
def test_live_withdrawal_prevents_native_maintenance_before_signal(revoked):
    registration=dict(source_host='source',destination_host='destination')
    policy=dict(allowed_hosts=['source','destination'],global_dispatch_prohibitions={revoked:{}})
    with pytest.raises(PermissionError):verify_live_hosts(registration,policy)


def test_authorized_host_pair_passes_without_inferred_new_authorization():
    verify_live_hosts(dict(source_host='source',destination_host='destination'),
                      dict(allowed_hosts=['source','destination']))
