"""Reject unrelated or changed jobs before any maintenance signal or launch."""
import pytest
from scripts.process_improvement.maintenance_native import verify_command


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
