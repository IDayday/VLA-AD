"""CPU interface fixtures only; never scientific training results."""
import pytest
import torch
from scripts.process_improvement.common import save
from scripts.process_improvement.endpoint_state import freeze_endpoint_state


def fixture(run):
    run.mkdir()
    cfg = dict(world=1, global_batch=32)
    policy = dict(student={'w': torch.tensor([2.])}, step=977, cursor=977*32,
                  physical_scores=977*256, config=cfg)
    state = dict(policy, full_student=policy['student'], optimizer={'state': {}},
                 rank_states=[dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.get_rng_state(),
                                   numpy_rng=None, python_rng=None, counts={})],
                 reference_sha256='unit-only', scheduler_state={}, amp_state=None,
                 history=[], seconds=1.)
    save(run/'latest.pt', state)
    save(run/'head_977.pt', policy)
    return state


def test_archive_survives_native_atomic_latest_replacement(tmp_path):
    run=tmp_path/'run';state=fixture(run)
    original=freeze_endpoint_state(run,977)
    state['step']=3907;state['optimizer']={'state': {'unit': torch.ones(1)}}
    save(run/'latest.pt',state)
    assert freeze_endpoint_state(run,977)==original
    actual=torch.load(run/'endpoint_states/977/full.pt',weights_only=False)
    assert actual['step']==977 and actual['optimizer']=={'state': {}}


def test_wrong_step_is_not_archived_as_historical_state(tmp_path):
    run=tmp_path/'run';state=fixture(run)
    state['step']=1000;save(run/'latest.pt',state)
    with pytest.raises(ValueError,match='different-step'):
        freeze_endpoint_state(run,977)
    assert not (run/'endpoint_states/977/receipt.json').exists()
    assert not (run/'endpoint_states/977/full.pt').exists()


def test_modified_archive_fails_integrity_check(tmp_path):
    run=tmp_path/'run';fixture(run);freeze_endpoint_state(run,977)
    (run/'endpoint_states/977/full.pt').write_bytes(b'unit-corruption')
    with pytest.raises(ValueError,match='changed'):
        freeze_endpoint_state(run,977)
