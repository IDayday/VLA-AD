"""Protect measured endpoint costs from live-file overwrite and imputation."""
import json
import pytest
from scripts.process_improvement import formal_controls


def actual_ledger_fixture(tmp_path):
    # Interface-only unit fixture, never a NAVSIM experiment or result.
    run=tmp_path/'run';run.mkdir()
    (run/'config.json').write_text(json.dumps(dict(world=1,global_batch=32,group_size=8)))
    (run/'head_977.pt').write_bytes(b'unit-interface-checkpoint')
    ledger=dict(step=977,compute={},budget=dict(physical_scores=977*256),
                allocated_GPU_seconds=100.,peak_gpu_bytes=123,peak_history_complete=True)
    (run/'cost_rank_0.json').write_text(json.dumps(ledger))
    return run,ledger


def test_later_live_costs_cannot_relabel_an_earlier_measured_endpoint(tmp_path):
    run,ledger=actual_ledger_fixture(tmp_path)
    first=formal_controls.freeze_endpoint_cost(run,977)
    ledger.update(step=3907,allocated_GPU_seconds=400.)
    ledger['budget']['physical_scores']=3907*256
    (run/'cost_rank_0.json').write_text(json.dumps(ledger))
    retained=formal_controls.freeze_endpoint_cost(run,977)
    assert retained==first
    assert retained['ledger']['GPU_seconds']==100.
    assert retained['ledger']['budget']['physical_scores']==977*256


def test_wrong_training_step_cannot_fill_missing_endpoint_costs(tmp_path):
    run,_=actual_ledger_fixture(tmp_path)
    (run/'head_3907.pt').write_bytes(b'unit-interface-later-checkpoint')
    with pytest.raises(ValueError,match='another training step'):
        formal_controls.freeze_endpoint_cost(run,3907)


def test_changed_archived_costs_fail_instead_of_silently_repairing(tmp_path):
    run,_=actual_ledger_fixture(tmp_path)
    formal_controls.freeze_endpoint_cost(run,977)
    path=run/'endpoint_costs/977/rank_0.json'
    data=json.loads(path.read_text());data['allocated_GPU_seconds']=0.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='ledger changed'):
        formal_controls.freeze_endpoint_cost(run,977)
