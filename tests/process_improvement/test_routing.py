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
