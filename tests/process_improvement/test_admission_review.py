"""A reviewed new job ID is allowed only for a proven pre-Popen rejection."""
import json
import sys

import pytest

from scripts.process_improvement import review_admission


def setup(tmp_path,monkeypatch,error='RuntimeError: GPU occupied: [7, 0, 100]'):
    monkeypatch.setattr(review_admission,'ROOT',tmp_path)
    monkeypatch.setattr(review_admission,'identity',lambda: {})
    monkeypatch.setattr(sys,'argv',['review_admission.py','--jobs','original'])
    directory=tmp_path/'jobs/original';directory.mkdir(parents=True)
    (directory/'spec.json').write_text('{"name":"original","host":"authorized"}')
    (directory/'admission_failure.json').write_text(json.dumps(dict(error=error)))
    return directory


def test_review_preserves_rejection_and_creates_only_a_new_mapping(tmp_path,monkeypatch):
    directory=setup(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in directory.iterdir()}
    review_admission.main()
    value=json.loads((tmp_path/'controller/admission_retry_map.json').read_text())
    assert value['current']=={'original':'reviewed_retry1_original'}
    assert {p.name:p.read_bytes() for p in directory.iterdir()}==before
    assert not (tmp_path/'jobs/reviewed_retry1_original').exists()


@pytest.mark.parametrize('marker',['launch.json','owner.json','child_0.json'])
def test_existing_launch_evidence_refuses_a_retry(tmp_path,monkeypatch,marker):
    directory=setup(tmp_path,monkeypatch)
    (directory/marker).write_text('{}')
    with pytest.raises(ValueError,match='potentially launched'):
        review_admission.main()
    assert not (tmp_path/'controller/admission_retry_map.json').exists()


def test_network_timeout_needs_process_investigation(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch,error='TimeoutError: SSH transport timed out after possible launch')
    with pytest.raises(ValueError,match='Ambiguous'):
        review_admission.main()
    assert not (tmp_path/'controller/admission_retry_map.json').exists()
