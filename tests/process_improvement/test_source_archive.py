"""Launch snapshots preserve actual source bytes, not inferred Git versions."""
import hashlib
import tarfile
import json

import pytest
from scripts.process_improvement import dispatch


def test_archive_retains_uncommitted_small_sources_and_excludes_assets(tmp_path,monkeypatch):
    source=tmp_path/'source';dest=tmp_path/'job';dest.mkdir()
    monkeypatch.setattr(dispatch,'SOURCE',source)
    expected={
        'scripts/process_improvement/train.py':b'# actual uncommitted text\n',
        'scripts/opd/inputs.py':b'# observation input\n',
        'navsim/agents/recogdrive/recogdrive_agent.py':b'# base model\n',
        'configs/process_improvement/base.yaml':b'frozen_VLM: true\n'}
    for name,value in expected.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    (source/'model.ckpt').write_bytes(b'must not enter source archive')
    (source/'.env').write_text('PRIVATE_TEST_PLACEHOLDER=not_a_real_credential')
    hashes,receipt=dispatch.archive_launch_sources(dest,0)
    assert hashes=={n:hashlib.sha256(v).hexdigest() for n,v in expected.items()}
    assert receipt['files']==4
    with tarfile.open(receipt['path']) as archive:
        assert set(archive.getnames())==set(expected)
        for name,value in expected.items():assert archive.extractfile(name).read()==value
    assert hashlib.sha256((dest/'source_snapshot_0.tar.gz').read_bytes()).hexdigest()==receipt['sha256']
    with pytest.raises(FileExistsError,match='immutable'):dispatch.archive_launch_sources(dest,0)


def test_source_preflight_failure_is_recorded_without_starting_child(tmp_path,monkeypatch):
    spec=tmp_path/'spec.json';spec.write_text(json.dumps(dict(commands=[['must_not_start.py']])))
    monkeypatch.setattr(dispatch,'identity',lambda:dict(host='unit_fixture',pid=0))
    def fail(*args):raise OSError('source snapshot unavailable')
    monkeypatch.setattr(dispatch,'archive_launch_sources',fail)
    monkeypatch.setattr(dispatch.subprocess,'Popen',lambda *a,**k:pytest.fail('Child launched after failed source preflight'))
    with pytest.raises(SystemExit):dispatch.worker(spec)
    value=json.loads((tmp_path/'complete.json').read_text())
    assert value['status']=='FAILED' and value['results'][0]['returncode'] is None
    assert 'source snapshot unavailable' in value['results'][0]['prelaunch_error']
    assert not (tmp_path/'child_0.json').exists()
