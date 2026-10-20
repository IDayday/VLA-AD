"""Selection logic only; analytic fixtures are never experimental evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.process_improvement import calibrate


@pytest.mark.parametrize('extend', [False, True])
def test_warm_uses_all_and_only_registered_eligible_endpoints(tmp_path, monkeypatch, extend):
    (tmp_path/'selection').mkdir()
    choice = {'decisions': {
        'OPD': {'extend_to_four_epochs': extend},
        'ANCHOR': {'extend_to_four_epochs': False},
    }}
    (tmp_path/'selection/four_epoch_extension.json').write_text(json.dumps(choice))
    monkeypatch.setattr(calibrate, 'ROOT', tmp_path)
    monkeypatch.setattr(calibrate, 'config', lambda: {'train_seeds': [42, 43, 44]})
    monkeypatch.setattr(calibrate, 'sha', lambda path: str(Path(path).name))
    calls = []

    def scores(name, split):
        assert split == 'cal_tune'
        epoch = int(name.rsplit('epoch', 1)[1])
        assert epoch in [1, 2] or (extend and name.startswith('full_OPD_') and epoch == 4)
        calls.append(name)
        return np.full((2, 5, 7), epoch/10), []

    monkeypatch.setattr(calibrate, 'predictions', scores)
    result = calibrate.warm()
    assert len(calls) == (15 if extend else 12)
    assert result['chosen']['method'] == 'OPD'
    assert result['chosen']['epoch'] == (4 if extend else 2)
    assert set(result['chosen']['checkpoints']) == {'42', '43', '44'}
    assert len(result['trials']) == (5 if extend else 4)
    # A rerun must reproduce the same lock rather than retune it.
    assert calibrate.warm() == result


def test_calibration_lock_rejects_changed_decision(tmp_path):
    target = tmp_path/'choice.json'
    calibrate.immutable(target, {'chosen': 'original'})
    with pytest.raises(ValueError, match='cannot be overwritten'):
        calibrate.immutable(target, {'chosen': 'new'})
    assert json.loads(target.read_text()) == {'chosen': 'original'}
