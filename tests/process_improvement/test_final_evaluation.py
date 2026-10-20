"""Scientific admission tests; no fabricated policy outcomes."""
from copy import deepcopy

import pytest

from navsim.agents.recogdrive.process_improvement.final_evaluation import validate_final_evaluation


def arguments():
    return dict(name='student42', checkpoint_sha256='weights', seeds=[7101, 7102],
                screening=False, limit=None, population_sha256='population',
                protocol_sha256='protocol')


def fixture_lock():
    return dict(status='LOCKED', protocol_sha256='protocol',
                navtest_manifest_sha256='population',
                candidates=[dict(name='student42', checkpoint_sha256='weights',
                                 seeds=[7101, 7102], screening=False)])


def test_exact_final_candidate_is_admitted():
    lock = fixture_lock()
    before = deepcopy(lock)
    assert validate_final_evaluation(lock, **arguments()) == lock['candidates'][0]
    assert lock == before


@pytest.mark.parametrize('field,value', [
    ('name', 'unregistered_best'), ('checkpoint_sha256', 'new_weights'),
    ('seeds', [999]), ('screening', True), ('limit', 64),
    ('population_sha256', 'selected_scenes'), ('protocol_sha256', 'new_protocol'),
])
def test_changed_final_candidate_is_rejected(field, value):
    args = arguments()
    args[field] = value
    with pytest.raises(ValueError):
        validate_final_evaluation(fixture_lock(), **args)


def test_duplicate_names_or_unlocked_spec_are_rejected():
    lock = fixture_lock()
    lock['candidates'] *= 2
    with pytest.raises(ValueError, match='unique'):
        validate_final_evaluation(lock, **arguments())
    lock = fixture_lock()
    lock['status'] = 'DRAFT'
    with pytest.raises(ValueError, match='not locked'):
        validate_final_evaluation(lock, **arguments())
