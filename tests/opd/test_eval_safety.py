import pytest
from scripts.opd.common import gt


def test_T22_no_test_GT():
    with pytest.raises(ValueError,match='outside registered training'):gt(dict(split='navtest',token='not_a_training_token'))


def test_T08_trainable_teacher_rejected():
    import torch
    from navsim.agents.recogdrive.opd.trainer import frozen_query
    with pytest.raises(ValueError,match='must be frozen'):frozen_query(torch.nn.Linear(3,3),None,None,None)
