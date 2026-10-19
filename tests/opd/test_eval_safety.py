import pytest
from scripts.opd.common import gt


def test_T22_no_test_GT():
    with pytest.raises(ValueError,match='outside registered training'):gt(dict(split='navtest',token='not_a_training_token'))


def test_T08_trainable_teacher_rejected():
    import torch
    from navsim.agents.recogdrive.opd.trainer import frozen_query
    with pytest.raises(ValueError,match='must be frozen'):frozen_query(torch.nn.Linear(3,3),None,None,None)


def test_resume_rejects_a_different_checkpoint_or_scene():
    from scripts.opd.common import digest
    from scripts.opd.evaluate import validate_metadata
    cfg={'inference_seeds':[1,2,3,4,5]};meta=dict(checkpoint_sha256={'student':'old'},config_sha256=digest(cfg),seeds=cfg['inference_seeds'],split='dev',token='scene_a')
    validate_metadata(meta,{'student':'old'},cfg,'dev','scene_a')
    with pytest.raises(ValueError,match='model/config'):validate_metadata(meta,{'student':'new'},cfg,'dev','scene_a')
    with pytest.raises(ValueError,match='scene/seed'):validate_metadata(meta,{'student':'old'},cfg,'dev','scene_b')
