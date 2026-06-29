import pytest
import torch

from scripts.onevl.bridge_ar_answer_to_recogdrive_dit import select_current_image, split_status_feature_values


def test_current_image_policy_first_last_and_single_or_last():
    row = {"images": ["old.jpg", "current.jpg"]}

    assert select_current_image(row, "first") == "old.jpg"
    assert select_current_image(row, "last") == "current.jpg"
    assert select_current_image(row, "single_or_last") == "current.jpg"


def test_current_image_policy_strict_single_rejects_multi_image():
    row = {"images": ["old.jpg", "current.jpg"]}

    with pytest.raises(ValueError, match="strict_single"):
        select_current_image(row, "strict_single")


def test_current_image_policy_strict_single_accepts_single_image():
    row = {"images": ["current.jpg"]}

    assert select_current_image(row, "strict_single") == "current.jpg"


def test_split_status_feature_prefers_command4_when_fourth_command_slot_is_zero():
    status = torch.tensor([0.0, 1.0, 0.0, 0.0, 6.04, -0.17, -0.03, 1.72])
    high_command = torch.tensor([0.0, 1.0, 0.0])

    velocity, acceleration, policy = split_status_feature_values(status, high_command)

    assert velocity == pytest.approx([6.04, -0.17])
    assert acceleration == pytest.approx([-0.03, 1.72])
    assert policy == "command4_velocity2_acceleration2"


def test_split_status_feature_supports_command3_layout():
    status = torch.tensor([0.0, 1.0, 0.0, 6.04, -0.17, -0.03, 1.72, 0.12])
    high_command = torch.tensor([0.0, 1.0, 0.0])

    velocity, acceleration, policy = split_status_feature_values(status, high_command)

    assert velocity == pytest.approx([6.04, -0.17])
    assert acceleration == pytest.approx([-0.03, 1.72, 0.12])
    assert policy == "command3_velocity2_acceleration3"
