import pytest

from scripts.onevl.bridge_ar_answer_to_recogdrive_dit import select_current_image


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
