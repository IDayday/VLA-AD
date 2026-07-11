from scripts.bench2drive.check_recogdrive_b2d_stage1_environment import normalize_version


def test_normalize_version_removes_cuda_build_suffix() -> None:
    assert normalize_version("2.2.2+cu121") == "2.2.2"


def test_normalize_version_leaves_plain_version_unchanged() -> None:
    assert normalize_version("4.37.2") == "4.37.2"
