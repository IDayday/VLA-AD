from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.bench2drive.check_recogdrive_b2d_reproduction_gate import (
    ReproductionGateError,
    audit_official_jsonl,
    normalize_official_image_path,
    sha256_file,
)


CAMERAS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back_left",
    "rgb_back_right",
    "rgb_back",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(clip: str, frame: int, *, qa: bool) -> dict:
    images = [f"./Bench2drive/v1/{clip}/camera/{camera}/{frame:05d}.jpg" for camera in CAMERAS]
    if qa:
        conversations = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ]
    else:
        conversations = [
            {"from": "system", "value": "system"},
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ]
    return {"id": frame, "image": images, "conversations": conversations}


def _materialize_images(raw_root: Path, record: dict) -> None:
    for value in record["image"]:
        relative = normalize_official_image_path(value)
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")


def test_normalize_official_image_path_rejects_nonofficial_and_parent_paths():
    assert normalize_official_image_path(
        "./Bench2drive/v1/clip/camera/rgb_front/00010.jpg"
    ) == Path("clip/camera/rgb_front/00010.jpg")
    with pytest.raises(ReproductionGateError, match="prefix"):
        normalize_official_image_path("clip/camera/rgb_front/00010.jpg")
    with pytest.raises(ReproductionGateError, match="unsafe"):
        normalize_official_image_path("./Bench2drive/v1/../secret.jpg")


def test_audit_official_jsonl_checks_schema_roles_and_images(tmp_path: Path):
    raw_root = tmp_path / "raw"
    rows = [_record("clip_a", 10, qa=False), _record("clip_b", 11, qa=False)]
    for row in rows:
        _materialize_images(raw_root, row)
    jsonl = tmp_path / "traj.jsonl"
    jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    expected = {
        "expected_rows": 2,
        "expected_clips": 2,
        "expected_images_per_record": 6,
        "expected_conversation_min": 3,
        "expected_conversation_max": 3,
        "expected_role_counts": {"system": 2, "human": 2, "gpt": 2},
    }

    summary = audit_official_jsonl(
        jsonl,
        raw_root,
        expected,
        verify_all_images=True,
    )

    assert summary["matches_expected"] is True
    assert summary["clips"] == 2
    assert summary["checked_images"] == 12
    assert set(summary["cameras"]) == set(CAMERAS)


def test_audit_official_jsonl_detects_missing_images(tmp_path: Path):
    raw_root = tmp_path / "raw"
    row = _record("clip_a", 10, qa=True)
    jsonl = tmp_path / "qa.jsonl"
    jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
    expected = {
        "expected_rows": 1,
        "expected_clips": 1,
        "expected_images_per_record": 6,
        "expected_conversation_min": 2,
        "expected_conversation_max": 2,
        "expected_role_counts": {"human": 1, "gpt": 1},
    }

    summary = audit_official_jsonl(
        jsonl,
        raw_root,
        expected,
        verify_all_images=True,
    )

    assert summary["matches_expected"] is False
    assert len(summary["missing_images"]) == 6


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize(
    "launcher",
    [
        "run_recogdrive_b2d_stage1_sft.sh",
        "run_recogdrive_b2d_stage2_il_official.sh",
    ],
)
def test_retired_custom_launchers_refuse_by_default(launcher: str):
    env = os.environ.copy()
    env.pop("ALLOW_RETIRED_CUSTOM_B2D_PIPELINE", None)
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "bench2drive" / launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert "retired custom" in result.stderr
