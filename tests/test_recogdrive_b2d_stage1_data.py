from __future__ import annotations

import gzip
import json
from pathlib import Path

from PIL import Image

from navsim.agents.recogdrive.recogdrive_backbone import (
    BENCH2DRIVE_SYSTEM_MESSAGE,
    build_recogdrive_planning_question,
    format_recogdrive_trajectory_answer,
)
from scripts.bench2drive.prepare_recogdrive_b2d_stage1_sft import (
    parse_args,
    prepare_dataset,
    stable_clip_split,
)


def _make_clip(data_root: Path, name: str) -> None:
    clip = data_root / name
    for index in range(60):
        annotation = {
            "x": 100.0 + 0.1 * index,
            "y": 20.0 + 0.01 * index,
            "theta": 0.001 * index,
            "speed": 2.0,
            "command_near": 4,
            "command_far": 4,
            "next_command": 4,
            "acceleration": [0.0, 0.0, 9.8],
        }
        path = clip / "anno" / f"{index:05d}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as output:
            json.dump(annotation, output)
    image_dir = clip / "camera" / "rgb_front"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (32, 18), color=(20, 40, 60)).save(image_dir / "00015.jpg")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_prompt_and_answer_contract():
    history = [[-1.0, 0.0, -0.1], [-0.5, 0.0, -0.05], [0.0, 0.0, 0.0], [0.1, 0.0, 0.01]]
    question = build_recogdrive_planning_question(history, [0.0, 1.0, 0.0])
    assert question.startswith("<image>\n")
    assert "last 4 timesteps" in question
    assert "[GO STRAIGHT]" in question
    assert "Predict 8 future trajectory points" in question

    answer = format_recogdrive_trajectory_answer([[0.0, float(index), 0.0] for index in range(8)])
    assert answer.startswith("Here is the planning trajectory [PT, ")
    assert answer.count("(") == 8
    assert answer.endswith("].")


def test_stage1_dataset_is_clip_disjoint_and_deterministic(tmp_path: Path):
    data_root = tmp_path / "Bench2Drive-Base"
    for index in range(4):
        _make_clip(data_root, f"Scenario_Town01_Route{index}_Weather0")

    first_output = tmp_path / "stage1_a"
    first = prepare_dataset(
        parse_args(
            [
                "--data-root",
                str(data_root),
                "--output-dir",
                str(first_output),
                "--validation-fraction",
                "0.25",
            ]
        )
    )
    assert first["num_train_clips"] == 3
    assert first["num_val_clips"] == 1
    assert first["num_train_samples"] == 3
    assert first["num_val_samples"] == 1

    train_clips = set((first_output / "train_clips.txt").read_text().splitlines())
    val_clips = set((first_output / "val_clips.txt").read_text().splitlines())
    assert train_clips.isdisjoint(val_clips)
    assert train_clips | val_clips == {f"Scenario_Town01_Route{index}_Weather0" for index in range(4)}

    train_rows = _read_jsonl(first_output / "train.jsonl")
    val_rows = _read_jsonl(first_output / "val.jsonl")
    for row in train_rows + val_rows:
        assert row["conversations"][0] == {"from": "system", "value": BENCH2DRIVE_SYSTEM_MESSAGE}
        assert row["conversations"][1]["value"].startswith("<image>\n")
        assert row["conversations"][2]["value"].count("(") == 8
        assert not Path(row["image"]).is_absolute()
        assert (data_root / row["image"]).is_file()

    second_output = tmp_path / "stage1_b"
    second = prepare_dataset(
        parse_args(
            [
                "--data-root",
                str(data_root),
                "--output-dir",
                str(second_output),
                "--validation-fraction",
                "0.25",
            ]
        )
    )
    assert first["train_jsonl_sha256"] == second["train_jsonl_sha256"]
    assert first["val_jsonl_sha256"] == second["val_jsonl_sha256"]
    assert (first_output / "train_clips.txt").read_text() == (second_output / "train_clips.txt").read_text()


def test_stable_split_rejects_invalid_fraction():
    try:
        stable_clip_split(["a", "b"], validation_fraction=0.0, seed=1)
    except ValueError as error:
        assert "validation-fraction" in str(error)
    else:
        raise AssertionError("Expected invalid validation fraction to fail")
