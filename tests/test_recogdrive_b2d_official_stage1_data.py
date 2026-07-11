from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.bench2drive.prepare_recogdrive_b2d_official_stage1 import (
    prepare_official_stage1_data,
)


CAMERAS = (
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back_left",
    "rgb_back_right",
    "rgb_back",
)


def _record(clip: str, frame: int, conversations: int, *, system: bool) -> dict:
    turns = []
    if system:
        turns.append({"from": "system", "value": "system"})
    while len(turns) < conversations:
        turns.append({"from": "human", "value": "<image>" * (6 if len(turns) == int(system) else 0)})
        if len(turns) < conversations:
            turns.append({"from": "gpt", "value": "answer"})
    return {
        "id": frame,
        "image": [f"./Bench2drive/v1/{clip}/camera/{camera}/{frame:05d}.jpg" for camera in CAMERAS],
        "conversations": turns,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_official_stage1_data_preserves_sources_and_selects_long_qa(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    traj = tmp_path / "traj.jsonl"
    qa = tmp_path / "qa.jsonl"
    traj_rows = [_record("clip_a", index, 3, system=True) for index in range(4)]
    qa_rows = [_record("clip_a", index, length, system=False) for index, length in enumerate((4, 6, 8, 20))]
    _write_jsonl(traj, traj_rows)
    _write_jsonl(qa, qa_rows)
    original_traj = traj.read_bytes()
    original_qa = qa.read_bytes()
    output = tmp_path / "prepared"

    summary = prepare_official_stage1_data(
        Namespace(
            traj_jsonl=traj,
            qa_jsonl=qa,
            raw_data_root=raw,
            output_dir=output,
            smoke_records_per_dataset=2,
        )
    )

    assert traj.read_bytes() == original_traj
    assert qa.read_bytes() == original_qa
    assert (output / "dataset_view/Bench2drive/v1").is_symlink()
    assert (output / "dataset_view/Bench2drive/v1").resolve() == raw.resolve()
    formal_meta = json.loads((output / "official_meta.json").read_text())
    assert list(formal_meta) == ["Bench2drive_Traj", "Bench2drive_QA"]
    assert formal_meta["Bench2drive_Traj"]["repeat_time"] == 1
    assert formal_meta["Bench2drive_QA"]["max_dynamic_patch"] == 12
    assert summary["smoke"]["max_qa_conversation_length"] == 20
    smoke_qa = Path(summary["smoke"]["qa_jsonl"])
    assert len(smoke_qa.read_text().splitlines()) == 2
