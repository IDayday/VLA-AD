from __future__ import annotations

from scripts.bench2drive.evaluate_recogdrive_b2d_stage1 import (
    parse_trajectory,
    select_rows,
    shifted_command,
)


def test_parse_trajectory_requires_exactly_eight_finite_points():
    points = ", ".join(f"({index}.0, {-index}.5, 0.0)" for index in range(8))
    parsed = parse_trajectory(f"Here is the planning trajectory [PT, {points}].")
    assert parsed is not None
    assert parsed.shape == (8, 3)
    assert parse_trajectory("[PT, (0, 0, 0)]") is None


def test_select_rows_covers_clips_before_second_sample():
    rows = [
        {"id": f"{clip}_{index}", "clip": clip}
        for clip in ("a", "b", "c")
        for index in range(3)
    ]
    selected = select_rows(rows, 3, seed=7)
    assert {row["clip"] for row in selected} == {"a", "b", "c"}
    assert [row["id"] for row in selected] == [row["id"] for row in select_rows(rows, 3, seed=7)]


def test_shifted_command_changes_known_command():
    question = "3. Active navigation command: [GO STRAIGHT]"
    assert shifted_command(question) == "3. Active navigation command: [TURN LEFT]"
