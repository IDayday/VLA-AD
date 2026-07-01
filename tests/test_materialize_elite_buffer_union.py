from __future__ import annotations

import os
from pathlib import Path


def _write_record(root: Path, name: str, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def test_materialize_elite_buffer_union_later_sources_override(tmp_path: Path):
    from scripts.materialize_elite_buffer_union import materialize_union

    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    output = tmp_path / "union"
    base_a = _write_record(base, "a.pkl.xz", "base-a")
    _write_record(base, "shared.pkl.xz", "base-shared")
    supplement_shared = _write_record(supplement, "shared.pkl.xz", "supplement-shared")
    supplement_b = _write_record(supplement, "b.pkl.xz", "supplement-b")
    _write_record(supplement, "ignored.txt", "ignored")

    summary = materialize_union([base, supplement], output)

    assert summary["union_records"] == 3
    assert (output / "union_manifest.json").is_file()
    assert (output / "a.pkl.xz").is_symlink()
    assert (output / "b.pkl.xz").is_symlink()
    assert (output / "shared.pkl.xz").is_symlink()
    assert Path(os.readlink(output / "a.pkl.xz")) == base_a.resolve()
    assert Path(os.readlink(output / "b.pkl.xz")) == supplement_b.resolve()
    assert Path(os.readlink(output / "shared.pkl.xz")) == supplement_shared.resolve()


def test_materialize_elite_buffer_union_overwrite_updates_stale_link(tmp_path: Path):
    from scripts.materialize_elite_buffer_union import materialize_union

    source = tmp_path / "source"
    stale = tmp_path / "stale"
    output = tmp_path / "union"
    target = _write_record(source, "a.pkl.xz", "source")
    stale_target = _write_record(stale, "a.pkl.xz", "stale")
    output.mkdir()
    os.symlink(stale_target, output / "a.pkl.xz")

    summary = materialize_union([source], output, overwrite=True)

    assert summary["replaced"] == 1
    assert Path(os.readlink(output / "a.pkl.xz")) == target.resolve()
