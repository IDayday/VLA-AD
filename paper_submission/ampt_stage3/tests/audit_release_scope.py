"""Fail on training code, bundled artifacts, or double-blind identity leaks."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".json", ".toml", ".txt"}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".ckpt", ".pt", ".pth", ".pkl", ".parquet", ".xlsx", ".csv", ".npy", ".npz"
}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
ABSOLUTE_PRIVATE_PATH = re.compile(r"/(?:mnt|home|root)/", re.IGNORECASE)
NETWORK_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def main() -> None:
    scope = json.loads((ROOT / "configs/release_scope.json").read_text(encoding="utf-8"))
    assert scope["release_mode"] == "anonymous_double_blind"
    included = " ".join(scope["included"]).lower()
    excluded = " ".join(scope["excluded"]).lower()
    assert "evaluation" in included and "all model training" in excluded

    assert not list((ROOT / "src").rglob("train*.py"))
    assert not list((ROOT / "scripts").glob("train*.sh"))
    assert not list((ROOT / "configs").glob("*train*.yaml"))

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        assert path.suffix.lower() not in FORBIDDEN_ARTIFACT_SUFFIXES, relative
        assert path.stat().st_size < 1_000_000, f"unexpected large file: {relative}"
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == "LICENSE":
            text = path.read_text(encoding="utf-8")
            assert not EMAIL.search(text), f"email address in {relative}"
            assert not ABSOLUTE_PRIVATE_PATH.search(text), f"private path in {relative}"
            assert not NETWORK_ADDRESS.search(text), f"network address in {relative}"

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright 2026 Anonymous Authors." in license_text

    scripts = list((ROOT / "scripts").glob("*.sh"))
    assert scripts
    for script in scripts:
        assert script.stat().st_mode & 0o100, f"not executable: {script.name}"

    for name in ("eval_navsim_v1.yaml", "eval_navsim_v2.yaml"):
        config = (ROOT / "configs" / name).read_text(encoding="utf-8")
        assert "inference_trajectories_per_scene: 1" in config
        assert "test_time_scorer: false" in config
        assert "reranking: false" in config
    print("evaluation-only double-blind audit: PASS")


if __name__ == "__main__":
    main()
