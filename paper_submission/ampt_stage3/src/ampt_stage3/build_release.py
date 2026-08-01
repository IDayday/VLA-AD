"""Create the file-hash manifest and deterministic paper-submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile


EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def release_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or relative.as_posix() == "RELEASE_MANIFEST.json":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> Path:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in release_files(root)
    ]
    payload = {
        "schema_version": 1,
        "release": "AMPT Evaluation (Anonymous)",
        "scope": "evaluation and result verification only",
        "files": rows,
    }
    output = root / "RELEASE_MANIFEST.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_archive(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(root) + [root / "RELEASE_MANIFEST.json"]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative = Path(root.name) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 8, 1, 0, 0, 0))
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "configs" / "release_scope.json").is_file():
        raise FileNotFoundError(f"not an AMPT evaluation release root: {root}")
    manifest = write_manifest(root)
    if args.archive is not None:
        write_archive(root, args.archive.expanduser().resolve())
    print(manifest)


if __name__ == "__main__":
    main()
