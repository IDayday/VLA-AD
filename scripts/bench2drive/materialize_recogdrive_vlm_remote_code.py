#!/usr/bin/env python3
"""Make a Trainer-saved InternVL checkpoint self-contained for AutoModel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


REMOTE_CODE_FILES = (
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "conversation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(checkpoint: Path, source: Path) -> dict:
    if not (checkpoint / "model.safetensors").is_file() or not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Incomplete VLM checkpoint: {checkpoint}")
    copied = []
    files = {}
    for name in REMOTE_CODE_FILES:
        source_path = source / name
        target_path = checkpoint / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Pinned VLM code source is missing {source_path}")
        source_hash = sha256(source_path)
        if target_path.exists():
            target_hash = sha256(target_path)
            if target_hash != source_hash:
                raise RuntimeError(
                    f"Refusing to replace non-identical remote code {target_path}: "
                    f"checkpoint={target_hash}, source={source_hash}"
                )
        else:
            with tempfile.NamedTemporaryFile(dir=checkpoint, prefix=f".{name}.", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copy2(source_path, temporary)
                temporary.replace(target_path)
            finally:
                temporary.unlink(missing_ok=True)
            copied.append(name)
            target_hash = sha256(target_path)
        files[name] = target_hash
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "source": str(source.resolve()),
        "copied": copied,
        "files": files,
    }
    report_path = checkpoint / "recogdrive_remote_code_provenance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.checkpoint, args.source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
