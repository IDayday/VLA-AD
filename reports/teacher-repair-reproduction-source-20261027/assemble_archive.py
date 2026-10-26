#!/usr/bin/env python3
"""Join checked source-archive volumes; optionally safely extract and audit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parts-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--extract-to', type=Path)
    p.add_argument('--verify-sources', action='store_true')
    a = p.parse_args()
    d = a.parts_dir.resolve()
    m = json.loads((d/'ARCHIVE_PARTS.json').read_text())
    a.archive.parent.mkdir(parents=True, exist_ok=True)
    total_hash = hashlib.sha256(); total_bytes = 0
    # Refuse to silently overwrite an existing handoff/archive.
    with a.archive.open('xb') as output:
        for rec in m['parts']:
            name = rec['filename']
            assert Path(name).name == name
            part = d/name; h = hashlib.sha256(); count = 0
            with part.open('rb') as f:
                for b in iter(lambda:f.read(8<<20), b''):
                    output.write(b); h.update(b); total_hash.update(b)
                    count += len(b); total_bytes += len(b)
            assert count == rec['bytes'] and h.hexdigest() == rec['sha256'], name
    assert total_bytes == m['archive_bytes'] and total_hash.hexdigest() == m['archive_sha256']
    print(json.dumps({'archive_verified':str(a.archive.resolve()), 'sha256':total_hash.hexdigest(), 'bytes':total_bytes}), flush=True)
    if a.extract_to:
        a.extract_to.mkdir(parents=True, exist_ok=True)
        target = a.extract_to.resolve()
        assert not (target/m['root_directory']).exists(), 'Extract into a fresh directory to preserve prior copies'
        with tarfile.open(a.archive, 'r:gz') as tar:
            for member in tar.getmembers():
                parts = PurePosixPath(member.name).parts
                assert parts and parts[0] == m['root_directory']
                assert not member.name.startswith('/') and '..' not in parts
                assert member.isfile() or member.isdir(), member.name
            tar.extractall(target)
        root = target/m['root_directory']
        print(json.dumps({'extracted':str(root)}), flush=True)
        if a.verify_sources:
            subprocess.run([sys.executable, '-B', str(root/'verify_sources.py'), '--root', str(root)], check=True)
    elif a.verify_sources:
        p.error('--verify-sources requires --extract-to')


if __name__ == '__main__':
    main()
