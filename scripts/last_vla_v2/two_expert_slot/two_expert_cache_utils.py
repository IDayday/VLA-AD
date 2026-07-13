from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from navsim.agents.recogdrive.expert_cache import iter_index


def chunk_dirs(root: Path, pattern: Optional[str] = None) -> List[Path]:
    root = Path(root)
    if (root / "index.jsonl").is_file():
        return [root]

    dirs: List[Path] = []
    seen = set()
    if pattern:
        patterns = [item.strip() for item in str(pattern).split(",") if item.strip()]
        for item in patterns:
            for path in sorted(root.glob(item)):
                if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                    dirs.append(path)
                    seen.add(path)
    else:
        for path in sorted(root.glob("shards/shard_*")):
            if path.is_dir() and (path / "index.jsonl").is_file() and path not in seen:
                dirs.append(path)
                seen.add(path)
        if not dirs:
            for index_path in sorted(root.glob("**/index.jsonl")):
                path = index_path.parent
                if path not in seen:
                    dirs.append(path)
                    seen.add(path)

    if not dirs:
        suffix = f" matching {pattern!r}" if pattern else ""
        raise FileNotFoundError(f"No indexed chunk dirs{suffix} under {root}")
    return dirs


def resolve_index_record_path(chunk_dir: Path, record: Dict[str, Any]) -> Path:
    if "path" not in record:
        raise KeyError("index record missing required 'path' field.")
    path = Path(record["path"])
    return path if path.is_absolute() else Path(chunk_dir) / path


def iter_indexed_records(
    root: Path,
    pattern: Optional[str] = None,
    max_records: Optional[int] = None,
) -> Iterable[Tuple[Path, Path, Dict[str, Any]]]:
    count = 0
    for chunk_dir in chunk_dirs(Path(root), pattern):
        for record in iter_index(chunk_dir):
            yield chunk_dir, resolve_index_record_path(chunk_dir, record), record
            count += 1
            if max_records is not None and count >= max_records:
                return


def normalize_merged_index_path(output_root: Path, shard_dir: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    path = Path(normalized["path"])
    absolute = path if path.is_absolute() else Path(shard_dir) / path
    normalized["path"] = str(absolute.relative_to(output_root))
    return normalized


def load_path_index(
    root: Path,
    *,
    pattern: Optional[str] = None,
    max_records: Optional[int] = None,
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for _, sample_path, record in iter_indexed_records(root, pattern=pattern, max_records=max_records):
        token = str(record.get("sample_token") or sample_path.stem)
        if token in mapping:
            raise ValueError(f"Duplicate sample_token in indexed cache {root}: {token}")
        mapping[token] = sample_path
    return mapping
