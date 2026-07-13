from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv
import lzma
import pickle

from navsim.planning.metric_caching.metric_cache import MetricCache


XZ_MAGIC = b"\xfd7zXZ\x00"


def load_metric_cache_auto(path: Path) -> MetricCache:
    """Load a metric cache from either original lzma-compressed or plain pickle storage."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(len(XZ_MAGIC))
    opener = lzma.open if magic == XZ_MAGIC else open
    with opener(path, "rb") as f:
        metric_cache: MetricCache = pickle.load(f)
    return metric_cache


class FastMetricCacheLoader:
    """Metric cache loader for opt-in plain-pickle mirrors.

    The mirror keeps the same MetricCache objects and token mapping, but avoids lzma decompression
    in hot evaluation paths. It is intentionally separate from the default MetricCacheLoader.
    """

    def __init__(self, cache_path: Path):
        self.metric_cache_paths = self._load_metric_cache_paths(Path(cache_path))

    @staticmethod
    def _token_from_path(path: str) -> str:
        return Path(path).parent.name

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        metadata_dir = cache_path / "metadata"
        if not metadata_dir.is_dir():
            raise FileNotFoundError(f"Fast metric cache metadata directory not found: {metadata_dir}")
        metadata_files = sorted(metadata_dir.glob("*.csv"))
        if not metadata_files:
            raise FileNotFoundError(f"No fast metric cache metadata CSV files found under {metadata_dir}")

        output: Dict[str, Path] = {}
        for metadata_file in metadata_files:
            with metadata_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                for row in reader:
                    file_name = row.get("file_name") or row.get("path")
                    if not file_name:
                        continue
                    token = row.get("token") or self._token_from_path(file_name)
                    path = Path(file_name)
                    if not path.is_absolute():
                        path = cache_path / path
                    output[str(token)] = path
        if not output:
            raise FileNotFoundError(f"No metric cache paths found in metadata under {metadata_dir}")
        return output

    @property
    def tokens(self) -> List[str]:
        return list(self.metric_cache_paths.keys())

    def __len__(self) -> int:
        return len(self.metric_cache_paths)

    def get_from_token(self, token: str) -> MetricCache:
        return load_metric_cache_auto(self.metric_cache_paths[token])
