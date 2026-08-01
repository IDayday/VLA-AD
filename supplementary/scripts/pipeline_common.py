#!/usr/bin/env python3
"""Shared, dependency-light helpers for the supplementary data pipeline.

This module deliberately contains no paper claims or experiment constants.  In
particular, ``ANALYSIS_SEED`` controls only deterministic resampling/plotting;
it must never be interpreted as a training or evaluation seed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_SEED = 2027

CANONICAL_SCHEMA: dict[str, str] = {
    "scene_id": "string",
    "split": "string",
    "benchmark": "string",
    "method": "string",
    "stage": "string",
    "variant": "string",
    "seed": "Int64",
    "round": "Int64",
    "sample_index": "Int64",
    "aggregate_score": "Float64",
    "NC": "Float64",
    "DAC": "Float64",
    "DDC": "Float64",
    "TLC": "Float64",
    "TTC": "Float64",
    "EP": "Float64",
    "C": "Float64",
    "LK": "Float64",
    "HC": "Float64",
    "EC": "Float64",
    "feasible": "boolean",
    "zero_score": "boolean",
    "initial_failure": "boolean",
    "recovered": "boolean",
    "new_failure": "boolean",
    "advantage": "Float64",
    "positive_credit": "boolean",
    "pareto_front": "boolean",
    "reference_guard_pass": "boolean",
    "teacher_source": "string",
    "teacher_status": "string",
    "source_file": "string",
    "source_record": "string",
}

METRIC_COLUMNS = [
    "aggregate_score",
    "NC",
    "DAC",
    "DDC",
    "TLC",
    "TTC",
    "EP",
    "C",
    "LK",
    "HC",
    "EC",
]
BOOLEAN_COLUMNS = [name for name, dtype in CANONICAL_SCHEMA.items() if dtype == "boolean"]
INTEGER_COLUMNS = [name for name, dtype in CANONICAL_SCHEMA.items() if dtype == "Int64"]


def repository_root() -> Path:
    """Return the repository root for the in-tree supplementary scripts."""

    return Path(__file__).resolve().parents[2]


def supplementary_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def empty_canonical_frame(extra_columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Create a typed, zero-row canonical scene-level table."""

    frame = pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in CANONICAL_SCHEMA.items()}
    )
    for column in extra_columns or ():
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame


def parse_nullable_bool(series: pd.Series, column: str) -> pd.Series:
    """Parse common boolean encodings and reject ambiguous non-null values."""

    if str(series.dtype) == "boolean":
        return series
    mapping: dict[Any, bool] = {
        True: True,
        False: False,
        1: True,
        0: False,
        1.0: True,
        0.0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        key = value.strip().lower() if isinstance(value, str) else value
        if key not in mapping:
            raise ValueError(f"column {column!r} contains non-boolean value {value!r}")
        return mapping[key]

    return series.map(convert).astype("boolean")


def coerce_canonical_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Add missing canonical columns and enforce nullable canonical dtypes."""

    result = frame.copy()
    for column, dtype in CANONICAL_SCHEMA.items():
        if column not in result:
            result[column] = pd.NA
        if dtype == "boolean":
            result[column] = parse_nullable_bool(result[column], column)
        elif dtype in {"Float64", "Int64"}:
            numeric = pd.to_numeric(result[column], errors="coerce")
            invalid = result[column].notna() & numeric.isna()
            if invalid.any():
                examples = result.loc[invalid, column].astype(str).head(3).tolist()
                raise ValueError(f"column {column!r} contains non-numeric values: {examples}")
            if dtype == "Int64":
                non_integral = numeric.notna() & (numeric % 1 != 0)
                if non_integral.any():
                    raise ValueError(f"column {column!r} contains non-integral values")
            result[column] = numeric.astype(dtype)
        else:
            result[column] = result[column].astype("string")

    canonical = list(CANONICAL_SCHEMA)
    extras = sorted(column for column in result.columns if column not in CANONICAL_SCHEMA)
    return result.loc[:, canonical + extras]


def read_table(path: Path, *, allow_pickle: bool = False) -> pd.DataFrame:
    """Read a supported tabular format with explicit pickle opt-in."""

    suffix = path.suffix.lower()
    if not path.is_file():
        raise FileNotFoundError(f"input does not exist: {path}")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.json_normalize(payload, sep=".")
        if isinstance(payload, dict):
            for key in ("records", "results", "data", "scenes", "samples"):
                value = payload.get(key)
                if isinstance(value, list):
                    return pd.json_normalize(value, sep=".")
            # A mapping keyed by scene id is common in evaluator exports.  Only
            # accept it when every value is itself a record.
            if payload and all(isinstance(value, dict) for value in payload.values()):
                rows = []
                for key, value in payload.items():
                    row = dict(value)
                    row.setdefault("scene_id", key)
                    rows.append(row)
                return pd.json_normalize(rows, sep=".")
            if not payload:
                return pd.DataFrame()
            return pd.json_normalize([payload], sep=".")
        raise ValueError(f"JSON root must be a list or object: {path}")
    if suffix in {".pkl", ".pickle"}:
        if not allow_pickle:
            raise ValueError(
                f"refusing to unpickle {path}; pass --allow-pickle only for trusted files"
            )
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, (list, dict)):
            return pd.json_normalize(obj, sep=".")
        raise ValueError(f"pickle does not contain a tabular object: {path}")
    raise ValueError(f"unsupported input extension {suffix!r}: {path}")


def write_dataframe(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    """Write a dataframe to CSV and Parquet without mutating the input."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    try:
        frame.to_parquet(parquet_path, index=False, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install it before running the pipeline."
        ) from exc


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def relative_display(path: Path, root: Path | None = None) -> str:
    """Prefer repository-relative paths in manifests and reports."""

    root = (root or repository_root()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def latex_escape(value: Any) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return "--"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def dataframe_to_latex(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    column_headers: Mapping[str, str] | None = None,
    alignment: str | None = None,
) -> str:
    """Render a compact booktabs table without a Jinja dependency."""

    headers = dict(column_headers or {})
    columns = list(frame.columns)
    alignment = alignment or ("l" + "r" * max(0, len(columns) - 1))
    if len(alignment) != len(columns):
        raise ValueError("LaTeX alignment length must equal the number of columns")
    if not re.fullmatch(r"[A-Za-z0-9:._/-]+", label):
        raise ValueError(f"unsafe LaTeX label: {label!r}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(headers.get(column, column)) for column in columns)
        + r" \\",
        r"\midrule",
    ]
    if frame.empty:
        lines.append(
            rf"\multicolumn{{{len(columns)}}}{{c}}{{No verified records available.}} \\"
        )
    else:
        for row in frame.itertuples(index=False, name=None):
            lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def git_revision(root: Path | None = None) -> dict[str, Any]:
    root = (root or repository_root()).resolve()
    result: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        result = {"commit": commit, "dirty": bool(dirty_proc.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def environment_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "pyarrow", "matplotlib", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def command_display(argv: Sequence[str] | None = None) -> str:
    return shlex.join(list(argv or sys.argv))


def build_provenance(
    *,
    seed: int,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    status: str,
    notes: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or repository_root()).resolve()

    def records(paths: Iterable[Path]) -> list[dict[str, Any]]:
        result = []
        for path in paths:
            item: dict[str, Any] = {"path": relative_display(path, root)}
            if path.is_file():
                item.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
            else:
                item.update({"bytes": None, "sha256": None})
            result.append(item)
        return result

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "analysis_seed": seed,
        "analysis_seed_scope": "resampling and plotting only; not a training seed",
        "command": command_display(),
        "git": git_revision(root),
        "environment": environment_versions(),
        "inputs": records(inputs),
        "outputs": records(outputs),
        "notes": list(notes or ()),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None or value is pd.NA:
            return ""
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(cell(value) for value in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])
