from __future__ import annotations

import hashlib
import json
import lzma
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


REFERENCE_COMPONENTS = ("scalar", "ep", "ttc", "quality", "nc", "dac", "ddc")
SOURCE_CODES = {"gt": 1, "stage2": 2}


@dataclass
class ReferenceBatch:
    scalar: torch.Tensor
    ep: torch.Tensor
    ttc: torch.Tensor
    quality: torch.Tensor
    nc: torch.Tensor
    dac: torch.Tensor
    ddc: torch.Tensor
    tlc: Optional[torch.Tensor]
    gt_ddc: torch.Tensor
    selected_source_code: torch.Tensor
    fallback_mask: torch.Tensor


def select_coherent_reference(
    gt: Mapping[str, Any],
    stage2: Mapping[str, Any],
    *,
    gt_feasible: bool,
    stage2_feasible: bool,
) -> Dict[str, Any]:
    """Selects one whole reference row; components are never mixed by metric."""
    for source_name, row in (("gt", gt), ("stage2", stage2)):
        missing = [key for key in REFERENCE_COMPONENTS if key not in row]
        if missing:
            raise KeyError(f"{source_name} reference is missing components {missing}.")

    if gt_feasible and stage2_feasible:
        source = "gt" if float(gt["scalar"]) >= float(stage2["scalar"]) else "stage2"
        fallback = False
    elif gt_feasible:
        source, fallback = "gt", False
    elif stage2_feasible:
        source, fallback = "stage2", False
    else:
        source = "gt" if float(gt["scalar"]) >= float(stage2["scalar"]) else "stage2"
        fallback = True

    selected = dict(gt if source == "gt" else stage2)
    return {
        "selected": selected,
        "selected_source": source,
        "selected_source_code": SOURCE_CODES[source],
        "reference_fallback": fallback,
        "gt_ddc": float(gt["ddc"]),
    }


class Stage3ReferenceCache:
    VERSION = 1

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        benchmark: Optional[str] = None,
    ) -> None:
        if payload is None:
            if path is None or not str(path):
                raise ValueError("LFP-GRPO requires a non-empty reference_cache_path.")
            self.path = Path(path).expanduser().resolve()
            payload = self._load(self.path)
        else:
            self.path = Path(path).expanduser().resolve() if path else None
        if not isinstance(payload, Mapping):
            raise TypeError("Reference cache payload must be a mapping.")
        self.metadata = dict(payload.get("metadata", {}))
        records = payload.get("records")
        if not isinstance(records, Mapping):
            raise KeyError("Reference cache payload must contain a 'records' mapping.")
        self.records = {str(token): self._validate_record(str(token), record) for token, record in records.items()}
        if not self.records:
            raise ValueError("Reference cache contains no scene records.")
        cache_benchmark = str(self.metadata.get("benchmark", ""))
        if benchmark and cache_benchmark and cache_benchmark != str(benchmark):
            raise ValueError(
                f"Reference cache benchmark {cache_benchmark!r} does not match configured {benchmark!r}."
            )
        effective_benchmark = str(benchmark or cache_benchmark)
        if effective_benchmark == "navsim_v2":
            missing_tlc = [
                token
                for token, record in self.records.items()
                if record["selected"].get("tlc") is None
            ]
            if missing_tlc:
                raise KeyError(
                    "NAVSIM v2 reference rows require official filtered TLC; missing for "
                    f"{len(missing_tlc)} token(s), first={missing_tlc[0]!r}."
                )
        canonical = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), default=str)
        self.metadata_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _load(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"LFP reference cache not found: {path}")
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        if path.suffix == ".jsonl":
            metadata: Dict[str, Any] = {}
            records: Dict[str, Any] = {}
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "metadata" in row and len(row) == 1:
                        metadata = dict(row["metadata"])
                    else:
                        token = str(row.pop("token"))
                        records[token] = row
            return {"metadata": metadata, "records": records}
        if path.suffixes[-2:] == [".pkl", ".xz"]:
            with lzma.open(path, "rb") as handle:
                return pickle.load(handle)
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _validate_record(token: str, record: Any) -> Dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError(f"Reference record {token!r} must be a mapping.")
        selected = record.get("selected", record)
        if not isinstance(selected, Mapping):
            raise TypeError(f"Reference record {token!r} selected row must be a mapping.")
        missing = [key for key in REFERENCE_COMPONENTS if key not in selected]
        if missing:
            raise KeyError(f"Reference record {token!r} is missing selected components {missing}.")
        if "gt_ddc" not in record:
            raise KeyError(f"Reference record {token!r} is missing gt_ddc.")
        numeric_values = {key: selected[key] for key in REFERENCE_COMPONENTS}
        numeric_values["gt_ddc"] = record["gt_ddc"]
        if selected.get("tlc") is not None:
            numeric_values["tlc"] = selected["tlc"]
        for key, value in numeric_values.items():
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Reference record {token!r} component {key!r} must be numeric."
                ) from error
            if not finite:
                raise ValueError(
                    f"Reference record {token!r} component {key!r} must be finite, got {value!r}."
                )
        source = str(record.get("selected_source", ""))
        source_code = int(record.get("selected_source_code", SOURCE_CODES.get(source, 0)))
        if source_code not in SOURCE_CODES.values():
            raise ValueError(f"Reference record {token!r} has invalid selected source {source!r}/{source_code}.")
        out = dict(record)
        out["selected"] = dict(selected)
        out["selected_source_code"] = source_code
        out["reference_fallback"] = bool(record.get("reference_fallback", record.get("fallback", False)))
        return out

    @staticmethod
    def _tensor(values: Iterable[Any], device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([float(value) for value in values], device=device, dtype=dtype)

    def get(
        self,
        tokens: list[str],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ReferenceBatch:
        normalized_tokens = [str(token) for token in tokens]
        missing = [token for token in normalized_tokens if token not in self.records]
        if missing:
            preview = ", ".join(repr(token) for token in missing[:8])
            raise KeyError(
                f"LFP reference cache is missing {len(missing)} requested token(s): {preview}. "
                "Dynamic/random reference fallback is forbidden."
            )
        rows = [self.records[token] for token in normalized_tokens]
        selected = [row["selected"] for row in rows]

        def component(name: str) -> torch.Tensor:
            return self._tensor((row[name] for row in selected), device, dtype)

        has_tlc = ["tlc" in row and row["tlc"] is not None for row in selected]
        if any(has_tlc) and not all(has_tlc):
            raise ValueError("Reference cache mixes rows with and without TLC in one batch.")
        tlc = component("tlc") if all(has_tlc) else None
        return ReferenceBatch(
            scalar=component("scalar"),
            ep=component("ep"),
            ttc=component("ttc"),
            quality=component("quality"),
            nc=component("nc"),
            dac=component("dac"),
            ddc=component("ddc"),
            tlc=tlc,
            gt_ddc=self._tensor((row["gt_ddc"] for row in rows), device, dtype),
            selected_source_code=torch.tensor(
                [int(row["selected_source_code"]) for row in rows],
                device=device,
                dtype=torch.long,
            ),
            fallback_mask=torch.tensor(
                [bool(row["reference_fallback"]) for row in rows],
                device=device,
                dtype=torch.bool,
            ),
        )
