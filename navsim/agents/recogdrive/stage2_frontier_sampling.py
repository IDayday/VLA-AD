from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pytorch_lightning as pl
import torch

from navsim.agents.recogdrive.stage3_frontier_curriculum import DistributedFrontierSampler


@dataclass(frozen=True)
class Stage2FrontierWeights:
    weights: torch.Tensor
    eligible_mask: torch.Tensor
    diagnostics: Mapping[str, float]


def load_stage2_frontier_weights(
    index_path: str | Path,
    dataset_tokens: Sequence[str],
    *,
    uniform_ratio: float,
    priority_exponent: float,
) -> Stage2FrontierWeights:
    """Build a global frontier/uniform scene mixture without per-scene quotas."""

    if not 0.0 < float(uniform_ratio) < 1.0:
        raise ValueError("Stage2 frontier uniform_ratio must be in (0, 1).")
    if float(priority_exponent) < 0.0:
        raise ValueError("Stage2 frontier priority_exponent must be non-negative.")
    tokens = [str(token) for token in dataset_tokens]
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Stage2 frontier sampling requires non-empty, unique dataset tokens.")

    path = Path(index_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Stage2 frontier scene index does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported Stage2 frontier scene index version: {payload.get('version')!r}.")
    if payload.get("capacity_semantics") != "observed_selected_support_not_scene_intrinsic":
        raise ValueError("Stage2 frontier scene index has incompatible capacity semantics.")
    if not bool(payload.get("no_per_scene_candidate_quota", False)):
        raise ValueError("Stage2 frontier scene index must explicitly disable per-scene candidate quotas.")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("Stage2 frontier scene index records must be a list.")
    records: dict[str, Mapping[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise TypeError("Stage2 frontier scene index record must be a mapping.")
        token = str(raw_record.get("token", ""))
        if not token or token in records:
            raise ValueError(f"Stage2 frontier scene index contains an empty or duplicate token: {token!r}.")
        records[token] = raw_record

    missing = [token for token in tokens if token not in records]
    if missing:
        raise KeyError(
            "Stage2 frontier scene index does not cover the training dataset: "
            f"missing={len(missing)}, first={missing[0]!r}."
        )
    eligible = torch.tensor(
        [bool(records[token].get("frontier_eligible", False)) for token in tokens],
        dtype=torch.bool,
    )
    if not bool(eligible.any()):
        raise ValueError("Stage2 frontier scene sampling is enabled but the index has no eligible scene.")
    raw_priority = torch.tensor(
        [
            float(records[token].get("priority", 0.0)) if bool(eligible[index]) else 0.0
            for index, token in enumerate(tokens)
        ],
        dtype=torch.float64,
    )
    if not torch.isfinite(raw_priority).all() or bool((raw_priority < 0.0).any()):
        raise ValueError("Stage2 frontier scene priorities must be finite and non-negative.")
    raw_priority = torch.where(eligible, raw_priority.clamp_min(1e-6), raw_priority)
    frontier = torch.where(
        eligible,
        raw_priority.pow(float(priority_exponent)),
        torch.zeros_like(raw_priority),
    )
    frontier = frontier / frontier.sum().clamp_min(torch.finfo(frontier.dtype).eps)
    uniform = torch.full_like(frontier, 1.0 / len(tokens))
    mixed = float(uniform_ratio) * uniform + (1.0 - float(uniform_ratio)) * frontier
    eligible_ratio = float(eligible.float().mean())
    expected_eligible_ratio = (1.0 - float(uniform_ratio)) + float(uniform_ratio) * eligible_ratio
    diagnostics = {
        "stage2_frontier_index_scene_count": float(len(records)),
        "stage2_frontier_dataset_scene_count": float(len(tokens)),
        "stage2_frontier_eligible_scene_ratio": eligible_ratio,
        "stage2_frontier_expected_sampled_scene_ratio": expected_eligible_ratio,
        "stage2_frontier_uniform_ratio": float(uniform_ratio),
        "stage2_frontier_sampler_entropy": float(-(mixed * mixed.clamp_min(1e-12).log()).sum()),
    }
    return Stage2FrontierWeights(weights=mixed, eligible_mask=eligible, diagnostics=diagnostics)


class DistributedStage2FrontierSampler(DistributedFrontierSampler):
    def __init__(self, *args: Any, eligible_mask: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eligible_mask = torch.as_tensor(eligible_mask, dtype=torch.bool, device="cpu").flatten()
        if self.eligible_mask.shape != (self.dataset_size,):
            raise ValueError(
                f"eligible_mask must have shape [{self.dataset_size}], got {tuple(self.eligible_mask.shape)}."
            )
        self.last_frontier_scene_ratio_actual = 0.0

    def __iter__(self) -> Iterator[int]:
        indices = list(super().__iter__())
        if indices:
            self.last_frontier_scene_ratio_actual = float(
                self.eligible_mask[torch.tensor(indices, dtype=torch.long)].float().mean()
            )
        else:
            self.last_frontier_scene_ratio_actual = 0.0
        return iter(indices)


class Stage2FrontierSamplingCallback(pl.Callback):
    def __init__(
        self,
        sampler: DistributedStage2FrontierSampler,
        diagnostics: Mapping[str, float],
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.diagnostics = dict(diagnostics)

    @property
    def state_key(self) -> str:
        return "Stage2FrontierSamplingCallback"

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        self.sampler.set_epoch(int(trainer.current_epoch))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer
        values = {
            **self.diagnostics,
            "stage2_frontier_uniform_ratio_actual": self.sampler.last_uniform_ratio_actual,
            "stage2_frontier_sampled_scene_ratio_actual": self.sampler.last_frontier_scene_ratio_actual,
        }
        for key, value in values.items():
            pl_module.log(
                f"train/{key}",
                float(value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
