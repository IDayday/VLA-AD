from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import Sampler


@dataclass
class SceneFrontierState:
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    seen_count: int = 0
    last_epoch: int = -1


class FrontierStateStore:
    def __init__(
        self,
        *,
        fast_ema: float = 0.80,
        slow_ema: float = 0.98,
        progress_weight: float = 0.50,
        priority_exponent: float = 0.50,
        uniform_ratio: float = 0.20,
        priority_eps: float = 1e-3,
        priority_quantile_cap: float = 0.95,
    ) -> None:
        self.fast_alpha = float(fast_ema)
        self.slow_alpha = float(slow_ema)
        self.progress_weight = float(progress_weight)
        self.priority_exponent = float(priority_exponent)
        self.uniform_ratio = float(uniform_ratio)
        self.priority_eps = float(priority_eps)
        self.priority_quantile_cap = float(priority_quantile_cap)
        self.states: Dict[str, SceneFrontierState] = {}
        self.last_diagnostics: Dict[str, float] = {}
        self._validate()

    def _validate(self) -> None:
        for name, value in (("fast_ema", self.fast_alpha), ("slow_ema", self.slow_alpha)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.progress_weight < 0.0 or self.priority_exponent < 0.0 or self.priority_eps <= 0.0:
            raise ValueError("Frontier progress/exponent must be non-negative and eps positive.")
        if not 0.0 <= self.uniform_ratio <= 1.0:
            raise ValueError("uniform_ratio must be in [0, 1].")
        if not 0.0 < self.priority_quantile_cap <= 1.0:
            raise ValueError("priority_quantile_cap must be in (0, 1].")

    @staticmethod
    def _epoch_mean(value: Any) -> float:
        if isinstance(value, Mapping):
            total = float(value.get("sum", 0.0))
            count = int(value.get("count", 0))
            return total / max(count, 1)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0]) / max(int(value[1]), 1)
        return float(value)

    def update_epoch(self, epoch_energy: Mapping[str, Any], epoch: int) -> None:
        for token, value in epoch_energy.items():
            energy = self._epoch_mean(value)
            if not math.isfinite(energy):
                raise ValueError(f"Non-finite frontier energy for token {token!r}: {energy}.")
            state = self.states.setdefault(str(token), SceneFrontierState())
            if state.last_epoch == int(epoch):
                raise ValueError(f"Frontier state for token {token!r} was already updated for epoch {epoch}.")
            state.fast_ema = self.fast_alpha * state.fast_ema + (1.0 - self.fast_alpha) * energy
            state.slow_ema = self.slow_alpha * state.slow_ema + (1.0 - self.slow_alpha) * energy
            state.seen_count += 1
            state.last_epoch = int(epoch)

    def _raw_priority(self, state: SceneFrontierState) -> float:
        progress = abs(float(state.fast_ema) - float(state.slow_ema))
        return max(self.priority_eps, self.priority_eps + float(state.fast_ema) + self.progress_weight * progress)

    def priorities(self, tokens: Iterable[str]) -> torch.Tensor:
        tokens = [str(token) for token in tokens]
        if not tokens:
            return torch.empty(0, dtype=torch.float64)
        seen_priorities = torch.tensor(
            [self._raw_priority(state) for state in self.states.values() if state.seen_count > 0],
            dtype=torch.float64,
        )
        unseen_value = float(torch.median(seen_priorities).item()) if seen_priorities.numel() else 1.0
        raw = torch.tensor(
            [self._raw_priority(self.states[token]) if token in self.states and self.states[token].seen_count > 0 else unseen_value for token in tokens],
            dtype=torch.float64,
        ).clamp_min(self.priority_eps)
        cap = (
            torch.quantile(seen_priorities, self.priority_quantile_cap)
            if seen_priorities.numel()
            else raw.new_tensor(max(unseen_value, self.priority_eps))
        )
        capped = raw.clamp_max(cap.clamp_min(self.priority_eps))
        frontier = capped.pow(self.priority_exponent)
        frontier = frontier / frontier.sum().clamp_min(torch.finfo(frontier.dtype).eps)
        uniform = torch.full_like(frontier, 1.0 / len(tokens))
        mixed = (1.0 - self.uniform_ratio) * frontier + self.uniform_ratio * uniform

        seen_states = [state for state in self.states.values() if state.seen_count > 0]
        fast_values = torch.tensor([state.fast_ema for state in seen_states], dtype=torch.float64)
        slow_values = torch.tensor([state.slow_ema for state in seen_states], dtype=torch.float64)
        progress_values = (fast_values - slow_values).abs()
        median = mixed.median().clamp_min(torch.finfo(mixed.dtype).eps)
        self.last_diagnostics = {
            "lfp_frontier_fast_ema_mean": float(fast_values.mean()) if fast_values.numel() else 0.0,
            "lfp_frontier_slow_ema_mean": float(slow_values.mean()) if slow_values.numel() else 0.0,
            "lfp_learning_progress_mean": float(progress_values.mean()) if progress_values.numel() else 0.0,
            "lfp_sampler_entropy": float(-(mixed * mixed.clamp_min(1e-12).log()).sum()),
            "lfp_sampler_priority_max_to_median": float(mixed.max() / median),
            "lfp_sampler_unseen_scene_ratio": float(
                sum(token not in self.states or self.states[token].seen_count == 0 for token in tokens) / len(tokens)
            ),
            "lfp_frontier_priority_cap": float(cap),
        }
        return mixed

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "config": {
                "fast_ema": self.fast_alpha,
                "slow_ema": self.slow_alpha,
                "progress_weight": self.progress_weight,
                "priority_exponent": self.priority_exponent,
                "uniform_ratio": self.uniform_ratio,
                "priority_eps": self.priority_eps,
                "priority_quantile_cap": self.priority_quantile_cap,
            },
            "states": {token: asdict(state) for token, state in self.states.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict.get("version", 0)) != 1:
            raise ValueError("Unsupported FrontierStateStore checkpoint version.")
        config = state_dict.get("config", {})
        expected = {
            "fast_ema": self.fast_alpha,
            "slow_ema": self.slow_alpha,
            "progress_weight": self.progress_weight,
            "priority_exponent": self.priority_exponent,
            "uniform_ratio": self.uniform_ratio,
            "priority_eps": self.priority_eps,
            "priority_quantile_cap": self.priority_quantile_cap,
        }
        for key, value in expected.items():
            if key in config and not math.isclose(float(config[key]), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Frontier checkpoint config mismatch for {key}: {config[key]} != {value}.")
        states = state_dict.get("states", {})
        if not isinstance(states, Mapping):
            raise TypeError("Frontier checkpoint states must be a mapping.")
        self.states = {str(token): SceneFrontierState(**dict(value)) for token, value in states.items()}


def merge_distributed_epoch_energy(local: Mapping[str, Any]) -> Dict[str, tuple[float, int]]:
    normalized: Dict[str, tuple[float, int]] = {}
    for token, value in local.items():
        if isinstance(value, Mapping):
            pair = (float(value.get("sum", 0.0)), int(value.get("count", 0)))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            pair = (float(value[0]), int(value[1]))
        else:
            pair = (float(value), 1)
        normalized[str(token)] = pair
    gathered = [normalized]
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, normalized)
    merged: Dict[str, tuple[float, int]] = {}
    for rank_map in gathered:
        for token, (energy_sum, count) in rank_map.items():
            previous_sum, previous_count = merged.get(token, (0.0, 0))
            merged[token] = (previous_sum + float(energy_sum), previous_count + int(count))
    return merged


class DistributedFrontierSampler(Sampler[int]):
    """Epoch-lagged replacement sampler with an explicit uniform draw branch."""

    def __init__(
        self,
        dataset_size: int,
        weights: Optional[torch.Tensor] = None,
        *,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        warmup_epochs: int = 1,
        uniform_ratio: float = 0.20,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError("rank must be in [0, num_replicas).")
        if not 0.0 <= float(uniform_ratio) <= 1.0:
            raise ValueError("uniform_ratio must be in [0, 1].")
        self.dataset_size = int(dataset_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = int(math.ceil(self.dataset_size / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.warmup_epochs = int(warmup_epochs)
        self.uniform_ratio = float(uniform_ratio)
        self.epoch = 0
        self.last_uniform_ratio_actual = 1.0
        self.set_weights(weights if weights is not None else torch.ones(self.dataset_size))

    def set_weights(self, weights: torch.Tensor) -> None:
        weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu").flatten()
        if weights.shape != (self.dataset_size,):
            raise ValueError(f"weights must have shape [{self.dataset_size}], got {tuple(weights.shape)}.")
        if not torch.isfinite(weights).all() or bool((weights < 0.0).any()):
            raise ValueError("sampler weights must be finite and non-negative.")
        if float(weights.sum()) <= 0.0:
            raise ValueError("sampler weights must have positive sum.")
        mixed = weights / weights.sum()
        uniform = torch.full_like(mixed, 1.0 / self.dataset_size)
        if self.uniform_ratio < 1.0:
            frontier = ((mixed - self.uniform_ratio * uniform) / (1.0 - self.uniform_ratio)).clamp_min(0.0)
            if float(frontier.sum()) <= 0.0:
                frontier = uniform
            else:
                frontier = frontier / frontier.sum()
        else:
            frontier = uniform
        self.weights = mixed
        self.frontier_weights = frontier

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.epoch < self.warmup_epochs:
            uniform_source = torch.ones(self.total_size, dtype=torch.bool)
            indices = torch.randperm(self.dataset_size, generator=generator)
            if self.total_size > self.dataset_size:
                indices = torch.cat((indices, indices[: self.total_size - self.dataset_size]))
        else:
            uniform_source = torch.rand(self.total_size, generator=generator) < self.uniform_ratio
            uniform_indices = torch.randint(self.dataset_size, (self.total_size,), generator=generator)
            frontier_indices = torch.multinomial(
                self.frontier_weights,
                self.total_size,
                replacement=True,
                generator=generator,
            )
            indices = torch.where(uniform_source, uniform_indices, frontier_indices)
        rank_positions = torch.arange(self.rank, self.total_size, self.num_replicas)
        rank_indices = indices[rank_positions]
        rank_sources = uniform_source[rank_positions]
        self.last_uniform_ratio_actual = float(rank_sources.float().mean()) if rank_sources.numel() else 0.0
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "weights": self.weights.clone(),
            "last_uniform_ratio_actual": self.last_uniform_ratio_actual,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.set_weights(torch.as_tensor(state_dict["weights"]))
        self.epoch = int(state_dict.get("epoch", 0))
        self.last_uniform_ratio_actual = float(state_dict.get("last_uniform_ratio_actual", 1.0))


class LFPFrontierCurriculumCallback(pl.Callback):
    """Merges epoch energy once, updates rank-0 frontier state, and broadcasts weights."""

    def __init__(
        self,
        *,
        sampler: DistributedFrontierSampler,
        dataset_tokens: list[str],
        cfg: Any,
        reference_cache_metadata_hash: str,
        benchmark: str,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.dataset_tokens = [str(token) for token in dataset_tokens]
        if len(self.dataset_tokens) != self.sampler.dataset_size:
            raise ValueError("dataset_tokens must follow the sampler's dataset index order.")
        self.cfg = cfg
        self.reference_cache_metadata_hash = str(reference_cache_metadata_hash)
        self.benchmark = str(benchmark)
        self.store = FrontierStateStore(
            fast_ema=float(cfg.frontier_fast_ema),
            slow_ema=float(cfg.frontier_slow_ema),
            progress_weight=float(cfg.frontier_progress_weight),
            priority_exponent=float(cfg.frontier_priority_exponent),
            uniform_ratio=float(cfg.frontier_uniform_ratio),
            priority_eps=float(cfg.frontier_priority_eps),
            priority_quantile_cap=float(cfg.frontier_priority_quantile_cap),
        )
        self._dataset_tokens_hash = self._token_hash(self.dataset_tokens)
        self._maybe_load_external_state()

    @property
    def state_key(self) -> str:
        return f"LFPFrontierCurriculumCallback[{self.benchmark}]"

    @staticmethod
    def _token_hash(tokens: list[str]) -> str:
        import hashlib

        digest = hashlib.sha256()
        for token in tokens:
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _maybe_load_external_state(self) -> None:
        state_path = str(getattr(self.cfg, "frontier_state_path", "") or "")
        if not state_path:
            return
        path = Path(state_path).expanduser()
        if not path.is_file():
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        self.load_state_dict(payload)

    def _save_external_state(self) -> None:
        state_path = str(getattr(self.cfg, "frontier_state_path", "") or "")
        if not state_path or self._rank() != 0:
            return
        path = Path(state_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temporary)
        temporary.replace(path)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.sampler.set_epoch(int(trainer.current_epoch))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        agent = getattr(pl_module, "agent", None)
        action_head = getattr(agent, "action_head", None)
        if action_head is None or not hasattr(action_head, "consume_lfp_epoch_energy"):
            raise RuntimeError("LFP frontier callback cannot find planner epoch energy accumulator.")
        merged = merge_distributed_epoch_energy(action_head.consume_lfp_epoch_energy())
        epoch = int(trainer.current_epoch)
        if self._rank() == 0:
            self.store.update_epoch(merged, epoch)
            weights = self.store.priorities(self.dataset_tokens)
        else:
            weights = torch.empty(self.sampler.dataset_size, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            device = getattr(pl_module, "device", torch.device("cpu"))
            device_weights = weights.to(device=device)
            dist.broadcast(device_weights, src=0)
            weights = device_weights.cpu()
        self.sampler.set_weights(weights)
        self._save_external_state()

        diagnostic_keys = (
            "lfp_frontier_fast_ema_mean",
            "lfp_frontier_slow_ema_mean",
            "lfp_learning_progress_mean",
            "lfp_sampler_entropy",
            "lfp_sampler_priority_max_to_median",
            "lfp_sampler_unseen_scene_ratio",
            "lfp_frontier_priority_cap",
        )
        diagnostic_tensor = torch.tensor(
            [float(self.store.last_diagnostics.get(key, 0.0)) for key in diagnostic_keys],
            device=pl_module.device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(diagnostic_tensor, src=0)
        diagnostics = {
            key: float(value)
            for key, value in zip(diagnostic_keys, diagnostic_tensor.cpu().tolist())
        }
        diagnostics["lfp_sampler_uniform_ratio_actual"] = float(self.sampler.last_uniform_ratio_actual)
        for key, value in diagnostics.items():
            pl_module.log(
                f"train/{key}",
                torch.tensor(float(value), device=pl_module.device),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "benchmark": self.benchmark,
            "reference_cache_metadata_hash": self.reference_cache_metadata_hash,
            "dataset_tokens_hash": self._dataset_tokens_hash,
            "frontier_store": self.store.state_dict(),
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict.get("version", 0)) != 1:
            raise ValueError("Unsupported LFP curriculum checkpoint version.")
        if str(state_dict.get("benchmark")) != self.benchmark:
            raise ValueError("LFP curriculum benchmark changed across resume.")
        if str(state_dict.get("reference_cache_metadata_hash")) != self.reference_cache_metadata_hash:
            raise ValueError("LFP reference cache metadata changed across resume.")
        if str(state_dict.get("dataset_tokens_hash")) != self._dataset_tokens_hash:
            raise ValueError("LFP dataset token order changed across resume.")
        self.store.load_state_dict(state_dict["frontier_store"])
        self.sampler.load_state_dict(state_dict["sampler"])

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["lfp_frontier_curriculum"] = self.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        state = checkpoint.get("lfp_frontier_curriculum")
        if state is not None:
            self.load_state_dict(state)
