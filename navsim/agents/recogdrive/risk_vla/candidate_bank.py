from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    strategy_name: str
    source: str
    weight: float = 1.0
    checkpoint_path: Optional[str] = None
    deterministic: bool = True


DEFAULT_CANDIDATE_SPECS = [
    CandidateSpec("base_deterministic", "base", "base"),
    CandidateSpec("base_stochastic", "base", "stochastic", deterministic=False),
    CandidateSpec("bit_path_intent", "path_intent", "bit"),
    CandidateSpec("d5_conservative_bit", "d5_conservative", "d5"),
    CandidateSpec("risk_vla_v1_router", "risk_vla_v1", "risk_vla_v1"),
    CandidateSpec("risk_vla_v2_heuristic", "risk_vla_v2_heuristic", "risk_vla_v2"),
    CandidateSpec("interaction_conservative", "interaction", "interaction"),
    CandidateSpec("progress_recovery", "progress", "progress"),
    CandidateSpec("comfort_stabilized", "comfort", "comfort"),
    CandidateSpec("learned_residual", "learned_residual", "learned_residual"),
]


class CandidateBank:
    def __init__(self, horizon: int = 8, action_dim: int = 3, specs: Optional[Sequence[CandidateSpec]] = None) -> None:
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.specs = list(specs or DEFAULT_CANDIDATE_SPECS)

    def _fit(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2] != self.horizon:
            if value.shape[-2] > self.horizon:
                value = value[..., : self.horizon, :]
            else:
                pad = value.new_zeros(*value.shape[:-2], self.horizon - value.shape[-2], value.shape[-1])
                value = torch.cat([value, pad], dim=-2)
        if value.shape[-1] != self.action_dim:
            if value.shape[-1] > self.action_dim:
                value = value[..., : self.action_dim]
            else:
                value = torch.cat([value, value.new_zeros(*value.shape[:-1], self.action_dim - value.shape[-1])], dim=-1)
        return value

    def generate(
        self,
        base_trajectory: torch.Tensor,
        bit_trajectory: Optional[torch.Tensor] = None,
        d5_trajectory: Optional[torch.Tensor] = None,
        risk_vla_trajectory: Optional[torch.Tensor] = None,
        risk_vla_v1_trajectory: Optional[torch.Tensor] = None,
        risk_vla_v2_trajectory: Optional[torch.Tensor] = None,
        learned_residual: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        noise_scale: float = 0.05,
        seed: int = 0,
    ) -> tuple[torch.Tensor, List[Dict[str, object]]]:
        base = self._fit(base_trajectory)
        batch = base.shape[0]
        generator = torch.Generator(device=base.device)
        generator.manual_seed(int(seed))
        candidates: List[torch.Tensor] = []
        metadata: List[Dict[str, object]] = []
        specs = self.specs[: k or len(self.specs)]
        for idx, spec in enumerate(specs):
            if spec.source == "base":
                traj = base
            elif spec.source == "stochastic":
                traj = base + torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype) * float(noise_scale)
            elif spec.source == "bit" and bit_trajectory is not None:
                traj = self._fit(bit_trajectory)
            elif spec.source == "d5" and d5_trajectory is not None:
                traj = self._fit(d5_trajectory)
            elif spec.source == "risk_vla_v1" and risk_vla_v1_trajectory is not None:
                traj = self._fit(risk_vla_v1_trajectory)
            elif spec.source == "risk_vla_v2" and risk_vla_v2_trajectory is not None:
                traj = self._fit(risk_vla_v2_trajectory)
            elif spec.source == "risk_vla" and risk_vla_trajectory is not None:
                traj = self._fit(risk_vla_trajectory)
            elif spec.source == "interaction":
                traj = base.clone()
                traj[:, : max(1, self.horizon // 3), 0] *= 0.85
            elif spec.source == "progress":
                traj = base.clone()
                ramp = torch.linspace(0.0, 0.4, steps=self.horizon, device=base.device, dtype=base.dtype).view(1, self.horizon)
                traj[..., 0] = traj[..., 0] + ramp
            elif spec.source == "comfort":
                traj = base.clone()
                traj[:, 1:, :] = 0.7 * traj[:, 1:, :] + 0.3 * traj[:, :-1, :]
            elif spec.source == "learned_residual" and learned_residual is not None:
                traj = base + self._fit(learned_residual)
            else:
                traj = base
            candidates.append(traj)
            metadata.append(
                {
                    "candidate_id": idx,
                    "name": spec.name,
                    "strategy_name": spec.strategy_name,
                    "source": spec.source,
                    "checkpoint_path": spec.checkpoint_path,
                    "seed": int(seed),
                    "deterministic": bool(spec.deterministic and spec.source != "stochastic"),
                }
            )
        return torch.stack(candidates, dim=1), metadata


def _default_config_hash(metadata: Sequence[Dict[str, object]]) -> str:
    payload = json.dumps(list(metadata), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def save_candidate_cache(
    path: Path,
    sample_tokens: Sequence[str],
    trajectories: torch.Tensor,
    metadata: Sequence[Dict[str, object]],
    *,
    split: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "candidate_trajectories.npz", sample_tokens=np.array(list(sample_tokens)), trajectories=trajectories.detach().cpu().numpy())
    config_hash = config_hash or _default_config_hash(metadata)
    with (path / "candidate_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            enriched = {**row, "split": split, "config_hash": config_hash}
            handle.write(json.dumps(enriched, sort_keys=True))
            handle.write("\n")
    with (path / "candidate_records.jsonl").open("w", encoding="utf-8") as handle:
        for token in sample_tokens:
            for row in metadata:
                record = {
                    "token": str(token),
                    "sample_token": str(token),
                    "split": split,
                    "candidate_id": row.get("candidate_id"),
                    "strategy_name": row.get("strategy_name"),
                    "source": row.get("source"),
                    "checkpoint_path": row.get("checkpoint_path"),
                    "seed": row.get("seed"),
                    "deterministic": row.get("deterministic"),
                    "config_hash": config_hash,
                }
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")


def load_candidate_cache(path: Path) -> tuple[list[str], torch.Tensor, list[Dict[str, object]]]:
    data = np.load(path / "candidate_trajectories.npz", allow_pickle=False)
    tokens = [str(token) for token in data["sample_tokens"].tolist()]
    trajectories = torch.from_numpy(data["trajectories"]).float()
    metadata = []
    with (path / "candidate_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return tokens, trajectories, metadata
