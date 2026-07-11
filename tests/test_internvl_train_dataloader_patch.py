from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "internvl_chat"))

from internvl.patch import train_dataloader_patch  # noqa: E402


def test_build_worker_init_fn_uses_legacy_transformers_contract(monkeypatch) -> None:
    def legacy_seed_worker(worker_id):
        return worker_id

    monkeypatch.setattr(train_dataloader_patch, "seed_worker", legacy_seed_worker)
    worker_init_fn = train_dataloader_patch.build_worker_init_fn(num_workers=4, rank=2)

    assert worker_init_fn is legacy_seed_worker
    assert worker_init_fn(3) == 3


def test_build_worker_init_fn_uses_rank_aware_transformers_contract(monkeypatch) -> None:
    def rank_aware_seed_worker(worker_id, num_workers, rank):
        return worker_id, num_workers, rank

    monkeypatch.setattr(train_dataloader_patch, "seed_worker", rank_aware_seed_worker)
    worker_init_fn = train_dataloader_patch.build_worker_init_fn(num_workers=4, rank=2)

    assert worker_init_fn(3) == (3, 4, 2)
