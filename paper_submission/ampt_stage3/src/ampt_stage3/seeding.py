"""Shard-independent deterministic inference seeds."""

from __future__ import annotations

import hashlib


def seed_for_token(base_seed: int, token: str) -> int:
    if int(base_seed) < 0 or not str(token):
        raise ValueError("base_seed must be non-negative and token non-empty")
    digest = hashlib.blake2b(
        f"{int(base_seed)}:{str(token)}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)
