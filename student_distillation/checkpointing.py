"""Atomic complete checkpoints with independent nested snapshots and all-rank RNG."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import uuid
import numpy as np
import torch
import torch.distributed as dist


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    return clone_nested({"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch_cpu": torch.get_rng_state(),
                         "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def all_rank_rng(env):
    current = rng_state()
    if env.world_size == 1:
        return [current]
    gathered = [None] * env.world_size
    dist.all_gather_object(gathered, current)
    return gathered


def save_checkpoint(path, *, adapter, optimizer, scheduler, progress, sampler, identities,
                    config, calibration, parameter_groups, env):
    from .distributed import unwrap, barrier
    random_states = all_rank_rng(env)
    if env.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = Path(str(path) + ".COMPLETE.json")
        marker.unlink(missing_ok=True)
        state = clone_nested({"schema": 1, "adapter": unwrap(adapter).state_dict(),
                              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                              "progress": progress, "sampler": sampler, "identities": identities,
                              "config": config, "calibration": calibration,
                              "parameter_groups": parameter_groups, "rng_by_rank": random_states,
                              "world_size": env.world_size})
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        atomic_json(marker, {"status": "COMPLETE", "sha256": sha256(path),
                             "global_step": progress["global_step"], "progress": progress})
    barrier()


def read_checkpoint(path, expected_identities=None):
    path = Path(path)
    marker = Path(str(path) + ".COMPLETE.json")
    if not marker.exists():
        raise ValueError(f"Checkpoint has no completion marker: {path}")
    info = json.loads(marker.read_text())
    if info.get("status") != "COMPLETE":
        raise ValueError(f"Checkpoint marker is not complete: {path}")
    if sha256(path) != info["sha256"]:
        raise ValueError(f"Checkpoint SHA mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != 1 or state.get("progress") != info.get("progress") or state["progress"]["global_step"] != info.get("global_step"):
        raise ValueError(f"Checkpoint payload/progress does not match its completion marker: {path}")
    if len(state.get("rng_by_rank", [])) != state.get("world_size"):
        raise ValueError("Checkpoint is missing one or more ranks' RNG states")
    if expected_identities is not None and state["identities"] != expected_identities:
        raise ValueError("Checkpoint protocol/cache/teacher/code identity mismatch")
    return state


def load_checkpoint(path, adapter, optimizer, scheduler, env, expected_identities=None):
    from .distributed import unwrap
    state = read_checkpoint(path, expected_identities)
    if state["world_size"] != env.world_size:
        raise ValueError("Exact RNG replay requires the checkpoint world size")
    unwrap(adapter).load_state_dict(clone_nested(state["adapter"]), strict=True)
    optimizer.load_state_dict(clone_nested(state["optimizer"]))
    scheduler.load_state_dict(clone_nested(state["scheduler"]))
    restore_rng(state["rng_by_rank"][env.rank])
    return state


def latest_complete(directory):
    directory = Path(directory)
    candidates, invalid = [], []
    for marker in Path(directory).glob("*.pt.COMPLETE.json"):
        try:
            info = json.loads(marker.read_text())
            checkpoint = Path(str(marker).removesuffix(".COMPLETE.json"))
            candidates.append((info["global_step"], checkpoint.stat().st_mtime_ns, checkpoint))
        except (OSError, ValueError, KeyError) as error:
            invalid.append({"marker": str(marker), "error": repr(error)})
    for _, _, checkpoint in sorted(candidates, reverse=True):
        try:
            read_checkpoint(checkpoint)
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            invalid.append({"checkpoint": str(checkpoint), "error": repr(error)})
            continue
        if invalid:
            atomic_json(directory / "RECOVERY_SCAN.json", {"invalid_candidates": invalid, "selected": str(checkpoint)})
        return checkpoint
    if invalid:
        atomic_json(directory / "RECOVERY_SCAN.json", {"invalid_candidates": invalid, "selected": None})
        raise ValueError("No valid complete checkpoint remains; refusing to restart this run from scratch")
    return None
