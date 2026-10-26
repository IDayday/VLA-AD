"""Scene-exact global batches. Only the adapter is wrapped in DDP."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import os
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Environment:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cpu"


def initialize() -> Environment:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        # A full evaluation may run longer than the default process-group timeout.
        from datetime import timedelta
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=timedelta(hours=24))
    if 64 % world:
        raise ValueError("Round1 world size must divide global batch 64")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return Environment(rank, world, local, device)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def wrap_adapter(model, env):
    if env.world_size == 1:
        return model
    from torch.nn.parallel import DistributedDataParallel
    return DistributedDataParallel(model, device_ids=[env.local_rank] if "cuda" in env.device else None,
                                   broadcast_buffers=False, find_unused_parameters=False)


def epoch_permutation(count, seed, epoch):
    payload = f"NRKD_shuffle_v1|{seed}|{epoch}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return torch.randperm(count, generator=generator).tolist()


def global_batch_slots(permutation, batch_index, rank, world_size, global_batch_size=64):
    """Return (dataset index, real sample) for equal microsteps on all ranks.

    Padded slots execute the same forward/backward graph but carry exactly zero
    weight. W/N_real is the sole backward normalization before DDP averaging.
    """
    if global_batch_size % world_size:
        raise ValueError("world size must divide the global batch")
    real = permutation[batch_index * global_batch_size:(batch_index + 1) * global_batch_size]
    if not real:
        raise IndexError("global batch beyond the epoch")
    padded = [(index, True) for index in real]
    padded += [(real[0], False)] * (global_batch_size - len(real))
    return padded[rank::world_size], len(real)


def synchronization(model, last_microstep):
    return nullcontext() if last_microstep or not hasattr(model, "no_sync") else model.no_sync()


def sum_tensor(values, env):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=env.device)
    if env.world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()
