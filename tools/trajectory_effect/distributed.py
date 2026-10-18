"""Exact global-batch gradient accumulation across independent native forwards."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist


def initialize():
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    return rank,world


def sum_values(values,world):
    value=torch.as_tensor(values,device='cuda',dtype=torch.float64)
    if world>1:dist.all_reduce(value)
    return value


def average_gradients(parameters,world):
    if world==1:return
    # A single deterministic FP32 bucket avoids hundreds of tiny collectives.
    active=torch.tensor([p.grad is not None for p in parameters],device='cuda',dtype=torch.int32)
    dist.all_reduce(active,op=dist.ReduceOp.MAX)
    parameters=[p for p,a in zip(parameters,active.tolist()) if a]
    gradients=[p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    flat=torch.cat([g.reshape(-1) for g in gradients]);dist.all_reduce(flat);flat.div_(world)
    offset=0
    for p in parameters:
        n=p.numel()
        if p.grad is None:p.grad=torch.empty_like(p)
        p.grad.copy_(flat[offset:offset+n].view_as(p));offset+=n


def finish(world):
    if world>1:dist.barrier();dist.destroy_process_group()
