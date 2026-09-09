from __future__ import annotations

import os

import torch
import torch.distributed as dist


def initialize_distributed() -> tuple[torch.device, int, int]:
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), rank, world_size


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    value = value.detach().clone()
    dist.all_reduce(value)
    return value / world_size


def close_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()

