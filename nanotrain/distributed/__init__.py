"""Distributed process-group and communication helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed state derived from torchrun environment variables."""

    is_ddp: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: str
    is_master: bool

    @classmethod
    def initialize(cls, *, backend: str, requested_device: str) -> DistributedContext:
        rank = int(os.environ.get("RANK", "-1"))
        is_ddp = rank != -1
        if not is_ddp:
            return cls(
                is_ddp=False,
                rank=0,
                local_rank=0,
                world_size=1,
                backend=backend,
                device=requested_device,
                is_master=True,
            )

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = requested_device
        if "cuda" in requested_device:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DDP requested a CUDA device, but CUDA is not available. "
                    "Use runtime.device=cpu with distributed.backend=gloo for CPU tests."
                )
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        return cls(
            is_ddp=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=device,
            is_master=(rank == 0),
        )

    @property
    def seed_offset(self) -> int:
        return self.rank if self.is_ddp else 0

    def destroy(self) -> None:
        if self.is_ddp and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["DistributedContext"]
