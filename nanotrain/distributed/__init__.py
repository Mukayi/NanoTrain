"""Distributed process-group and communication helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed state derived from torchrun environment variables."""

    is_distributed: bool
    is_ddp: bool
    is_tensor_parallel: bool
    rank: int
    local_rank: int
    world_size: int
    tp_size: int
    tp_rank: int
    backend: str
    device: str
    is_master: bool
    tp_group: Any | None

    @classmethod
    def initialize(
        cls,
        *,
        backend: str,
        requested_device: str,
        tp_size: int = 1,
    ) -> DistributedContext:
        if tp_size < 1:
            raise ValueError("distributed.tp_size must be >= 1")

        rank = int(os.environ.get("RANK", "-1"))
        is_distributed = rank != -1
        if not is_distributed:
            if tp_size != 1:
                raise RuntimeError("Tensor Parallel requires torchrun when distributed.tp_size > 1")
            return cls(
                is_distributed=False,
                is_ddp=False,
                is_tensor_parallel=False,
                rank=0,
                local_rank=0,
                world_size=1,
                tp_size=1,
                tp_rank=0,
                backend=backend,
                device=requested_device,
                is_master=True,
                tp_group=None,
            )

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_tensor_parallel = tp_size > 1
        if is_tensor_parallel and world_size != tp_size:
            raise RuntimeError(
                "NanoTrain currently supports pure Tensor Parallel only, so "
                f"WORLD_SIZE must equal distributed.tp_size ({world_size} != {tp_size})"
            )
        is_ddp = not is_tensor_parallel
        device = requested_device
        if "cuda" in requested_device:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Distributed training requested a CUDA device, but CUDA is not available. "
                    "Use runtime.device=cpu with distributed.backend=gloo for CPU tests."
                )
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        return cls(
            is_distributed=True,
            is_ddp=is_ddp,
            is_tensor_parallel=is_tensor_parallel,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            tp_size=tp_size,
            tp_rank=rank if is_tensor_parallel else 0,
            backend=backend,
            device=device,
            is_master=(rank == 0),
            tp_group=dist.group.WORLD if is_tensor_parallel else None,
        )

    @property
    def seed_offset(self) -> int:
        return self.rank if self.is_ddp else 0

    def destroy(self) -> None:
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["DistributedContext"]
