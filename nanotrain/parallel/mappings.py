"""Autograd-aware tensor-parallel communication mappings.

These mirror the core Megatron-LM tensor model parallel regions:
copy, scatter, gather, and reduce. They are no-ops for tp_size=1.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _can_communicate(group: object | None, world_size: int) -> bool:
    return world_size > 1 and group is not None and dist.is_available() and dist.is_initialized()


def _all_reduce(input_: torch.Tensor, group: object | None, world_size: int) -> torch.Tensor:
    if _can_communicate(group, world_size):
        dist.all_reduce(input_, group=group)
    return input_


class _CopyToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: object | None, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return _all_reduce(grad_output, ctx.group, ctx.world_size), None, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: object | None, world_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        return _all_reduce(input_, group, world_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad_output, None, None


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        group: object | None,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        if world_size == 1:
            return input_
        chunks = torch.chunk(input_, world_size, dim=-1)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return (
            gather_from_tensor_parallel_region(
                grad_output,
                group=ctx.group,
                world_size=ctx.world_size,
                rank=ctx.rank,
            ),
            None,
            None,
            None,
        )


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        group: object | None,
        world_size: int,
        rank: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.local_size = input_.size(-1)
        if not _can_communicate(group, world_size):
            return input_
        tensors = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensors, input_, group=group)
        return torch.cat(tensors, dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        if ctx.world_size == 1:
            return grad_output, None, None, None
        start = ctx.rank * ctx.local_size
        end = start + ctx.local_size
        return grad_output[..., start:end].contiguous(), None, None, None


def copy_to_tensor_parallel_region(
    input_: torch.Tensor,
    *,
    group: object | None,
    world_size: int,
) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(input_, group, world_size)


def reduce_from_tensor_parallel_region(
    input_: torch.Tensor,
    *,
    group: object | None,
    world_size: int,
) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_, group, world_size)


def scatter_to_tensor_parallel_region(
    input_: torch.Tensor,
    *,
    group: object | None,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    return _ScatterToTensorParallelRegion.apply(input_, group, world_size, rank)


def gather_from_tensor_parallel_region(
    input_: torch.Tensor,
    *,
    group: object | None,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    return _GatherFromTensorParallelRegion.apply(input_, group, world_size, rank)
