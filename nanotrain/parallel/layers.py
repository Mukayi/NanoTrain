"""Megatron-style tensor-parallel layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanotrain.distributed import DistributedContext
from nanotrain.parallel.mappings import (
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)


def _divide(numerator: int, denominator: int) -> int:
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} must be divisible by tensor parallel size {denominator}")
    return numerator // denominator


def _vocab_range(vocab_size: int, rank: int, world_size: int) -> tuple[int, int]:
    per_partition = math.ceil(vocab_size / world_size)
    start = min(rank * per_partition, vocab_size)
    end = min(start + per_partition, vocab_size)
    return start, end


class ColumnParallelLinear(nn.Module):
    """Linear layer with output features sharded across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        gather_output: bool,
        context: DistributedContext,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_size_per_partition = _divide(out_features, context.tp_size)
        self.gather_output = gather_output
        self.context = context

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition)) if bias else None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_parallel_region(
            input_,
            group=self.context.tp_group,
            world_size=self.context.tp_size,
        )
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            return gather_from_tensor_parallel_region(
                output_parallel,
                group=self.context.tp_group,
                world_size=self.context.tp_size,
                rank=self.context.tp_rank,
            )
        return output_parallel


class RowParallelLinear(nn.Module):
    """Linear layer with input features sharded across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        input_is_parallel: bool,
        context: DistributedContext,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_size_per_partition = _divide(in_features, context.tp_size)
        self.input_is_parallel = input_is_parallel
        self.context = context

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_parallel_region(
                input_,
                group=self.context.tp_group,
                world_size=self.context.tp_size,
                rank=self.context.tp_rank,
            )
        output_parallel = F.linear(input_parallel, self.weight, None)
        output = reduce_from_tensor_parallel_region(
            output_parallel,
            group=self.context.tp_group,
            world_size=self.context.tp_size,
        )
        return output + self.bias if self.bias is not None else output


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded along the vocabulary dimension."""

    def __init__(self, vocab_size: int, embedding_dim: int, *, context: DistributedContext) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context = context
        self.vocab_start_index, self.vocab_end_index = _vocab_range(
            vocab_size,
            context.tp_rank,
            context.tp_size,
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_ - self.vocab_start_index
        masked_input = masked_input.masked_fill(input_mask, 0)
        output_parallel = F.embedding(masked_input, self.weight)
        output_parallel = output_parallel.masked_fill(input_mask.unsqueeze(-1), 0.0)
        return reduce_from_tensor_parallel_region(
            output_parallel,
            group=self.context.tp_group,
            world_size=self.context.tp_size,
        )


class VocabParallelLinear(nn.Module):
    """LM head sharded along the vocabulary dimension."""

    def __init__(self, vocab_size: int, in_features: int, *, context: DistributedContext) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.in_features = in_features
        self.context = context
        self.vocab_start_index, self.vocab_end_index = _vocab_range(
            vocab_size,
            context.tp_rank,
            context.tp_size,
        )
        self.out_features_per_partition = self.vocab_end_index - self.vocab_start_index
        self.max_out_features_per_partition = math.ceil(vocab_size / context.tp_size)
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tensor_parallel_region(
            input_,
            group=self.context.tp_group,
            world_size=self.context.tp_size,
        )
        logits_parallel = F.linear(input_parallel, self.weight, None)
        pad = self.max_out_features_per_partition - self.out_features_per_partition
        if pad > 0:
            logits_parallel = F.pad(logits_parallel, (0, pad))
        logits = gather_from_tensor_parallel_region(
            logits_parallel,
            group=self.context.tp_group,
            world_size=self.context.tp_size,
            rank=self.context.tp_rank,
        )
        return logits[..., : self.vocab_size].contiguous()


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "VocabParallelLinear",
]
