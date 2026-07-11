"""Runtime builder APIs for data, model, and optimizer construction."""

from pathlib import Path

import torch

from nanotrain.config import DataConfig, DistributedConfig, OptimizerConfig
from nanotrain.data import BinTokenDataset, ShakespeareCharDataset
from nanotrain.distributed import DistributedContext
from nanotrain.model import GPT, GPTConfig, TensorParallelGPT
from nanotrain.optimizer import ZeroOneAdamW, build_optimizer


class DataBuilder:
    @staticmethod
    def build(
        config: DataConfig,
        *,
        data_dir: Path,
        block_size: int,
        device: str,
    ) -> ShakespeareCharDataset | BinTokenDataset:
        if config.dataset == "shakespeare_char":
            return ShakespeareCharDataset(
                data_dir=data_dir,
                block_size=block_size,
                batch_size=config.batch_size,
                device=device,
            )
        if config.dataset in {"bin_token", "openwebtext"}:
            return BinTokenDataset(
                data_dir=data_dir,
                block_size=block_size,
                batch_size=config.batch_size,
                device=device,
            )
        raise ValueError(f"unsupported dataset: {config.dataset}")


class ModelBuilder:
    @staticmethod
    def build(
        model_config: GPTConfig,
        *,
        distributed: DistributedContext,
        activation_checkpointing: bool,
    ) -> GPT | TensorParallelGPT:
        if distributed.is_tensor_parallel:
            model = TensorParallelGPT(model_config, context=distributed)
        else:
            model = GPT(model_config)
        model.activation_checkpointing = activation_checkpointing
        return model


class OptimizerBuilder:
    @staticmethod
    def build(
        model: GPT | TensorParallelGPT,
        config: OptimizerConfig,
        *,
        device_type: str,
        distributed_config: DistributedConfig,
        distributed_context: DistributedContext,
    ) -> torch.optim.Optimizer | ZeroOneAdamW:
        return build_optimizer(
            model,
            config,
            device_type,
            distributed_config=distributed_config,
            distributed_context=distributed_context,
        )


__all__ = ["DataBuilder", "ModelBuilder", "OptimizerBuilder"]
