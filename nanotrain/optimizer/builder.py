"""Optimizer builder utilities."""

import torch

from nanotrain.config import DistributedConfig, OptimizerConfig
from nanotrain.distributed import DistributedContext
from nanotrain.model import GPT, TensorParallelGPT
from nanotrain.optimizer.zero import ZeroOneAdamW, ZeroTwoAdamW


def build_optimizer(
    model: GPT | TensorParallelGPT,
    config: OptimizerConfig,
    device_type: str,
    distributed_config: DistributedConfig | None = None,
    distributed_context: DistributedContext | None = None,
) -> torch.optim.Optimizer | ZeroOneAdamW:
    if config.name != "adamw":
        raise ValueError(f"unsupported optimizer: {config.name}")
    zero_stage = distributed_config.zero_stage if distributed_config is not None else 0
    if zero_stage == 1:
        if distributed_context is None:
            raise RuntimeError("ZeRO-1 requires a distributed context")
        return ZeroOneAdamW(
            model.named_parameters(),
            config,
            device_type=device_type,
            distributed=distributed_context,
        )
    if zero_stage == 2:
        if distributed_context is None:
            raise RuntimeError("ZeRO-2 requires a distributed context")
        return ZeroTwoAdamW(
            model.named_parameters(),
            config,
            device_type=device_type,
            distributed=distributed_context,
        )
    if zero_stage != 0:
        raise ValueError(f"unsupported zero_stage: {zero_stage}")
    return model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )
