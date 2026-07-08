"""Optimizer builder utilities."""

import torch

from nanotrain.config import OptimizerConfig
from nanotrain.model import GPT


def build_optimizer(model: GPT, config: OptimizerConfig, device_type: str) -> torch.optim.Optimizer:
    if config.name != "adamw":
        raise ValueError(f"unsupported optimizer: {config.name}")
    return model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )
