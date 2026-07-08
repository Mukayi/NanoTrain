"""Configuration dataclasses for NanoTrain."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


@dataclass
class DataConfig:
    dataset: str = "shakespeare_char"
    data_dir: str = "nanoGPT/data/shakespeare_char"
    batch_size: int = 64


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5


@dataclass
class TrainConfig:
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    gradient_accumulation_steps: int = 1
    out_dir: str = "out-shakespeare-char"
    always_save_checkpoint: bool = False


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    tp_size: int = 1
    zero_stage: int = 0


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    amp: bool = True
    activation_checkpointing: bool = False


@dataclass
class NanoTrainConfig:
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _build_dataclass(cls: type, values: dict[str, Any]) -> Any:
    return cls(**values)


def load_config(path: str | Path) -> NanoTrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return NanoTrainConfig(
        seed=raw.get("seed", 1337),
        model=_build_dataclass(ModelConfig, raw.get("model", {})),
        data=_build_dataclass(DataConfig, raw.get("data", {})),
        optimizer=_build_dataclass(OptimizerConfig, raw.get("optimizer", {})),
        scheduler=_build_dataclass(SchedulerConfig, raw.get("scheduler", {})),
        train=_build_dataclass(TrainConfig, raw.get("train", {})),
        distributed=_build_dataclass(DistributedConfig, raw.get("distributed", {})),
        runtime=_build_dataclass(RuntimeConfig, raw.get("runtime", {})),
    )
