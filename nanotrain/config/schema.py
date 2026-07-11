"""Configuration dataclasses for NanoTrain."""

from __future__ import annotations

import dataclasses
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
    init_from: str = "scratch"
    resume_path: str | None = None


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


def _parse_override_value(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(config: NanoTrainConfig, overrides: list[str]) -> NanoTrainConfig:
    """Apply dot-path command-line overrides in-place and return config."""

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must use key=value syntax: {override!r}")
        key_path, raw_value = override.split("=", 1)
        parts = key_path.split(".")
        if len(parts) != 2:
            raise ValueError(f"override key must use section.field syntax: {key_path!r}")

        section_name, field_name = parts
        if not hasattr(config, section_name):
            raise ValueError(f"unknown config section: {section_name!r}")
        section = getattr(config, section_name)
        if not dataclasses.is_dataclass(section):
            raise ValueError(f"config section is not overrideable: {section_name!r}")
        field_names = {field.name for field in dataclasses.fields(section)}
        if field_name not in field_names:
            raise ValueError(f"unknown config field: {key_path!r}")
        setattr(section, field_name, _parse_override_value(raw_value))

    return config


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
