import pytest

from nanotrain.config import apply_overrides, load_config


def test_load_smoke_config() -> None:
    config = load_config("configs/nanotrain_shakespeare_char_smoke.yaml")

    assert config.model.n_layer == 2
    assert config.model.block_size == 32
    assert config.data.dataset == "shakespeare_char"
    assert config.runtime.device == "cpu"
    assert config.train.max_iters == 20
    assert config.train.init_from == "scratch"
    assert config.train.resume_path is None


def test_load_ddp_smoke_config() -> None:
    config = load_config("configs/gpt_ddp_smoke.yaml")

    assert config.distributed.backend == "nccl"
    assert config.distributed.tp_size == 1
    assert config.runtime.device == "cuda"
    assert config.train.gradient_accumulation_steps == 2


def test_load_tp_smoke_config() -> None:
    config = load_config("configs/gpt_tp_smoke.yaml")

    assert config.distributed.backend == "nccl"
    assert config.distributed.tp_size == 2
    assert config.runtime.device == "cuda"
    assert config.train.gradient_accumulation_steps == 1


def test_load_zero1_smoke_config() -> None:
    config = load_config("configs/gpt_zero1_smoke.yaml")

    assert config.distributed.backend == "nccl"
    assert config.distributed.tp_size == 1
    assert config.distributed.zero_stage == 1
    assert config.train.gradient_accumulation_steps == 2


def test_load_zero2_smoke_config() -> None:
    config = load_config("configs/gpt_zero2_smoke.yaml")

    assert config.distributed.backend == "nccl"
    assert config.distributed.tp_size == 1
    assert config.distributed.zero_stage == 2
    assert config.train.gradient_accumulation_steps == 2


def test_load_benchmark_configs() -> None:
    medium = load_config("configs/bench_gpt_medium.yaml")
    memory = load_config("configs/bench_gpt_memory_stress.yaml")
    tp = load_config("configs/bench_gpt_tp_medium.yaml")
    owt = load_config("configs/bench_gpt2_owt.yaml")

    assert medium.model.n_layer == 6
    assert memory.model.block_size == 512
    assert tp.distributed.tp_size == 2
    assert owt.data.dataset == "openwebtext"
    assert owt.model.n_layer == 12
    assert owt.model.n_embd == 768


def test_apply_config_overrides() -> None:
    config = load_config("configs/nanotrain_shakespeare_char_smoke.yaml")

    apply_overrides(
        config,
        [
            "train.max_iters=3",
            "runtime.activation_checkpointing=true",
            "runtime.device=cpu",
            "train.resume_path=null",
        ],
    )

    assert config.train.max_iters == 3
    assert config.runtime.activation_checkpointing is True
    assert config.runtime.device == "cpu"
    assert config.train.resume_path is None


def test_apply_config_overrides_rejects_unknown_field() -> None:
    config = load_config("configs/nanotrain_shakespeare_char_smoke.yaml")

    with pytest.raises(ValueError, match="unknown config field"):
        apply_overrides(config, ["train.nope=1"])
