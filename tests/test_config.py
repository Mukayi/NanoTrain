from nanotrain.config import load_config


def test_load_smoke_config() -> None:
    config = load_config("configs/nanotrain_shakespeare_char_smoke.yaml")

    assert config.model.n_layer == 2
    assert config.model.block_size == 32
    assert config.data.dataset == "shakespeare_char"
    assert config.runtime.device == "cpu"
    assert config.train.max_iters == 20


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
