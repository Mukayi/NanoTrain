from nanotrain.config import load_config


def test_load_smoke_config() -> None:
    config = load_config("configs/nanotrain_shakespeare_char_smoke.yaml")

    assert config.model.n_layer == 2
    assert config.model.block_size == 32
    assert config.data.dataset == "shakespeare_char"
    assert config.runtime.device == "cpu"
    assert config.train.max_iters == 20
