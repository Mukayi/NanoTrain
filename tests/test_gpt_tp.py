import torch

from nanotrain.distributed import DistributedContext
from nanotrain.model import GPT, GPTConfig, TensorParallelGPT


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=65,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )


def test_tensor_parallel_gpt_tp1_matches_baseline_gpt() -> None:
    torch.manual_seed(1234)
    config = tiny_config()
    baseline = GPT(config)
    context = DistributedContext.initialize(backend="gloo", requested_device="cpu")
    tp_model = TensorParallelGPT(config, context=context)
    tp_model.load_state_dict(baseline.state_dict())
    baseline.eval()
    tp_model.eval()

    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    baseline_logits, baseline_loss = baseline(idx, targets)
    tp_logits, tp_loss = tp_model(idx, targets)

    assert baseline_loss is not None
    assert tp_loss is not None
    torch.testing.assert_close(tp_logits, baseline_logits)
    torch.testing.assert_close(tp_loss, baseline_loss)
