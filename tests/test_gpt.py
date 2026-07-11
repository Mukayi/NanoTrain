import torch

from nanotrain.model import GPT, GPTConfig


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


def test_gpt_forward_shape_and_loss() -> None:
    model = GPT(tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (4, model.config.block_size))
    targets = torch.randint(0, model.config.vocab_size, (4, model.config.block_size))

    logits, loss = model(idx, targets)

    assert logits.shape == (4, model.config.block_size, model.config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_gpt_inference_only_returns_last_token_logits() -> None:
    model = GPT(tiny_config())
    idx = torch.randint(0, model.config.vocab_size, (4, model.config.block_size))

    logits, loss = model(idx)

    assert logits.shape == (4, 1, model.config.vocab_size)
    assert loss is None


def test_gpt_single_forward_backward_step() -> None:
    model = GPT(tiny_config())
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.99),
        device_type="cpu",
    )
    idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
    targets = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))

    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_gpt_activation_checkpointing_forward_backward_step() -> None:
    model = GPT(tiny_config())
    model.activation_checkpointing = True
    idx = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
    targets = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))

    logits, loss = model(idx, targets)

    assert logits.shape == (2, model.config.block_size, model.config.vocab_size)
    assert loss is not None
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())
