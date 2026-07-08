"""Tensor-parallel GPT model using Megatron-LM style layer sharding."""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanotrain.distributed import DistributedContext
from nanotrain.model.gpt import GPTConfig, LayerNorm
from nanotrain.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelLinear,
)


class TensorParallelCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, context: DistributedContext) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.n_head % context.tp_size != 0:
            raise ValueError("n_head must be divisible by distributed.tp_size")
        if config.n_embd % context.tp_size != 0:
            raise ValueError("n_embd must be divisible by distributed.tp_size")

        self.c_attn = ColumnParallelLinear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
            gather_output=False,
            context=context,
        )
        self.c_proj = RowParallelLinear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
            context=context,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head // context.tp_size
        self.n_embd = config.n_embd
        self.n_embd_per_partition = config.n_embd // context.tp_size
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd_per_partition, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd_per_partition)
        return self.resid_dropout(self.c_proj(y))


class TensorParallelMLP(nn.Module):
    def __init__(self, config: GPTConfig, context: DistributedContext) -> None:
        super().__init__()
        self.c_fc = ColumnParallelLinear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
            gather_output=False,
            context=context,
        )
        self.gelu = nn.GELU()
        self.c_proj = RowParallelLinear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
            context=context,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TensorParallelBlock(nn.Module):
    def __init__(self, config: GPTConfig, context: DistributedContext) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = TensorParallelCausalSelfAttention(config, context)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = TensorParallelMLP(config, context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TensorParallelGPT(nn.Module):
    def __init__(self, config: GPTConfig, context: DistributedContext) -> None:
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be set")
        if config.block_size is None:
            raise ValueError("block_size must be set")
        if context.tp_size < 1:
            raise ValueError("distributed.tp_size must be >= 1")
        self.config = config
        self.context = context

        self.transformer = nn.ModuleDict(
            {
                "wte": VocabParallelEmbedding(
                    config.vocab_size,
                    config.n_embd,
                    context=context,
                ),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [TensorParallelBlock(config, context) for _ in range(config.n_layer)]
                ),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = VocabParallelLinear(config.vocab_size, config.n_embd, context=context)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        linear_types = (nn.Linear, ColumnParallelLinear, RowParallelLinear, VocabParallelLinear)
        embedding_types = (nn.Embedding, VocabParallelEmbedding)
        if isinstance(module, linear_types):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, embedding_types):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}, "
                f"block size is only {self.config.block_size}"
            )
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        if dt <= 0:
            return 0.0
        num_params = self.get_num_params()
        cfg = self.config
        layers, heads, head_size, seq_len = (
            cfg.n_layer,
            cfg.n_head,
            cfg.n_embd // cfg.n_head,
            cfg.block_size,
        )
        flops_per_token = 6 * num_params + 12 * layers * heads * head_size * seq_len
        flops_per_fwdbwd = flops_per_token * seq_len
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised


__all__ = [
    "TensorParallelBlock",
    "TensorParallelCausalSelfAttention",
    "TensorParallelGPT",
    "TensorParallelMLP",
]
