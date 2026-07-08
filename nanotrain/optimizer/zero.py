"""ZeRO optimizer variants.

ZeRO-1 shards optimizer states across data-parallel ranks. ZeRO-2 additionally
reduces each gradient only to the rank that owns the corresponding optimizer
state.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import torch
import torch.distributed as dist

from nanotrain.config import OptimizerConfig
from nanotrain.distributed import DistributedContext


def parameter_owner(index: int, world_size: int) -> int:
    """Return the data-parallel rank responsible for a parameter index."""

    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return index % world_size


class ZeroOneAdamW:
    """AdamW with ZeRO-1 style optimizer-state sharding.

    Each rank owns a deterministic subset of parameters and only constructs
    AdamW state for that subset. After the local optimizer step, every parameter
    is broadcast from its owner rank so all model replicas stay synchronized.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        config: OptimizerConfig,
        *,
        device_type: str,
        distributed: DistributedContext,
    ) -> None:
        if config.name != "adamw":
            raise ValueError(f"unsupported optimizer: {config.name}")
        if not distributed.is_ddp:
            raise RuntimeError("ZeRO-1 currently requires DDP data parallelism")

        self.distributed = distributed
        self.param_names: list[str] = []
        self.params: list[torch.nn.Parameter] = []
        self.local_param_names: list[str] = []
        self.local_params: list[torch.nn.Parameter] = []

        for index, (name, param) in enumerate(named_parameters):
            if not param.requires_grad:
                continue
            self.param_names.append(name)
            self.params.append(param)
            if parameter_owner(index, distributed.world_size) == distributed.rank:
                self.local_param_names.append(name)
                self.local_params.append(param)

        decay_params = [p for p in self.local_params if p.dim() >= 2]
        nodecay_params = [p for p in self.local_params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        self.local_optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            **extra_args,
        )

    @property
    def param_groups(self) -> list[dict]:
        return self.local_optimizer.param_groups

    def step(self) -> None:
        self.local_optimizer.step()
        self.synchronize_parameters()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def unscale_(self, scaler: torch.amp.GradScaler) -> None:
        scaler.unscale_(self.local_optimizer)

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(self.local_params, max_norm)

    def synchronize_parameters(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for index, param in enumerate(self.params):
            owner = parameter_owner(index, self.distributed.world_size)
            dist.broadcast(param.data, src=owner)

    def state_dict(self) -> dict:
        return {
            "zero_stage": 1,
            "rank": self.distributed.rank,
            "world_size": self.distributed.world_size,
            "local_param_names": self.local_param_names,
            "optimizer": self.local_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.local_optimizer.load_state_dict(state_dict["optimizer"])


class ZeroTwoAdamW(ZeroOneAdamW):
    """ZeRO-2 MVP: shard optimizer states and reduce gradients to owner ranks."""

    def reduce_gradients(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        for index, param in enumerate(self.params):
            if param.grad is None:
                continue
            owner = parameter_owner(index, self.distributed.world_size)
            dist.reduce(param.grad, dst=owner, op=dist.ReduceOp.SUM)
            if self.distributed.rank == owner:
                param.grad.div_(self.distributed.world_size)
            else:
                param.grad = None

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["zero_stage"] = 2
        return state


__all__ = ["ZeroOneAdamW", "ZeroTwoAdamW", "parameter_owner"]
