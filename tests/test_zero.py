import torch.nn as nn

from nanotrain.config import OptimizerConfig
from nanotrain.distributed import DistributedContext
from nanotrain.optimizer import ZeroOneAdamW, parameter_owner


def fake_ddp_context(rank: int, world_size: int) -> DistributedContext:
    return DistributedContext(
        is_distributed=True,
        is_ddp=True,
        is_tensor_parallel=False,
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        tp_size=1,
        tp_rank=0,
        backend="gloo",
        device="cpu",
        is_master=(rank == 0),
        tp_group=None,
    )


def test_parameter_owner_round_robin() -> None:
    assert [parameter_owner(i, 2) for i in range(6)] == [0, 1, 0, 1, 0, 1]


def test_zero_one_adamw_keeps_only_local_optimizer_params() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    optimizer = ZeroOneAdamW(
        model.named_parameters(),
        OptimizerConfig(),
        device_type="cpu",
        distributed=fake_ddp_context(rank=1, world_size=2),
    )

    expected_names = [
        name
        for index, (name, param) in enumerate(model.named_parameters())
        if param.requires_grad and parameter_owner(index, 2) == 1
    ]
    local_group_params = {param for group in optimizer.param_groups for param in group["params"]}

    assert optimizer.local_param_names == expected_names
    assert set(optimizer.local_params) == local_group_params
    assert len(optimizer.local_params) < len(list(model.parameters()))
