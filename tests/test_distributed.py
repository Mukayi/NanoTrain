from nanotrain.distributed import DistributedContext


def test_distributed_context_defaults_to_single_process(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    context = DistributedContext.initialize(backend="gloo", requested_device="cpu")

    assert not context.is_distributed
    assert not context.is_ddp
    assert not context.is_tensor_parallel
    assert context.rank == 0
    assert context.local_rank == 0
    assert context.world_size == 1
    assert context.tp_size == 1
    assert context.tp_rank == 0
    assert context.device == "cpu"
    assert context.is_master
    assert context.seed_offset == 0


def test_tensor_parallel_requires_torchrun(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    try:
        DistributedContext.initialize(backend="gloo", requested_device="cpu", tp_size=2)
    except RuntimeError as exc:
        assert "requires torchrun" in str(exc)
    else:
        raise AssertionError("expected Tensor Parallel without torchrun to fail")
