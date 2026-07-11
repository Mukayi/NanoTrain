import torch

from nanotrain.checkpoint import CheckpointManager


def test_checkpoint_manager_saves_resume_state(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    manager = CheckpointManager(tmp_path)

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_args={
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 2,
            "block_size": 4,
            "bias": True,
            "vocab_size": 8,
        },
        iter_num=7,
        best_val_loss=1.25,
        config={"train": {"init_from": "scratch"}},
        scheduler_state={"iter_num": 7},
        rng_state={"torch": torch.get_rng_state()},
    )

    checkpoint = manager.load(path, map_location="cpu")

    assert checkpoint["iter_num"] == 7
    assert checkpoint["best_val_loss"] == 1.25
    assert checkpoint["model_args"]["vocab_size"] == 8
    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert "scaler" in checkpoint
    assert checkpoint["scheduler"] == {"iter_num": 7}
    assert "torch" in checkpoint["rng_state"]
