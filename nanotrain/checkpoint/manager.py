"""Checkpoint save utilities for the Phase 1 trainer."""

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_args: dict[str, Any],
        iter_num: int,
        best_val_loss: float,
        config: dict[str, Any],
    ) -> Path:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
        }
        path = self.out_dir / "ckpt.pt"
        torch.save(checkpoint, path)
        return path
