"""Checkpoint save/resume utilities adapted from nanoGPT's checkpoint format."""

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
        scaler: torch.amp.GradScaler,
        model_args: dict[str, Any],
        iter_num: int,
        best_val_loss: float,
        config: dict[str, Any],
        scheduler_state: dict[str, Any],
        rng_state: dict[str, Any],
    ) -> Path:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_args": model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config": config,
            "scheduler": scheduler_state,
            "rng_state": rng_state,
        }
        path = self.out_dir / "ckpt.pt"
        torch.save(checkpoint, path)
        return path

    def load(
        self, path: str | Path | None = None, *, map_location: str | torch.device
    ) -> dict[str, Any]:
        checkpoint_path = Path(path) if path is not None else self.out_dir / "ckpt.pt"
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
