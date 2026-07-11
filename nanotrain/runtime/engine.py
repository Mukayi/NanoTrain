"""Reusable runtime engine API."""

from pathlib import Path

from nanotrain.config import NanoTrainConfig
from nanotrain.runtime.trainer import Trainer


class TrainingEngine:
    """Small orchestration layer around Trainer for public runtime reuse."""

    def __init__(self, config: NanoTrainConfig, project_root: str | Path | None = None) -> None:
        self.trainer = Trainer(config, project_root=project_root)

    def run(self) -> None:
        try:
            self.trainer.train()
        finally:
            self.close()

    def close(self) -> None:
        self.trainer.close()
        self.trainer.distributed.destroy()


__all__ = ["TrainingEngine"]
