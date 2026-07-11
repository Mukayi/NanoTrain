"""Generic nanoGPT-style binary token dataset loader."""

from pathlib import Path

import numpy as np
import torch


class BinTokenDataset:
    """Reads train.bin/val.bin token files produced by nanoGPT prepare scripts."""

    def __init__(
        self,
        data_dir: str | Path,
        block_size: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self._validate_files()

    def _validate_files(self) -> None:
        missing = [
            path.name
            for path in [self.data_dir / "train.bin", self.data_dir / "val.bin"]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {self.data_dir}. "
                "Run the matching nanoGPT data prepare.py first."
            )

    def vocab_size_from_meta(self) -> int | None:
        return None

    def _read_split(self, split: str) -> np.memmap:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        return np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._read_split(split)
        if len(data) <= self.block_size:
            raise ValueError(
                f"{split} split has {len(data)} tokens, "
                f"but block_size is {self.block_size}; prepare a larger dataset "
                "or reduce block_size."
            )
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + self.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix]
        )
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
