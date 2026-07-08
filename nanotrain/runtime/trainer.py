"""Minimal Phase 1 trainer migrated from nanoGPT's training loop."""

import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nanotrain.checkpoint import CheckpointManager
from nanotrain.config import NanoTrainConfig
from nanotrain.data import ShakespeareCharDataset
from nanotrain.distributed import DistributedContext
from nanotrain.model import GPT, GPTConfig
from nanotrain.optimizer import build_optimizer


class Trainer:
    def __init__(self, config: NanoTrainConfig, project_root: str | Path | None = None) -> None:
        self.config = config
        self.project_root = Path(project_root or ".").resolve()
        self.distributed = DistributedContext.initialize(
            backend=config.distributed.backend,
            requested_device=config.runtime.device,
        )
        self.device = self._resolve_device(self.distributed.device)
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.gradient_accumulation_steps = self._resolve_gradient_accumulation_steps()
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[config.runtime.dtype]
        use_autocast = config.runtime.amp and self.device_type == "cuda"
        self.ctx = (
            torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
            if use_autocast
            else nullcontext()
        )
        torch.manual_seed(config.seed + self.distributed.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        data_dir = self._resolve_path(config.data.data_dir)
        self.dataset = ShakespeareCharDataset(
            data_dir=data_dir,
            block_size=config.model.block_size,
            batch_size=config.data.batch_size,
            device=self.device,
        )

        model_config = self._build_model_config()
        self.model_args = asdict(model_config)
        self.model = GPT(model_config).to(self.device)
        self.optimizer = build_optimizer(self.raw_model, config.optimizer, self.device_type)
        if config.runtime.compile:
            self.model = torch.compile(self.model)
        if self.distributed.is_ddp:
            if self.device_type == "cuda":
                self.model = DDP(self.model, device_ids=[self.distributed.local_rank])
            else:
                self.model = DDP(self.model)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(config.runtime.dtype == "float16" and self.device_type == "cuda")
        )
        self.checkpoint_manager = CheckpointManager(self._resolve_path(config.train.out_dir))
        self.best_val_loss = float("inf")
        self.iter_num = 0
        self.running_mfu = -1.0

    @property
    def raw_model(self) -> GPT:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self.project_root / path

    def _resolve_device(self, requested: str) -> str:
        if "cuda" in requested and not torch.cuda.is_available():
            raise RuntimeError(
                f"requested device={requested!r}, but CUDA is not available. "
                "Use runtime.device=cpu for local smoke runs."
            )
        return requested

    def _resolve_gradient_accumulation_steps(self) -> int:
        steps = self.config.train.gradient_accumulation_steps
        world_size = self.distributed.world_size
        if steps % world_size != 0:
            raise ValueError(
                "train.gradient_accumulation_steps must be divisible by DDP world size "
                f"({steps} % {world_size} != 0)"
            )
        return steps // world_size

    def _build_model_config(self) -> GPTConfig:
        model_cfg = self.config.model
        vocab_size = self.dataset.vocab_size_from_meta() or model_cfg.vocab_size
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=model_cfg.block_size,
            n_layer=model_cfg.n_layer,
            n_head=model_cfg.n_head,
            n_embd=model_cfg.n_embd,
            dropout=model_cfg.dropout,
            bias=model_cfg.bias,
        )

    def get_lr(self, iter_num: int) -> float:
        optimizer_cfg = self.config.optimizer
        scheduler_cfg = self.config.scheduler
        if not scheduler_cfg.decay_lr:
            return optimizer_cfg.learning_rate
        if iter_num < scheduler_cfg.warmup_iters:
            return optimizer_cfg.learning_rate * (iter_num + 1) / (scheduler_cfg.warmup_iters + 1)
        if iter_num > scheduler_cfg.lr_decay_iters:
            return scheduler_cfg.min_lr
        decay_ratio = (iter_num - scheduler_cfg.warmup_iters) / (
            scheduler_cfg.lr_decay_iters - scheduler_cfg.warmup_iters
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return scheduler_cfg.min_lr + coeff * (optimizer_cfg.learning_rate - scheduler_cfg.min_lr)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.train.eval_iters)
            for k in range(self.config.train.eval_iters):
                x, y = self.dataset.get_batch(split)
                with self.ctx:
                    _, loss = self.model(x, y)
                if loss is None:
                    raise RuntimeError("training loss unexpectedly missing")
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _save_checkpoint(self, losses: dict[str, float]) -> None:
        if not self.distributed.is_master:
            return
        should_save = losses["val"] < self.best_val_loss or self.config.train.always_save_checkpoint
        if not should_save:
            return
        self.best_val_loss = losses["val"]
        if self.iter_num == 0:
            return
        path = self.checkpoint_manager.save(
            model=self.raw_model,
            optimizer=self.optimizer,
            model_args=self.model_args,
            iter_num=self.iter_num,
            best_val_loss=self.best_val_loss,
            config=asdict(self.config),
        )
        print(f"saving checkpoint to {path}")

    def train(self) -> None:
        tokens_per_iter = (
            self.gradient_accumulation_steps
            * self.distributed.world_size
            * self.config.data.batch_size
            * self.config.model.block_size
        )
        if self.distributed.is_master:
            print(f"tokens per iteration will be: {tokens_per_iter:,}")
            print(f"number of parameters: {self.raw_model.get_num_params() / 1e6:.2f}M")

        x, y = self.dataset.get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        while True:
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if self.iter_num % self.config.train.eval_interval == 0 and self.distributed.is_master:
                losses = self.estimate_loss()
                print(
                    f"step {self.iter_num}: "
                    f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                self._save_checkpoint(losses)

            for micro_step in range(self.gradient_accumulation_steps):
                if self.distributed.is_ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                with self.ctx:
                    _, loss = self.model(x, y)
                    if loss is None:
                        raise RuntimeError("training loss unexpectedly missing")
                    loss = loss / self.gradient_accumulation_steps
                x, y = self.dataset.get_batch("train")
                self.scaler.scale(loss).backward()

            if self.config.optimizer.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.config.train.log_interval == 0 and self.distributed.is_master:
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:
                    mfu = self.raw_model.estimate_mfu(
                        self.config.data.batch_size * self.gradient_accumulation_steps,
                        dt,
                    )
                    self.running_mfu = (
                        mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {self.iter_num}: loss {lossf:.4f}, "
                    f"time {dt * 1000:.2f}ms, mfu {self.running_mfu * 100:.2f}%"
                )

            self.iter_num += 1
            local_iter_num += 1
            if self.iter_num > self.config.train.max_iters:
                break


def run_training(config: NanoTrainConfig, project_root: str | Path | None = None) -> None:
    trainer = Trainer(config, project_root=project_root)
    try:
        trainer.train()
    finally:
        trainer.distributed.destroy()
