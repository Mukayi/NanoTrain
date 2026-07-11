"""Training loop migrated from nanoGPT and extended into NanoTrain runtime."""

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanotrain.checkpoint import CheckpointManager
from nanotrain.config import NanoTrainConfig
from nanotrain.distributed import DistributedContext
from nanotrain.model import GPT, GPTConfig, TensorParallelGPT
from nanotrain.optimizer import ZeroOneAdamW, ZeroTwoAdamW
from nanotrain.runtime.builders import DataBuilder, ModelBuilder, OptimizerBuilder


class Trainer:
    def __init__(self, config: NanoTrainConfig, project_root: str | Path | None = None) -> None:
        self.config = config
        self.project_root = Path(project_root or ".").resolve()
        self.distributed = DistributedContext.initialize(
            backend=config.distributed.backend,
            requested_device=config.runtime.device,
            tp_size=config.distributed.tp_size,
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
        self.dataset = DataBuilder.build(
            config.data,
            data_dir=data_dir,
            block_size=config.model.block_size,
            device=self.device,
        )
        self.checkpoint_manager = CheckpointManager(self._resolve_path(config.train.out_dir))
        resume_checkpoint = self._load_resume_checkpoint()
        self.metrics_path = self.checkpoint_manager.out_dir / "metrics.jsonl"

        model_config = self._build_model_config()
        if resume_checkpoint is not None:
            model_config = self._model_config_with_checkpoint_args(
                model_config,
                resume_checkpoint["model_args"],
            )
        self.model_args = asdict(model_config)
        self.model = self._build_model(model_config).to(self.device)
        self.optimizer = OptimizerBuilder.build(
            self.raw_model,
            config.optimizer,
            device_type=self.device_type,
            distributed_config=config.distributed,
            distributed_context=self.distributed,
        )
        if config.runtime.compile:
            self.model = torch.compile(self.model)
        if self._should_wrap_ddp():
            if self.device_type == "cuda":
                self.model = DDP(self.model, device_ids=[self.distributed.local_rank])
            else:
                self.model = DDP(self.model)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(config.runtime.dtype == "float16" and self.device_type == "cuda")
        )
        self.best_val_loss = float("inf")
        self.iter_num = 0
        self.running_mfu = -1.0
        if resume_checkpoint is not None:
            self._load_checkpoint_state(resume_checkpoint)
        self.log_file: TextIO | None = None
        self.metrics_file: TextIO | None = None
        if self.distributed.is_master:
            self.log_file = (self.checkpoint_manager.out_dir / "train.log").open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
            self.metrics_file = self.metrics_path.open("a", encoding="utf-8", buffering=1)
            if resume_checkpoint is not None:
                self._log(
                    f"resumed checkpoint at iter {self.iter_num} "
                    f"with best val loss {self.best_val_loss:.4f}"
                )

    @property
    def raw_model(self) -> GPT | TensorParallelGPT:
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
        data_parallel_world_size = self.distributed.world_size if self.distributed.is_ddp else 1
        if steps % data_parallel_world_size != 0:
            raise ValueError(
                "train.gradient_accumulation_steps must be divisible by DDP world size "
                f"({steps} % {data_parallel_world_size} != 0)"
            )
        return steps // data_parallel_world_size

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

    def _build_model(self, model_config: GPTConfig) -> GPT | TensorParallelGPT:
        return ModelBuilder.build(
            model_config,
            distributed=self.distributed,
            activation_checkpointing=self.config.runtime.activation_checkpointing,
        )

    def _load_resume_checkpoint(self) -> dict[str, Any] | None:
        init_from = self.config.train.init_from
        if init_from == "scratch":
            return None
        if init_from != "resume":
            raise ValueError("train.init_from must be either 'scratch' or 'resume'")
        if self.distributed.is_tensor_parallel or self.config.distributed.zero_stage != 0:
            raise RuntimeError(
                "checkpoint resume currently supports non-TP, non-ZeRO training only"
            )
        resume_path = (
            self._resolve_path(self.config.train.resume_path)
            if self.config.train.resume_path is not None
            else None
        )
        return self.checkpoint_manager.load(resume_path, map_location=self.device)

    def _model_config_with_checkpoint_args(
        self,
        model_config: GPTConfig,
        checkpoint_model_args: dict[str, Any],
    ) -> GPTConfig:
        model_args = asdict(model_config)
        for key in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[key] = checkpoint_model_args[key]
        return GPTConfig(**model_args)

    def _load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        state_dict = dict(checkpoint["model"])
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        self.raw_model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.iter_num = int(checkpoint["iter_num"])
        self.best_val_loss = float(checkpoint["best_val_loss"])
        self._restore_rng_state(checkpoint.get("rng_state"))

    def _should_wrap_ddp(self) -> bool:
        return self.distributed.is_ddp and self.config.distributed.zero_stage != 2

    def _log(self, message: str) -> None:
        if not self.distributed.is_master:
            return
        print(message)
        if self.log_file is not None:
            self.log_file.write(f"{message}\n")

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.metrics_file is not None:
            self.metrics_file.close()
            self.metrics_file = None

    def _unscale_optimizer(self) -> None:
        if isinstance(self.optimizer, ZeroOneAdamW):
            self.optimizer.unscale_(self.scaler)
        else:
            self.scaler.unscale_(self.optimizer)

    def _clip_grad_norm(self) -> tuple[bool, float]:
        self._unscale_optimizer()
        max_norm = (
            self.config.optimizer.grad_clip
            if self.config.optimizer.grad_clip != 0.0
            else float("inf")
        )
        if isinstance(self.optimizer, ZeroOneAdamW):
            grad_norm = self.optimizer.clip_grad_norm_(max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm,
            )
        return True, float(grad_norm)

    def _step_optimizer(self, *, already_unscaled: bool) -> None:
        if isinstance(self.optimizer, ZeroOneAdamW):
            if self.scaler.is_enabled() and not already_unscaled:
                self.optimizer.unscale_(self.scaler)
            self.optimizer.step()
            self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def _reduce_zero2_gradients(self) -> None:
        if isinstance(self.optimizer, ZeroTwoAdamW):
            self.optimizer.reduce_gradients()

    def _sync_device(self) -> None:
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_peak_memory_stats(self) -> None:
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _scheduler_state(self) -> dict[str, Any]:
        return {
            "iter_num": self.iter_num,
            "decay_lr": self.config.scheduler.decay_lr,
            "warmup_iters": self.config.scheduler.warmup_iters,
            "lr_decay_iters": self.config.scheduler.lr_decay_iters,
            "min_lr": self.config.scheduler.min_lr,
        }

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _peak_gpu_memory_mb(self) -> float | None:
        if self.device_type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

    def _tensor_memory_bytes(self, tensors: list[torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors)

    def _parameter_memory_mb(self) -> float:
        return self._tensor_memory_bytes(list(self.raw_model.parameters())) / 1024 / 1024

    def _gradient_memory_mb(self) -> float:
        grads = [param.grad for param in self.raw_model.parameters() if param.grad is not None]
        return self._tensor_memory_bytes(grads) / 1024 / 1024

    def _optimizer_state_memory_mb(self) -> float:
        optimizer = (
            self.optimizer.local_optimizer
            if isinstance(self.optimizer, ZeroOneAdamW)
            else self.optimizer
        )
        tensors = []
        for state in optimizer.state.values():
            tensors.extend(value for value in state.values() if torch.is_tensor(value))
        return self._tensor_memory_bytes(tensors) / 1024 / 1024

    def _activation_memory_estimate_mb(self) -> float:
        bytes_per_element = torch.tensor([], dtype=self.ptdtype).element_size()
        activation_elements = (
            self.config.data.batch_size
            * self.config.model.block_size
            * self.config.model.n_embd
            * self.config.model.n_layer
        )
        return activation_elements * bytes_per_element / 1024 / 1024

    def _training_mode(self) -> str:
        if self.distributed.is_tensor_parallel:
            return f"tp{self.distributed.tp_size}"
        if self.config.distributed.zero_stage > 0:
            return f"zero{self.config.distributed.zero_stage}"
        if self.distributed.is_ddp:
            return f"ddp{self.distributed.world_size}"
        return "single"

    def _write_metric(self, record: dict[str, Any]) -> None:
        if self.metrics_file is None:
            return
        payload = {
            "time": time.time(),
            "rank": self.distributed.rank,
            "world_size": self.distributed.world_size,
            "mode": self._training_mode(),
            **record,
        }
        self.metrics_file.write(json.dumps(payload, sort_keys=True) + "\n")

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
                x, y = self._get_batch(split)
                with self.ctx:
                    _, loss = self.model(x, y)
                if loss is None:
                    raise RuntimeError("training loss unexpectedly missing")
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def _save_checkpoint(self, losses: dict[str, float]) -> Path | None:
        if not self.distributed.is_master:
            return None
        if self.distributed.is_tensor_parallel:
            if self.config.train.always_save_checkpoint:
                self._log(
                    "skipping checkpoint save: "
                    "Tensor Parallel checkpointing is not implemented yet"
                )
            return None
        if isinstance(self.optimizer, ZeroOneAdamW):
            if self.config.train.always_save_checkpoint:
                self._log("skipping checkpoint save: ZeRO checkpointing is not implemented yet")
            return None
        should_save = losses["val"] < self.best_val_loss or self.config.train.always_save_checkpoint
        if not should_save:
            return None
        self.best_val_loss = losses["val"]
        if self.iter_num == 0:
            return None
        path = self.checkpoint_manager.save(
            model=self.raw_model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            model_args=self.model_args,
            iter_num=self.iter_num,
            best_val_loss=self.best_val_loss,
            config=asdict(self.config),
            scheduler_state=self._scheduler_state(),
            rng_state=self._rng_state(),
        )
        self._log(f"saving checkpoint to {path}")
        return path

    def _get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset.get_batch(split)
        if self.distributed.is_tensor_parallel and dist.is_available() and dist.is_initialized():
            dist.broadcast(x, src=0, group=self.distributed.tp_group)
            dist.broadcast(y, src=0, group=self.distributed.tp_group)
        return x, y

    def train(self) -> None:
        tokens_per_iter = (
            self.gradient_accumulation_steps
            * (self.distributed.world_size if self.distributed.is_ddp else 1)
            * self.config.data.batch_size
            * self.config.model.block_size
        )
        if self.distributed.is_master:
            self._log(f"tokens per iteration will be: {tokens_per_iter:,}")
            self._log(f"number of parameters: {self.raw_model.get_num_params() / 1e6:.2f}M")
        self._reset_peak_memory_stats()

        x, y = self._get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        while True:
            lr = self.get_lr(self.iter_num)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if self.iter_num % self.config.train.eval_interval == 0:
                if self.distributed.is_master or self.distributed.is_tensor_parallel:
                    losses = self.estimate_loss()
                else:
                    losses = None
                if self.distributed.is_master and losses is not None:
                    self._log(
                        f"step {self.iter_num}: "
                        f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                    )
                    checkpoint_path = self._save_checkpoint(losses)
                    self._write_metric(
                        {
                            "type": "eval",
                            "iter": self.iter_num,
                            "train_loss": losses["train"],
                            "val_loss": losses["val"],
                            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                        }
                    )

            if self.iter_num >= self.config.train.max_iters:
                if self.distributed.is_master:
                    self._log(f"training complete at iter {self.iter_num}")
                break

            forward_time = 0.0
            backward_time = 0.0
            communication_time = 0.0
            optimizer_time = 0.0
            self._reset_peak_memory_stats()
            for micro_step in range(self.gradient_accumulation_steps):
                if isinstance(self.model, DDP):
                    self.model.require_backward_grad_sync = (
                        micro_step == self.gradient_accumulation_steps - 1
                    )
                self._sync_device()
                forward_start = time.perf_counter()
                with self.ctx:
                    _, loss = self.model(x, y)
                    if loss is None:
                        raise RuntimeError("training loss unexpectedly missing")
                    loss = loss / self.gradient_accumulation_steps
                self._sync_device()
                forward_time += time.perf_counter() - forward_start
                x, y = self._get_batch("train")
                self._sync_device()
                backward_start = time.perf_counter()
                self.scaler.scale(loss).backward()
                self._sync_device()
                backward_time += time.perf_counter() - backward_start

            communication_start = time.perf_counter()
            self._reduce_zero2_gradients()
            self._sync_device()
            communication_time += time.perf_counter() - communication_start
            already_unscaled, grad_norm = self._clip_grad_norm()
            optimizer_start = time.perf_counter()
            self._step_optimizer(already_unscaled=already_unscaled)
            self._sync_device()
            optimizer_time += time.perf_counter() - optimizer_start
            gradient_memory_mb = self._gradient_memory_mb()
            optimizer_state_memory_mb = self._optimizer_state_memory_mb()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.config.train.log_interval == 0 and self.distributed.is_master:
                lossf = loss.item() * self.gradient_accumulation_steps
                tokens_per_second = tokens_per_iter / dt if dt > 0 else 0.0
                step_peak_gpu_memory = self._peak_gpu_memory_mb()
                if local_iter_num >= 5:
                    mfu = self.raw_model.estimate_mfu(
                        self.config.data.batch_size * self.gradient_accumulation_steps,
                        dt,
                    )
                    self.running_mfu = (
                        mfu if self.running_mfu == -1.0 else 0.9 * self.running_mfu + 0.1 * mfu
                    )
                memory_msg = (
                    f", step_peak_gpu_mem {step_peak_gpu_memory:.0f}MB"
                    if step_peak_gpu_memory is not None
                    else ""
                )
                self._log(
                    f"iter {self.iter_num}: loss {lossf:.4f}, lr {lr:.4e}, "
                    f"grad_norm {grad_norm:.4f}, time {dt * 1000:.2f}ms, "
                    f"tokens/s {tokens_per_second:.2f}, mfu {self.running_mfu * 100:.2f}%"
                    f"{memory_msg}"
                )
                self._write_metric(
                    {
                        "type": "train_step",
                        "iter": self.iter_num,
                        "loss": lossf,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "step_time_ms": dt * 1000,
                        "forward_time_ms": forward_time * 1000,
                        "backward_time_ms": backward_time * 1000,
                        "optimizer_time_ms": optimizer_time * 1000,
                        "communication_time_ms": communication_time * 1000,
                        "tokens_per_second": tokens_per_second,
                        "samples_per_second": (
                            self.config.data.batch_size * self.gradient_accumulation_steps / dt
                            if dt > 0
                            else 0.0
                        ),
                        "tokens_per_iter": tokens_per_iter,
                        "batch_size": self.config.data.batch_size,
                        "gradient_accumulation_steps": self.gradient_accumulation_steps,
                        "peak_gpu_memory_mb": step_peak_gpu_memory,
                        "step_peak_gpu_memory_mb": step_peak_gpu_memory,
                        "parameter_memory_mb": self._parameter_memory_mb(),
                        "gradient_memory_mb": gradient_memory_mb,
                        "optimizer_state_memory_mb": optimizer_state_memory_mb,
                        "activation_memory_estimate_mb": self._activation_memory_estimate_mb(),
                        "mfu_percent": self.running_mfu * 100,
                    }
                )

            self.iter_num += 1
            local_iter_num += 1


def run_training(config: NanoTrainConfig, project_root: str | Path | None = None) -> None:
    from nanotrain.runtime.engine import TrainingEngine

    TrainingEngine(config, project_root=project_root).run()
