# NanoTrain

NanoTrain is a lightweight, extensible training engine built from a `nanoGPT` baseline.

The project starts with a known-good single-node GPT training implementation, then gradually factors the code into reusable training-engine components: model definitions, distributed process groups, tensor-parallel layers, optimizer-state sharding, runtime orchestration, checkpointing, mixed precision, and benchmark tooling.

## Motivation

The goal is not to reproduce a paper or write a one-off training script. The goal is to build a small but real AI infrastructure project that can evolve over time.

NanoTrain will use `nanoGPT/` as the initial reference implementation for:

- GPT model structure
- single-GPU training loop
- DDP launch pattern
- Shakespeare and OpenWebText data preparation
- baseline loss and throughput behavior

The `nanotrain/` package will become the clean training engine implementation.

## Current Status

NanoTrain currently has a working training path from the original single-process
baseline through distributed training MVPs:

- Single-process GPT training migrated from `nanoGPT`
- DDP training launched with `torchrun`
- Megatron-style pure Tensor Parallel over attention, MLP, embedding, and LM head
- ZeRO-1 optimizer-state sharding for DDP
- ZeRO-2 MVP with owner-rank gradient reduction for DDP
- Smoke configs and tests for single-process, DDP, TP, ZeRO-1, and ZeRO-2 paths

The original `GPT` baseline remains unchanged; TP and ZeRO paths are opt-in via
YAML config fields.



## Repository Layout

```text
NanoTrain/
├── nanoGPT/            # upstream baseline/reference implementation
├── nanotrain/          # NanoTrain training engine package
│   ├── model/
│   ├── parallel/
│   ├── optimizer/
│   ├── runtime/
│   ├── distributed/
│   ├── checkpoint/
│   ├── data/
│   ├── config/
│   └── utils/
├── benchmark/
├── configs/
├── docs/
├── examples/
├── scripts/
├── tests/
└── PLAN.md
```



## Quick Start

Create the development environment:

```bash
conda env create -f environment.yml
conda activate nanotrain
```

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run formatting and linting:

```bash
ruff check .
ruff format .
black .
```

Prepare the nanoGPT Shakespeare character dataset:

```bash
cd nanoGPT
python data/shakespeare_char/prepare.py
```

Run the nanoGPT baseline:

```bash
python train.py config/train_shakespeare_char.py
```

Run the NanoTrain migrated smoke trainer from the project root:

```bash
cd /data_2/xiatianze/nanotrain
python train.py --config configs/nanotrain_shakespeare_char_smoke.yaml
```

Run the NanoTrain single-GPU Shakespeare config:

```bash
python train.py --config configs/gpt_single_gpu.yaml
```

Run the NanoTrain DDP smoke config on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_ddp_smoke.yaml
```

The DDP path follows the `nanoGPT` launch pattern: `torchrun` provides
`RANK`, `LOCAL_RANK`, and `WORLD_SIZE`; NanoTrain binds each process to its
local CUDA device and only rank 0 logs evaluations or saves checkpoints.

Run the NanoTrain pure Tensor Parallel smoke config on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_tp_smoke.yaml
```

The Tensor Parallel path uses Megatron-style column-parallel and row-parallel
linear layers, vocab-parallel token embeddings, and a vocab-parallel LM head.
For now, `WORLD_SIZE` must equal `distributed.tp_size`; hybrid TP+DDP is a
later milestone. Tensor Parallel checkpoint save/resume is also deferred until
the distributed checkpoint phase.

Run the NanoTrain ZeRO-1 smoke config on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_zero1_smoke.yaml
```

The ZeRO-1 path shards AdamW optimizer state across DDP ranks while keeping
parameters and gradients replicated. After each local optimizer step, NanoTrain
broadcasts updated parameters from their owner rank. ZeRO-1 checkpoint
save/resume is deferred until the distributed checkpoint phase.

Run the NanoTrain ZeRO-2 smoke config on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py --config configs/gpt_zero2_smoke.yaml
```

The ZeRO-2 MVP avoids DDP's default gradient all-reduce. After backward,
NanoTrain reduces each parameter gradient only to its owner rank, clears
non-owner gradients, steps local AdamW shards, and broadcasts updated
parameters. This first version is DDP-only and does not yet support TP+ZeRO.



## Roadmap

- v0.1: GPT single-GPU training with normal loss convergence
- v0.2: DDP training migrated from the `nanoGPT` baseline
- v0.3: Megatron-style Tensor Parallel MVP
- v0.4: ZeRO-1 and ZeRO-2 MVPs for DDP
- v0.5: distributed checkpointing, resume, activation checkpointing, and runtime cleanup
- v0.6: benchmark results for throughput, memory, and scaling

