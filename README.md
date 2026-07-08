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

Phase 0 project setup is in progress:

- local git repository
- Python package skeleton
- `ruff`, `black`, `pytest`, and `pre-commit` configuration
- Conda and Docker environment files
- minimal import test
- initial YAML config schema
- `nanoGPT/` baseline kept intact for reference



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



## Roadmap

- v0.1: GPT single-GPU training with normal loss convergence
- v0.2: DDP training migrated from the `nanoGPT` baseline
- v0.3: Tensor Parallel training with multi-GPU loss aligned to single-GPU baseline
- v0.4: ZeRO-1 and activation checkpointing with memory comparison
- v0.5: reusable runtime with config, checkpoint, resume, AMP, and profiling
- v0.6: benchmark results for throughput, memory, and scaling

