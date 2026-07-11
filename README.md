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
- Benchmark configs and scripts for DDP scaling, TP memory/perf, ZeRO memory,
  activation checkpointing, and Nsight Systems traces

The original `GPT` baseline remains unchanged; TP and ZeRO paths are opt-in via
YAML config fields.

## Benchmark Results

Benchmarks below were run on the available RTX A6000 node with BF16 AMP, using
nanoGPT-style Shakespeare binary data for repeatability. OpenWebText/GPT-2 124M
configs are included, but `nanoGPT/data/openwebtext/train.bin` and `val.bin`
must be prepared before those runs can execute.

- DDP scaling on the medium GPT config reached 376K tokens/s on 1 GPU, 609K on
  2 GPUs, 1.12M on 4 GPUs, and 2.14M on 8 GPUs. Relative to the 1-GPU baseline,
  the 8-GPU run achieved 5.70x speedup and 71.2% scaling efficiency.
- Activation checkpointing on the 85M memory-stress config reduced single-GPU
  step peak memory from 1866 MB to 1410 MB, a 24.4% reduction, with tokens/s
  decreasing from 61.3K to 49.9K.
- ZeRO-1 on the 2-GPU 85M config reduced optimizer-state memory from 652 MB to
  270 MB and step peak memory from 2183 MB to 1797 MB. ZeRO-2 additionally
  reduced gradient memory from 326 MB to 135 MB and step peak memory to 1483 MB,
  a 32.1% peak-memory reduction versus DDP.
- Pure Tensor Parallel on the 56.7M/28.3M/14.2M per-rank medium config reduced
  step peak memory from 1256 MB on 1 GPU to 732 MB on 2 GPUs and 464 MB on
  4 GPUs. Throughput was lower than the single-GPU baseline for this small model,
  which makes communication overhead visible.

Known benchmark caveats: the current OpenWebText dataset is not prepared on this
machine, so GPT-2/OpenWebText configs are ready but not included in the measured
numbers. The 4-GPU TP run wrote its summary successfully, then emitted an NCCL
watchdog warning during shutdown. Nsight Systems reports were generated, but the
installed Nsight/driver combination reported non-fatal import analysis errors.

## Resume Bullet

- Refactored `nanoGPT` into NanoTrain, a modular GPT training engine with YAML
  configs, DDP, Megatron-style Tensor Parallel layers, ZeRO-1/2 optimizer
  sharding, activation checkpointing, checkpoint resume, and benchmark tooling;
  on RTX A6000 experiments, 8-GPU DDP reached 2.14M tokens/s with 5.70x speedup,
  ZeRO-2 reduced 2-GPU peak memory by 32.1%, and 4-GPU TP reduced per-rank peak
  memory by 63.1%.



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

