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



## Roadmap

- v0.1: GPT single-GPU training with normal loss convergence
- v0.2: Tensor Parallel training with multi-GPU loss aligned to single-GPU baseline
- v0.3: ZeRO-1 and activation checkpointing with memory comparison
- v0.4: reusable runtime with config, checkpoint, resume, AMP, and profiling
- v0.5: benchmark results for throughput, memory, and scaling

