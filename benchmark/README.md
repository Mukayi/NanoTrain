# Benchmark

This directory contains reproducible throughput, memory, communication, and
scaling benchmark outputs.

## Commands

Run DDP scaling on 1/2/4/8 GPUs:

```bash
RESULTS_DIR=benchmark/results/scaling_medium \
BENCH_ITERS=1000 \
EVAL_INTERVAL=200 \
EVAL_ITERS=20 \
LOG_INTERVAL=10 \
bash scripts/run_scaling_benchmarks.sh
```

Run memory experiments for activation checkpointing and ZeRO-1/ZeRO-2:

```bash
RESULTS_DIR=benchmark/results/memory_stress \
BENCH_ITERS=500 \
EVAL_INTERVAL=100 \
EVAL_ITERS=10 \
LOG_INTERVAL=10 \
bash scripts/run_memory_benchmarks.sh
```

Run Tensor Parallel medium experiments:

```bash
RESULTS_DIR=benchmark/results/tp_medium \
BENCH_ITERS=500 \
EVAL_INTERVAL=100 \
EVAL_ITERS=10 \
LOG_INTERVAL=10 \
bash -lc 'set -euo pipefail; PY="conda run --no-capture-output -n nanotrain python"; TR="conda run --no-capture-output -n nanotrain torchrun"; COMMON=(--results-dir "$RESULTS_DIR" --config configs/bench_gpt_tp_medium.yaml --override train.max_iters=$BENCH_ITERS --override train.eval_interval=$EVAL_INTERVAL --override train.eval_iters=$EVAL_ITERS --override train.log_interval=$LOG_INTERVAL --override train.always_save_checkpoint=false); CUDA_VISIBLE_DEVICES=0 $PY benchmark.py --name single_gpu_tp_medium "${COMMON[@]}" --override distributed.tp_size=1 --override train.gradient_accumulation_steps=1; CUDA_VISIBLE_DEVICES=0,1 $TR --standalone --nproc_per_node=2 benchmark.py --name tp_2gpu_medium "${COMMON[@]}" --override distributed.tp_size=2 --override train.gradient_accumulation_steps=1; CUDA_VISIBLE_DEVICES=0,1,2,3 $TR --standalone --nproc_per_node=4 benchmark.py --name tp_4gpu_medium "${COMMON[@]}" --override distributed.tp_size=4 --override train.gradient_accumulation_steps=1; $PY benchmark/aggregate_results.py --results-dir "$RESULTS_DIR" --baseline-name single_gpu_tp_medium'
```

Run Nsight Systems profiles:

```bash
PROFILE_DIR=benchmark/profiles \
PROFILE_ITERS=20 \
EVAL_INTERVAL=1000000 \
EVAL_ITERS=1 \
LOG_INTERVAL=10 \
bash scripts/run_nsys_profiles.sh
```

## Current Results

- `benchmark/results/scaling_medium/aggregate.json`: DDP scaling reached
  376K tokens/s on 1 GPU and 2.14M tokens/s on 8 GPUs, for 5.70x speedup and
  71.2% scaling efficiency.
- `benchmark/results/memory_stress/aggregate.json`: activation checkpointing
  reduced single-GPU step peak memory from 1866 MB to 1410 MB; ZeRO-2 reduced
  2-GPU step peak memory from 2183 MB to 1483 MB and gradient memory from
  326 MB to 135 MB.
- `benchmark/results/tp_medium/aggregate.json`: pure Tensor Parallel reduced
  per-rank step peak memory from 1256 MB on 1 GPU to 732 MB on 2 GPUs and
  464 MB on 4 GPUs.

OpenWebText/GPT-2 124M configs live in `configs/bench_gpt2_owt*.yaml`. They
require `nanoGPT/data/openwebtext/train.bin` and `val.bin`, which are not
currently present in this workspace.

