#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-nanotrain}"
RESULTS_DIR="${RESULTS_DIR:-benchmark/results/memory_stress}"
ZERO_CONFIG="${ZERO_CONFIG:-configs/bench_gpt_zero_memory.yaml}"
ACT_CONFIG="${ACT_CONFIG:-configs/bench_gpt_memory_stress.yaml}"
BENCH_ITERS="${BENCH_ITERS:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
EVAL_ITERS="${EVAL_ITERS:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
SINGLE_CUDA_DEVICE="${SINGLE_CUDA_DEVICE:-0}"

PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
TORCHRUN_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" torchrun)

COMMON_OVERRIDES=(
  --override "train.max_iters=${BENCH_ITERS}"
  --override "train.eval_interval=${EVAL_INTERVAL}"
  --override "train.eval_iters=${EVAL_ITERS}"
  --override "train.log_interval=${LOG_INTERVAL}"
  --override "train.always_save_checkpoint=false"
)

run_single_activation() {
  local name="$1"
  local enabled="$2"
  CUDA_VISIBLE_DEVICES="${SINGLE_CUDA_DEVICE}" "${PYTHON_CMD[@]}" benchmark.py \
    --name "${name}" \
    --results-dir "${RESULTS_DIR}" \
    --config "${ACT_CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --override "runtime.activation_checkpointing=${enabled}" \
    --override "distributed.zero_stage=0" \
    --override "distributed.tp_size=1" \
    --override "train.gradient_accumulation_steps=1"
}

run_zero_mode() {
  local name="$1"
  local zero_stage="$2"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    benchmark.py \
    --name "${name}" \
    --results-dir "${RESULTS_DIR}" \
    --config "${ZERO_CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --override "distributed.zero_stage=${zero_stage}" \
    --override "distributed.tp_size=1" \
    --override "train.gradient_accumulation_steps=${NPROC_PER_NODE}"
}

run_single_activation "activation_off" "false"
run_single_activation "activation_on" "true"
run_zero_mode "ddp_${NPROC_PER_NODE}gpu_memory" "0"
run_zero_mode "zero1_${NPROC_PER_NODE}gpu_memory" "1"
run_zero_mode "zero2_${NPROC_PER_NODE}gpu_memory" "2"

"${PYTHON_CMD[@]}" benchmark/aggregate_results.py \
  --results-dir "${RESULTS_DIR}" \
  --baseline-name "ddp_${NPROC_PER_NODE}gpu_memory"
