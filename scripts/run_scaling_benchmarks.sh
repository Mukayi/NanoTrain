#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-nanotrain}"
RESULTS_DIR="${RESULTS_DIR:-benchmark/results/scaling_medium}"
CONFIG="${CONFIG:-configs/bench_gpt_medium.yaml}"
BENCH_ITERS="${BENCH_ITERS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_ITERS="${EVAL_ITERS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CUDA_DEVICES_1="${CUDA_DEVICES_1:-0}"
CUDA_DEVICES_2="${CUDA_DEVICES_2:-0,1}"
CUDA_DEVICES_4="${CUDA_DEVICES_4:-0,1,2,3}"
CUDA_DEVICES_8="${CUDA_DEVICES_8:-0,1,2,3,4,5,6,7}"

PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
TORCHRUN_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" torchrun)

COMMON_OVERRIDES=(
  --override "train.max_iters=${BENCH_ITERS}"
  --override "train.eval_interval=${EVAL_INTERVAL}"
  --override "train.eval_iters=${EVAL_ITERS}"
  --override "train.log_interval=${LOG_INTERVAL}"
  --override "train.always_save_checkpoint=false"
)

run_single() {
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES_1}" "${PYTHON_CMD[@]}" benchmark.py \
    --name "single_gpu_bf16" \
    --results-dir "${RESULTS_DIR}" \
    --config "${CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --override "train.gradient_accumulation_steps=1" \
    --override "distributed.zero_stage=0" \
    --override "distributed.tp_size=1"
}

run_ddp() {
  local nproc="$1"
  local devices="$2"
  CUDA_VISIBLE_DEVICES="${devices}" "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="${nproc}" \
    benchmark.py \
    --name "ddp_${nproc}gpu" \
    --results-dir "${RESULTS_DIR}" \
    --config "${CONFIG}" \
    "${COMMON_OVERRIDES[@]}" \
    --override "train.gradient_accumulation_steps=${nproc}" \
    --override "distributed.zero_stage=0" \
    --override "distributed.tp_size=1"
}

run_single
run_ddp 2 "${CUDA_DEVICES_2}"
run_ddp 4 "${CUDA_DEVICES_4}"
run_ddp 8 "${CUDA_DEVICES_8}"

"${PYTHON_CMD[@]}" benchmark/aggregate_results.py \
  --results-dir "${RESULTS_DIR}" \
  --baseline-name "single_gpu_bf16"
