#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-nanotrain}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-benchmark/results/phase6}"
BENCH_ITERS="${BENCH_ITERS:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_ITERS="${EVAL_ITERS:-5}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

COMMON_OVERRIDES=(
  --override "train.max_iters=${BENCH_ITERS}"
  --override "train.eval_interval=${EVAL_INTERVAL}"
  --override "train.eval_iters=${EVAL_ITERS}"
  --override "train.log_interval=${LOG_INTERVAL}"
  --override "train.always_save_checkpoint=false"
)

if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
  TORCHRUN_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" torchrun)
else
  PYTHON_CMD=("${PYTHON_BIN}")
  TORCHRUN_CMD=(torchrun)
fi

run_single() {
  local name="$1"
  local config="$2"
  shift 2
  "${PYTHON_CMD[@]}" benchmark.py \
    --name "${name}" \
    --results-dir "${RESULTS_DIR}" \
    --config "${config}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

run_torchrun() {
  local name="$1"
  local config="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN_CMD[@]}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    benchmark.py \
    --name "${name}" \
    --results-dir "${RESULTS_DIR}" \
    --config "${config}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

echo "Writing Phase 6 benchmark results to ${RESULTS_DIR}"
echo "CONDA_ENV=${CONDA_ENV:-<disabled>}"
echo "BENCH_ITERS=${BENCH_ITERS} EVAL_INTERVAL=${EVAL_INTERVAL} EVAL_ITERS=${EVAL_ITERS}"

run_single "single_cpu" "configs/nanotrain_shakespeare_char_smoke.yaml"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" run_single \
  "single_gpu_bf16" \
  "configs/gpt_ddp_smoke.yaml" \
  --override "runtime.device=cuda" \
  --override "distributed.zero_stage=0" \
  --override "distributed.tp_size=1" \
  --override "train.gradient_accumulation_steps=1"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES%%,*}" run_single \
  "single_gpu_activation_checkpointing" \
  "configs/gpt_ddp_smoke.yaml" \
  --override "runtime.device=cuda" \
  --override "runtime.activation_checkpointing=true" \
  --override "distributed.zero_stage=0" \
  --override "distributed.tp_size=1" \
  --override "train.gradient_accumulation_steps=1"

run_torchrun "ddp_${NPROC_PER_NODE}gpu" "configs/gpt_ddp_smoke.yaml"
run_torchrun "tp_${NPROC_PER_NODE}gpu" "configs/gpt_tp_smoke.yaml"
run_torchrun "zero1_${NPROC_PER_NODE}gpu" "configs/gpt_zero1_smoke.yaml"
run_torchrun "zero2_${NPROC_PER_NODE}gpu" "configs/gpt_zero2_smoke.yaml"

"${PYTHON_CMD[@]}" benchmark/aggregate_results.py \
  --results-dir "${RESULTS_DIR}" \
  --baseline-name "single_gpu_bf16"
