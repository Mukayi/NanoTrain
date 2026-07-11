#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found on PATH. Install NVIDIA Nsight Systems or load the CUDA toolkit module."
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-nanotrain}"
PROFILE_DIR="${PROFILE_DIR:-benchmark/profiles}"
CONFIG_DDP="${CONFIG_DDP:-configs/bench_gpt_medium.yaml}"
CONFIG_TP="${CONFIG_TP:-configs/bench_gpt_tp_medium.yaml}"
CONFIG_ZERO="${CONFIG_ZERO:-configs/bench_gpt_zero_memory.yaml}"
PROFILE_ITERS="${PROFILE_ITERS:-80}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
EVAL_ITERS="${EVAL_ITERS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

mkdir -p "${PROFILE_DIR}"

profile_torchrun() {
  local name="$1"
  local config="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" nsys profile \
    --force-overwrite true \
    --trace cuda,nvtx,osrt,cudnn,cublas \
    --sample none \
    --output "${PROFILE_DIR}/${name}" \
    conda run --no-capture-output -n "${CONDA_ENV}" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
      benchmark.py \
      --name "${name}" \
      --results-dir "${PROFILE_DIR}/metrics" \
      --config "${config}" \
      --override "train.max_iters=${PROFILE_ITERS}" \
      --override "train.eval_interval=${EVAL_INTERVAL}" \
      --override "train.eval_iters=${EVAL_ITERS}" \
      --override "train.log_interval=${LOG_INTERVAL}" \
      --override "train.always_save_checkpoint=false" \
      "$@"
}

profile_torchrun "ddp_${NPROC_PER_NODE}gpu_nsys" "${CONFIG_DDP}" \
  --override "distributed.zero_stage=0" \
  --override "distributed.tp_size=1" \
  --override "train.gradient_accumulation_steps=${NPROC_PER_NODE}"

profile_torchrun "tp_${NPROC_PER_NODE}gpu_nsys" "${CONFIG_TP}" \
  --override "distributed.zero_stage=0" \
  --override "distributed.tp_size=${NPROC_PER_NODE}" \
  --override "train.gradient_accumulation_steps=1"

profile_torchrun "zero2_${NPROC_PER_NODE}gpu_nsys" "${CONFIG_ZERO}" \
  --override "distributed.zero_stage=2" \
  --override "distributed.tp_size=1" \
  --override "train.gradient_accumulation_steps=${NPROC_PER_NODE}"
