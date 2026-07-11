"""NanoTrain benchmark entrypoint with structured result recording."""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from nanotrain.config import apply_overrides, load_config
from nanotrain.runtime import run_training


def _is_master_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _default_run_name(config_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{config_path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a NanoTrain training benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nanotrain_shakespeare_char_smoke.yaml"),
        help="Path to a NanoTrain YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override config values, e.g. --override train.max_iters=10.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Benchmark run name. Use a stable name when launching with torchrun.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Directory where benchmark result folders are written.",
    )
    parser.add_argument(
        "--summary-warmup-steps",
        type=int,
        default=1,
        help="Number of initial train-step metric records to exclude from averages.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else None


def _max(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return max(values) if values else None


def _hardware_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }


def _summarize_metrics(
    *,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
    warmup_steps: int,
) -> dict[str, Any]:
    train_steps = [record for record in records if record.get("type") == "train_step"]
    eval_steps = [record for record in records if record.get("type") == "eval"]
    averaged_steps = train_steps[warmup_steps:] if len(train_steps) > warmup_steps else train_steps
    final_train = train_steps[-1] if train_steps else {}
    final_eval = eval_steps[-1] if eval_steps else {}

    return {
        "elapsed_seconds": elapsed_seconds,
        "num_train_step_records": len(train_steps),
        "num_eval_records": len(eval_steps),
        "summary_warmup_steps": warmup_steps,
        "mode": final_train.get("mode") or final_eval.get("mode"),
        "world_size": final_train.get("world_size") or final_eval.get("world_size"),
        "final_iter": final_train.get("iter") or final_eval.get("iter"),
        "final_train_loss": final_train.get("loss"),
        "final_eval_train_loss": final_eval.get("train_loss"),
        "final_eval_val_loss": final_eval.get("val_loss"),
        "mean_step_time_ms": _mean(averaged_steps, "step_time_ms"),
        "mean_forward_time_ms": _mean(averaged_steps, "forward_time_ms"),
        "mean_backward_time_ms": _mean(averaged_steps, "backward_time_ms"),
        "mean_optimizer_time_ms": _mean(averaged_steps, "optimizer_time_ms"),
        "mean_communication_time_ms": _mean(averaged_steps, "communication_time_ms"),
        "mean_tokens_per_second": _mean(averaged_steps, "tokens_per_second"),
        "mean_samples_per_second": _mean(averaged_steps, "samples_per_second"),
        "peak_gpu_memory_mb": _max(train_steps, "peak_gpu_memory_mb"),
        "step_peak_gpu_memory_mb": _max(train_steps, "step_peak_gpu_memory_mb"),
        "parameter_memory_mb": _max(train_steps, "parameter_memory_mb"),
        "gradient_memory_mb": _max(train_steps, "gradient_memory_mb"),
        "optimizer_state_memory_mb": _max(train_steps, "optimizer_state_memory_mb"),
        "activation_memory_estimate_mb": _max(train_steps, "activation_memory_estimate_mb"),
    }


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# NanoTrain Benchmark Summary",
        "",
        f"- Name: `{summary['name']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Mode: `{summary['metrics'].get('mode')}`",
        f"- World size: `{summary['metrics'].get('world_size')}`",
        f"- Elapsed seconds: `{summary['metrics'].get('elapsed_seconds'):.2f}`",
        f"- Mean tokens/s: `{summary['metrics'].get('mean_tokens_per_second')}`",
        f"- Mean step time ms: `{summary['metrics'].get('mean_step_time_ms')}`",
        f"- Final train loss: `{summary['metrics'].get('final_train_loss')}`",
        f"- Final val loss: `{summary['metrics'].get('final_eval_val_loss')}`",
        f"- Peak GPU memory MB: `{summary['metrics'].get('peak_gpu_memory_mb')}`",
        "",
        "Full per-step records are in `metrics.jsonl`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    run_name = args.name or _default_run_name(args.config)
    run_dir = (project_root / args.results_dir / run_name).resolve()
    out_dir = run_dir / "out"

    config = apply_overrides(load_config(args.config), args.override)
    if not any(override.startswith("train.out_dir=") for override in args.override):
        config.train.out_dir = str(out_dir)

    start = time.perf_counter()
    run_training(config, project_root=project_root)
    elapsed = time.perf_counter() - start

    if _is_master_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(config.train.out_dir) / "metrics.jsonl"
        records = _read_jsonl(metrics_path)
        summary = {
            "name": run_name,
            "config_path": str(args.config),
            "results_dir": str(run_dir),
            "out_dir": str(config.train.out_dir),
            "overrides": args.override,
            "config": asdict(config),
            "hardware": _hardware_info(),
            "metrics": _summarize_metrics(
                records=records,
                elapsed_seconds=elapsed,
                warmup_steps=args.summary_warmup_steps,
            ),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_summary_markdown(run_dir / "summary.md", summary)
        print(f"benchmark elapsed: {elapsed:.2f}s")
        print(f"benchmark summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
