"""Aggregate NanoTrain benchmark summaries with derived comparison metrics."""

import argparse
import json
from pathlib import Path
from typing import Any


def _pct_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0}:
        return None
    return (value - baseline) / baseline * 100


def _memory_reduction(value: float | None, baseline: float | None) -> float | None:
    change = _pct_change(value, baseline)
    return -change if change is not None else None


def load_summaries(results_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(results_dir.glob("*/summary.json"))]


def aggregate(results_dir: Path, *, baseline_name: str) -> list[dict[str, Any]]:
    summaries = load_summaries(results_dir)
    baseline = next((summary for summary in summaries if summary["name"] == baseline_name), None)
    baseline_metrics = baseline["metrics"] if baseline is not None else {}
    baseline_tokens = baseline_metrics.get("mean_tokens_per_second")
    baseline_memory = baseline_metrics.get("peak_gpu_memory_mb")
    baseline_step_memory = baseline_metrics.get("step_peak_gpu_memory_mb")

    rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        tokens = metrics.get("mean_tokens_per_second")
        world_size = metrics.get("world_size") or 1
        speedup = tokens / baseline_tokens if tokens is not None and baseline_tokens else None
        rows.append(
            {
                "name": summary["name"],
                "mode": metrics.get("mode"),
                "world_size": world_size,
                "mean_tokens_per_second": tokens,
                "mean_step_time_ms": metrics.get("mean_step_time_ms"),
                "mean_forward_time_ms": metrics.get("mean_forward_time_ms"),
                "mean_backward_time_ms": metrics.get("mean_backward_time_ms"),
                "mean_optimizer_time_ms": metrics.get("mean_optimizer_time_ms"),
                "mean_communication_time_ms": metrics.get("mean_communication_time_ms"),
                "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
                "step_peak_gpu_memory_mb": metrics.get("step_peak_gpu_memory_mb"),
                "parameter_memory_mb": metrics.get("parameter_memory_mb"),
                "gradient_memory_mb": metrics.get("gradient_memory_mb"),
                "optimizer_state_memory_mb": metrics.get("optimizer_state_memory_mb"),
                "final_train_loss": metrics.get("final_train_loss"),
                "final_eval_val_loss": metrics.get("final_eval_val_loss"),
                "speedup_vs_baseline": speedup,
                "scaling_efficiency_percent": speedup / world_size * 100 if speedup else None,
                "peak_memory_reduction_vs_baseline_percent": _memory_reduction(
                    metrics.get("peak_gpu_memory_mb"),
                    baseline_memory,
                ),
                "step_peak_memory_reduction_vs_baseline_percent": _memory_reduction(
                    metrics.get("step_peak_gpu_memory_mb"),
                    baseline_step_memory,
                ),
            }
        )
    return rows


def write_outputs(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    (results_dir / "aggregate.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# NanoTrain Benchmark Aggregate", ""]
    for row in rows:
        lines.append(
            "- {name}: mode={mode}, world_size={world_size}, tokens/s={tokens}, "
            "speedup={speedup}, efficiency={efficiency}%, peak_gpu_mb={memory}, "
            "step_peak_gpu_mb={step_memory}, final_val_loss={val_loss}".format(
                name=row["name"],
                mode=row["mode"],
                world_size=row["world_size"],
                tokens=row["mean_tokens_per_second"],
                speedup=row["speedup_vs_baseline"],
                efficiency=row["scaling_efficiency_percent"],
                memory=row["peak_gpu_memory_mb"],
                step_memory=row["step_peak_gpu_memory_mb"],
                val_loss=row["final_eval_val_loss"],
            )
        )
    (results_dir / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate NanoTrain benchmark summaries.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="single_gpu_bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = aggregate(args.results_dir, baseline_name=args.baseline_name)
    write_outputs(args.results_dir, rows)
    print(f"Wrote aggregate benchmark summary to {args.results_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
