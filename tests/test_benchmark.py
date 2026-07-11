import importlib.util
import json
from pathlib import Path

from benchmark import _summarize_metrics


def load_aggregate_function():
    module_path = Path("benchmark/aggregate_results.py")
    spec = importlib.util.spec_from_file_location("aggregate_results", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aggregate_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aggregate


def test_summarize_metrics_uses_train_and_eval_records() -> None:
    records = [
        {
            "type": "train_step",
            "iter": 0,
            "mode": "single",
            "world_size": 1,
            "loss": 4.0,
            "step_time_ms": 100.0,
            "forward_time_ms": 20.0,
            "backward_time_ms": 30.0,
            "optimizer_time_ms": 10.0,
            "communication_time_ms": 0.0,
            "tokens_per_second": 1000.0,
            "samples_per_second": 10.0,
            "peak_gpu_memory_mb": None,
            "step_peak_gpu_memory_mb": None,
            "parameter_memory_mb": 1.0,
            "gradient_memory_mb": 1.0,
            "optimizer_state_memory_mb": 2.0,
            "activation_memory_estimate_mb": 0.5,
        },
        {
            "type": "train_step",
            "iter": 1,
            "mode": "single",
            "world_size": 1,
            "loss": 3.5,
            "step_time_ms": 80.0,
            "forward_time_ms": 15.0,
            "backward_time_ms": 25.0,
            "optimizer_time_ms": 8.0,
            "communication_time_ms": 0.0,
            "tokens_per_second": 1250.0,
            "samples_per_second": 12.5,
            "peak_gpu_memory_mb": None,
            "step_peak_gpu_memory_mb": None,
            "parameter_memory_mb": 1.0,
            "gradient_memory_mb": 1.0,
            "optimizer_state_memory_mb": 2.0,
            "activation_memory_estimate_mb": 0.5,
        },
        {
            "type": "eval",
            "iter": 1,
            "train_loss": 3.4,
            "val_loss": 3.6,
        },
    ]

    summary = _summarize_metrics(records=records, elapsed_seconds=1.5, warmup_steps=1)

    assert summary["elapsed_seconds"] == 1.5
    assert summary["num_train_step_records"] == 2
    assert summary["final_train_loss"] == 3.5
    assert summary["final_eval_val_loss"] == 3.6
    assert summary["mean_step_time_ms"] == 80.0
    assert summary["mean_tokens_per_second"] == 1250.0
    assert "step_peak_gpu_memory_mb" in summary


def test_aggregate_results_adds_speedup_and_memory_reduction(tmp_path) -> None:
    for name, tokens, memory in [
        ("single_gpu_bf16", 100.0, 10.0),
        ("ddp_2gpu", 180.0, 12.0),
        ("zero2_2gpu", 160.0, 7.0),
    ]:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "metrics": {
                        "mode": name,
                        "world_size": 2 if "2gpu" in name else 1,
                        "mean_tokens_per_second": tokens,
                        "mean_step_time_ms": 1.0,
                        "peak_gpu_memory_mb": memory,
                        "step_peak_gpu_memory_mb": memory,
                    },
                }
            )
        )

    aggregate = load_aggregate_function()
    rows = aggregate(tmp_path, baseline_name="single_gpu_bf16")
    zero2 = next(row for row in rows if row["name"] == "zero2_2gpu")

    assert zero2["speedup_vs_baseline"] == 1.6
    assert zero2["peak_memory_reduction_vs_baseline_percent"] == 30.0
