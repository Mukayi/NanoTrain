"""NanoTrain training entrypoint."""

import argparse
from pathlib import Path

from nanotrain.config import apply_overrides, load_config
from nanotrain.runtime import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT model with NanoTrain.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gpt_single_gpu.yaml"),
        help="Path to a NanoTrain YAML config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="Override config values, e.g. --override train.max_iters=10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = apply_overrides(load_config(args.config), args.override)
    run_training(config, project_root=project_root)


if __name__ == "__main__":
    main()
