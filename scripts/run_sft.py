#!/usr/bin/env python3
"""Run SFT training. Execute from the repo root:

    python -m scripts.run_sft --config experiments/mistral-nemo-12b-v1/configs/sft_config.yaml
"""

import argparse
import sys
import os

# Add backend to path so we can import app.training.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def main():
    parser = argparse.ArgumentParser(description="AlvinAI SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SFT config YAML (e.g., experiments/mistral-nemo-12b-v1/configs/sft_config.yaml)",
    )
    args = parser.parse_args()

    from app.training.sft_trainer import run_sft
    from app.training.experiment_tracker import update_experiment_status

    try:
        run_sft(args.config)
        update_experiment_status(args.config, "sft", "sft_done")
    except Exception as e:
        update_experiment_status(args.config, "sft", "failed")
        raise


if __name__ == "__main__":
    main()
