#!/usr/bin/env python3
"""Run DPO training. Execute from the repo root:

    python -m scripts.run_dpo --config experiments/mistral-nemo-12b-v1/configs/dpo_config.yaml
"""

import argparse
import sys
import os

# Add backend to path so we can import app.training.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def main():
    parser = argparse.ArgumentParser(description="AlvinAI DPO Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to DPO config YAML (e.g., experiments/mistral-nemo-12b-v1/configs/dpo_config.yaml)",
    )
    args = parser.parse_args()

    from app.training.dpo_trainer import run_dpo
    from app.training.experiment_tracker import update_experiment_status

    try:
        run_dpo(args.config)
        update_experiment_status(args.config, "dpo", "dpo_done")
    except Exception as e:
        update_experiment_status(args.config, "dpo", "failed")
        raise


if __name__ == "__main__":
    main()
