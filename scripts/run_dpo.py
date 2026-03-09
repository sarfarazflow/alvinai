#!/usr/bin/env python3
"""Run DPO training. Execute from the repo root:

    python -m scripts.run_dpo --config configs/dpo_config.yaml
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
        default="configs/dpo_config.yaml",
        help="Path to DPO config YAML",
    )
    args = parser.parse_args()

    from app.training.dpo_trainer import run_dpo

    run_dpo(args.config)


if __name__ == "__main__":
    main()
