#!/usr/bin/env python3
"""Run SFT training. Execute from the repo root:

    python -m scripts.run_sft --config configs/sft_config.yaml
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
        default="configs/sft_config.yaml",
        help="Path to SFT config YAML",
    )
    args = parser.parse_args()

    from app.training.sft_trainer import run_sft

    run_sft(args.config)


if __name__ == "__main__":
    main()
