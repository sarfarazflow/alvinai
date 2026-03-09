#!/usr/bin/env python3
"""Run RAFT training. Execute from the repo root:

    python -m scripts.run_raft --config configs/raft_config.yaml
"""

import argparse
import sys
import os

# Add backend to path so we can import app.training.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def main():
    parser = argparse.ArgumentParser(description="AlvinAI RAFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/raft_config.yaml",
        help="Path to RAFT config YAML",
    )
    args = parser.parse_args()

    from app.training.raft_trainer import run_raft

    run_raft(args.config)


if __name__ == "__main__":
    main()
