#!/usr/bin/env python3
"""Upload a model to HuggingFace Hub with a versioned tag.

Usage:
    python -m scripts.upload_model_to_hf \
        --local-path experiments/mistral-nemo-12b-v1/models/production/ \
        --repo-id tbqguy/alvinai-nemo-12b-awq-20260401 \
        --commit-message "Nemo 12B DPO production model 2026-04-01"
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--local-path",
        type=str,
        required=True,
        help="Local path to the model directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., tbqguy/alvinai-nemo-12b-awq-20260401)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model",
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating repo {args.repo_id}...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {args.local_path} to {args.repo_id}...")
    api.upload_folder(
        folder_path=args.local_path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    print(f"Done. Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
