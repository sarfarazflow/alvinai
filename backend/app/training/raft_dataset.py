"""RAFT dataset loader — converts OpenAI messages format to model-specific chat templates.

RAFT data has the same format as SFT (OpenAI messages with system/user/assistant),
but the user content includes oracle + distractor documents and the assistant
uses ##begin_quote##/##end_quote## citation markers.
"""

import json
from functools import partial

from datasets import Dataset

from app.training.sft_dataset import format_for_template


def load_raft_jsonl(path: str) -> Dataset:
    """Load a JSONL file with OpenAI messages format into a HF Dataset."""
    records = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            records.append(row)
    return Dataset.from_list(records)


# Backward-compatible alias
def format_to_mistral(example: dict) -> dict:
    return format_for_template(example, chat_template="mistral")


def prepare_raft_dataset(
    train_path: str, val_path: str, chat_template: str = "mistral"
) -> tuple[Dataset, Dataset]:
    """Load and format RAFT train/val datasets."""
    train_ds = load_raft_jsonl(train_path)
    val_ds = load_raft_jsonl(val_path)

    formatter = partial(format_for_template, chat_template=chat_template)
    train_ds = train_ds.map(formatter, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(formatter, remove_columns=val_ds.column_names)

    return train_ds, val_ds
