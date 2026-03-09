"""DPO dataset loader — loads prompt/chosen/rejected pairs for preference optimization."""

import json
from datasets import Dataset


def load_dpo_jsonl(path: str) -> Dataset:
    """Load a JSONL file with DPO pairs into a HF Dataset.

    Expected format per line:
        {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return Dataset.from_list(records)


def prepare_dpo_dataset(train_path: str, val_path: str) -> tuple[Dataset, Dataset]:
    """Load DPO train/val datasets."""
    train_ds = load_dpo_jsonl(train_path)
    val_ds = load_dpo_jsonl(val_path)
    return train_ds, val_ds
