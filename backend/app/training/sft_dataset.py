"""SFT dataset loader — converts OpenAI messages format to Mistral chat template."""

import json
from datasets import Dataset


def load_sft_jsonl(path: str) -> Dataset:
    """Load a JSONL file with OpenAI messages format into a HF Dataset."""
    records = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            records.append(row)
    return Dataset.from_list(records)


def format_to_mistral(example: dict) -> dict:
    """Convert OpenAI messages format to Mistral instruction format.

    Input:  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    Output: {"text": "<s>[INST] {system}\n\n{user} [/INST] {assistant}</s>"}
    """
    messages = example["messages"]

    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for msg in messages:
        if msg["role"] == "system":
            system_msg = msg["content"]
        elif msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant":
            assistant_msg = msg["content"]

    # Mistral format: system prompt goes before user message inside [INST]
    if system_msg:
        instruction = f"{system_msg}\n\n{user_msg}"
    else:
        instruction = user_msg

    text = f"<s>[INST] {instruction} [/INST] {assistant_msg}</s>"
    return {"text": text}


def prepare_sft_dataset(train_path: str, val_path: str) -> tuple[Dataset, Dataset]:
    """Load and format SFT train/val datasets."""
    train_ds = load_sft_jsonl(train_path)
    val_ds = load_sft_jsonl(val_path)

    train_ds = train_ds.map(format_to_mistral, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_to_mistral, remove_columns=val_ds.column_names)

    return train_ds, val_ds
