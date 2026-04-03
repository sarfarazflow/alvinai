"""SFT dataset loader — converts OpenAI messages format to model-specific chat templates."""

import json
from functools import partial

from datasets import Dataset


def load_sft_jsonl(path: str) -> Dataset:
    """Load a JSONL file with OpenAI messages format into a HF Dataset."""
    records = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            records.append(row)
    return Dataset.from_list(records)


def _extract_messages(example: dict) -> tuple[str, str, str]:
    """Extract system, user, and assistant messages from OpenAI messages format."""
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
    return system_msg, user_msg, assistant_msg


def format_for_template(example: dict, chat_template: str = "mistral") -> dict:
    """Convert OpenAI messages format to model-specific chat template.

    Supported templates: mistral, gemma, phi, qwen, granite
    """
    system_msg, user_msg, assistant_msg = _extract_messages(example)

    if system_msg:
        instruction = f"{system_msg}\n\n{user_msg}"
    else:
        instruction = user_msg

    if chat_template == "mistral":
        text = f"<s>[INST] {instruction} [/INST] {assistant_msg}</s>"
    elif chat_template == "gemma":
        text = (
            f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant_msg}<end_of_turn>"
        )
    elif chat_template == "phi":
        text = (
            f"<|user|>\n{instruction}<|end|>\n"
            f"<|assistant|>\n{assistant_msg}<|end|>"
        )
    elif chat_template == "qwen":
        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    elif chat_template == "granite":
        text = (
            f"<|start_of_role|>user<|end_of_role|>{instruction}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>{assistant_msg}<|end_of_text|>"
        )
    else:
        raise ValueError(f"Unknown chat template: {chat_template}. "
                         f"Supported: mistral, gemma, phi, qwen, granite")

    return {"text": text}


# Backward-compatible alias
def format_to_mistral(example: dict) -> dict:
    return format_for_template(example, chat_template="mistral")


def prepare_sft_dataset(
    train_path: str, val_path: str, chat_template: str = "mistral"
) -> tuple[Dataset, Dataset]:
    """Load and format SFT train/val datasets."""
    train_ds = load_sft_jsonl(train_path)
    val_ds = load_sft_jsonl(val_path)

    formatter = partial(format_for_template, chat_template=chat_template)
    train_ds = train_ds.map(formatter, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(formatter, remove_columns=val_ds.column_names)

    return train_ds, val_ds
