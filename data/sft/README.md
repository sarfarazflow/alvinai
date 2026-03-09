# SFT Dataset

Supervised fine-tuning data. See CLAUDE.md for dataset construction notes.

## Contents

| File | Records | Source |
|---|---|---|
| `train.jsonl` | 3,200 | 1,600 automotive + 1,600 HR |
| `val.jsonl` | 800 | 400 automotive + 400 HR |

## Format

OpenAI messages format (`{"messages": [{"role":..., "content":...}], "metadata": {...}}`).

Categories: `customer`, `vendor`, `employee`, `engineer`, `compliance`.
