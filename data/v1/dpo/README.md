# DPO Dataset

Direct preference optimization data. See CLAUDE.md for dataset construction notes.

## Contents

| File | Records |
|---|---|
| `train.jsonl` | 560 |
| `val.jsonl` | 140 |

## Format

DPO pairs format: `{"prompt":..., "chosen":..., "rejected":..., "metadata": {...}}`.

Split from 700 total pairs (80/20 train/val, seed=42).
Categories include `refusal`, `safety_scope`, and domain-specific types.
