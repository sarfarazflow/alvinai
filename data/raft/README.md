# RAFT Dataset

Retrieval-augmented fine-tuning data. See CLAUDE.md for dataset construction notes.

## Contents

| File | Records |
|---|---|
| `train.jsonl` | 3,200 |
| `val.jsonl` | 800 |

## Format

OpenAI messages format with context documents embedded in user prompt.
Metadata includes `type` (`oracle` or `no_oracle`), `oracle_doc` ID, `category`.

Answers use `##begin_quote## ... ##end_quote##` citation markers.
