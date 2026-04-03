# AlvinAI Model Comparison

Generated: 2026-04-01 19:31 | Experiments: 7

## Training Summary

| Experiment | Status | SFT Loss | SFT Time | RAFT Loss | RAFT Time | DPO Reward Acc | DPO Time | Cost |
|---|---|---|---|---|---|---|---|---|
| gemma-3-12b-v1 | pending | — | — | — | — | — | — | — |
| granite-4.0-small-v1 | pending | — | — | — | — | — | — | — |
| mistral-7b-v1 | completed | 0.1395 | 18 min | 0.0560 | 37 min | 84.1% | 3 min | 0.79$ |
| mistral-nemo-12b-v1 | pending | — | — | — | — | — | — | — |
| mistral-small-24b-v1 | pending | — | — | — | — | — | — | — |
| phi-4-14b-v1 | pending | — | — | — | — | — | — | — |
| qwen3-14b-v1 | pending | — | — | — | — | — | — | — |

## RAFT Quality Gates

All thresholds must be >= 0.80 to proceed to DPO.

| Experiment | Oracle Citation | Distractor Rejection | Abstention | Passed |
|---|---|---|---|---|
| gemma-3-12b-v1 | — | — | — | — |
| granite-4.0-small-v1 | — | — | — | — |
| mistral-7b-v1 | — | — | — | — |
| mistral-nemo-12b-v1 | — | — | — | — |
| mistral-small-24b-v1 | — | — | — | — |
| phi-4-14b-v1 | — | — | — | — |
| qwen3-14b-v1 | — | — | — | — |

## RAGAS Evaluation

Targets: Faithfulness >= 0.88, Answer Relevance >= 0.85, Context Precision >= 0.80, Context Recall >= 0.82

| Experiment | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|---|
| gemma-3-12b-v1 | — | — | — | — |
| granite-4.0-small-v1 | — | — | — | — |
| mistral-7b-v1 | — | — | — | — |
| mistral-nemo-12b-v1 | — | — | — | — |
| mistral-small-24b-v1 | — | — | — | — |
| phi-4-14b-v1 | — | — | — | — |
| qwen3-14b-v1 | — | — | — | — |

## Latency Benchmarks (AWQ on vLLM)

| Experiment | Factual P95 | Doc Search P95 (warm) | Doc Search P95 (cold) | Tokens/s |
|---|---|---|---|---|
| gemma-3-12b-v1 | — | — | — | — |
| granite-4.0-small-v1 | — | — | — | — |
| mistral-7b-v1 | — | — | — | — |
| mistral-nemo-12b-v1 | — | — | — | — |
| mistral-small-24b-v1 | — | — | — | — |
| phi-4-14b-v1 | — | — | — | — |
| qwen3-14b-v1 | — | — | — | — |

## Winner Selection

```
Tier 1 — 7-8B:   Mistral 7B (baseline)
Tier 2 — 12B:    Best of {Nemo 12B, Gemma 3 12B}
Tier 3 — 14B:    Best of {Phi-4 14B, Qwen3 14B}
Tier 4 — 24B+:   Best of {Mistral Small 24B, Granite 4.0 Small}

Decision: If 14B ≈ 12B (<2% improvement) → stay at 12B (cheaper inference)
```
