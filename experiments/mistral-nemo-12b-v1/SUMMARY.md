# Mistral Nemo 12B v1 — Fine-tuning Summary

**Experiment:** `mistral-nemo-12b-v1`
**Model:** `mistralai/Mistral-Nemo-Instruct-2407` (12.3B params)
**Hardware:** RunPod A40 48GB VRAM
**Date:** 2026-04-03
**Total Training Time:** ~7.2 hours | **Cost:** ~$5.69

---

## Pipeline: SFT → RAFT → DPO

### Stage 1: SFT (Supervised Fine-Tuning)

| Metric | Value |
|---|---|
| Data | 9,398 train / 2,388 val |
| Duration | ~100 min |
| Steps | 882 planned, best at 200 |
| Best Eval Loss | **1.022** (epoch 0.34) |
| Trainable Params | 57M (0.46% of 12.3B) |

**Key Finding:** Overfitting after epoch 0.34. Template-based data lacks diversity.
Best checkpoint is step 200 — later training was counterproductive.

### Stage 2: RAFT (Retrieval-Augmented Fine-Tuning)

| Metric | Value |
|---|---|
| Data | 16,578 train / 4,145 val |
| Duration | ~314 min |
| Steps | 519 |
| Best Eval Loss | **0.113** |

**Key Finding:** No overfitting. Eval loss decreased continuously from 0.414 to 0.113.
Context documents provide natural variety that prevents memorization.

### Stage 3: DPO (Direct Preference Optimization)

| Metric | Value |
|---|---|
| Data | 1,992 train / 499 val |
| Duration | ~18 min |
| Steps | 63 |
| Train Loss (avg) | **0.0204** |
| Eval Loss | **0.0138** |
| Reward Accuracy (train) | **100%** |
| Reward Accuracy (eval) | **99.8%** |
| Reward Margin (final) | **18.55** |

**Key Finding:** Strong preference separation. Model clearly distinguishes
chosen vs rejected responses. No reward hacking — eval tracks train closely.

---

## RAFT Validation (Quality Gates)

| Metric | SFT-only | RAFT-trained | Target | Status |
|---|---|---|---|---|
| Oracle Citation Rate | 0.587 | **1.000** | >= 0.80 | PASS |
| Distractor Rejection Rate | 1.000 | **1.000** | >= 0.80 | PASS |
| Abstention Rate | 0.000 | **0.946** | >= 0.80 | PASS |

**Critical Insight:** SFT-only model is dangerous for RAG — 0% abstention rate
means it always fabricates answers when context is insufficient. RAFT brought
abstention to 94.6%, making the model production-safe.

---

## Training Configuration (all stages)

| Parameter | SFT | RAFT | DPO |
|---|---|---|---|
| LoRA r / alpha | 16 / 32 | 16 / 32 | 16 / 32 |
| Batch size | 2 | 2 | 2 |
| Grad accumulation | 16 | 16 | 16 |
| Effective batch | 32 | 32 | 32 |
| Learning rate | 1.5e-4 | 3e-5 | 1e-5 |
| Max seq length | 4096 | 4096 | 1024 |
| Scheduler | cosine | cosine | cosine |
| VRAM peak | 16 GB | 17 GB | ~16 GB |

---

## Artifacts on RunPod Pod

```
/workspace/alvinai/experiments/mistral-nemo-12b-v1/models/
  sft_checkpoint/checkpoint-200/         # LoRA adapter (best SFT)
  raft_checkpoint/                       # LoRA adapter (final RAFT)
  raft_checkpoint_merged/                # Full 16-bit merged model (9.2GB)
  dpo_checkpoint/                        # LoRA adapter (final DPO)
  dpo_checkpoint_merged/                 # Full 16-bit merged model (9.2GB)
```

---

## Next Steps

1. **RAGAS Evaluation** — Run against DPO model using generated eval datasets (285 questions, 6 namespaces)
2. **AWQ Quantization** — Compress merged 16-bit model to 4-bit for vLLM serving
3. **Upload to HuggingFace** — Push versioned model to `tbqguy/alvinai-nemo-12b-awq-YYYYMMDD`
4. **Production Deployment** — Serve via vLLM with blue-green swap

---

## W&B Dashboard

https://wandb.ai/sarfarazflow-/alvinai-training
