# AlvinAI Training Results

## SFT — Stage 1: Supervised Fine-Tuning

**Date:** 2026-03-09
**GPU:** RTX A6000 (48GB VRAM)
**Duration:** ~18 minutes (300 steps)

### Config
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Method: QLoRA (4-bit NF4)
- LoRA r/alpha: 16/32
- Learning rate: 2e-4 (cosine schedule)
- Epochs: 3
- Batch size: 8, grad_accum: 4 (effective batch = 32)
- Dataset: 3,200 train / 800 val

### Loss Curve

| Step | Loss | Grad Norm | LR | Epoch |
|------|------|-----------|-----|-------|
| 10 | 2.471 | — | — | 0.3 |
| 20 | 0.433 | — | — | 0.7 |
| 30 | 0.252 | — | — | 1.0 |
| 40 | 0.167 | — | — | 1.3 |
| 50 | 0.150 | — | — | 1.7 |
| 60 | 0.144 | — | — | 2.0 |
| 70 | 0.100 | — | — | 2.3 |
| 80 | 0.092 | — | — | 2.7 |
| 90 | 0.087 | — | — | 3.0 |

### Final Metrics
- **Train loss (avg):** 0.2621
- **Eval loss:** 0.1395
- **Convergence:** Loss stabilized by end of epoch 1

### Artifacts
- LoRA adapter: `models/sft_checkpoint/` (161MB)
- Merged model: `models/sft_checkpoint_merged/` (14GB, 3 safetensor shards)

---

## RAFT — Stage 2: Retrieval-Augmented Fine-Tuning

**Date:** 2026-03-10
**GPU:** RTX A6000 (48GB VRAM)
**Duration:** ~37 minutes (200 steps)

### Config
- Base model: `models/sft_checkpoint` (from SFT)
- Method: QLoRA (4-bit NF4)
- LoRA r/alpha: 16/32
- Learning rate: 5e-5 (cosine schedule)
- Epochs: 2
- Batch size: 2, grad_accum: 16 (effective batch = 32)
- Dataset: 3,200 train / 800 val

### Loss Curve

| Step | Loss | Grad Norm | LR | Epoch |
|------|------|-----------|-----|-------|
| 10 | 1.292 | 1.853 | 4.50e-05 | 0.1 |
| 20 | 0.383 | 0.611 | 4.97e-05 | 0.2 |
| 30 | 0.124 | 0.331 | 4.88e-05 | 0.3 |
| 40 | 0.079 | 0.304 | 4.72e-05 | 0.4 |
| 50 | 0.065 | 0.148 | 4.50e-05 | 0.5 |
| 60 | 0.061 | 0.153 | 4.22e-05 | 0.6 |
| 70 | 0.059 | 0.153 | 3.90e-05 | 0.7 |
| 80 | 0.058 | 0.163 | 3.54e-05 | 0.8 |
| 90 | 0.057 | 0.102 | 3.15e-05 | 0.9 |
| 100 | 0.057 | 0.201 | 2.75e-05 | 1.0 |
| 110 | 0.056 | 0.141 | 2.34e-05 | 1.1 |
| 120 | 0.056 | 0.141 | 1.93e-05 | 1.2 |
| 130 | 0.056 | 0.115 | 1.53e-05 | 1.3 |
| 140 | 0.056 | 0.104 | 1.17e-05 | 1.4 |
| 150 | 0.056 | 0.127 | 8.37e-06 | 1.5 |
| 160 | 0.056 | 0.112 | 5.53e-06 | 1.6 |
| 170 | 0.055 | 0.129 | 3.21e-06 | 1.7 |
| 180 | 0.055 | 0.132 | 1.49e-06 | 1.8 |
| 190 | 0.056 | 0.135 | 4.12e-07 | 1.9 |
| 200 | 0.056 | 0.114 | 3.42e-09 | 2.0 |

### Final Metrics
- **Train loss (avg):** 0.1396
- **Eval loss (epoch 1):** 0.0571
- **Eval loss (epoch 2):** 0.0560
- **Convergence:** Loss stabilized at ~0.057 by epoch 0.7
- **Train runtime:** 2,355 seconds (~39 min)

### Artifacts
- LoRA adapter: `models/raft_checkpoint/` (648MB)
- Merged model: `models/raft_checkpoint_merged/` (14GB, 3 safetensor shards)
- Note: Initial `merge_and_unload()` failed with transformers 5.2 `NotImplementedError`. Fixed using Unsloth's `save_pretrained_merged()`.

### Observations
- Initial OOM at batch_size=8 (RAFT sequences are 3-4K tokens). Fixed by reducing to batch_size=2 with grad_accum=16.
- Loss converged very quickly in epoch 1; epoch 2 provided minimal additional improvement (~0.057 → 0.055).
- Gradient norms remained low throughout (0.10-0.20), indicating stable training.

---

## DPO — Stage 3: Direct Preference Optimization

**Date:** 2026-03-10
**GPU:** RTX A6000 (48GB VRAM)
**Duration:** ~2 min 45 sec (18 steps)

### Config
- Base model: `models/raft_checkpoint` (from RAFT)
- Method: QLoRA (4-bit NF4)
- LoRA r/alpha: 16/32
- Learning rate: 1e-5 (cosine schedule)
- Epochs: 1
- Batch size: 2, grad_accum: 16 (effective batch = 32)
- Dataset: 560 train pairs
- DPO beta: 0.1, loss_type: sigmoid
- Max length: 1024, max prompt length: 256

### Training Log (logged at step 10)

| Metric | Value |
|--------|-------|
| Loss | 0.4482 |
| Grad Norm | 0.3067 |
| Learning Rate | 5.975e-06 |
| Rewards/Chosen | 1.148 |
| Rewards/Rejected | -1.673 |
| Rewards/Accuracies | 0.8406 (84.1%) |
| Rewards/Margins | 2.821 |
| Logps/Chosen | -117.2 |
| Logps/Rejected | -90.13 |
| Logits/Chosen | -3.329 |
| Logits/Rejected | -3.233 |

### Final Metrics
- **Train loss (avg):** 0.2653
- **Rewards accuracy:** 84.1% (model correctly prefers chosen over rejected)
- **Reward margin:** 2.821 (strong separation between chosen/rejected)
- **Train runtime:** 165.9 seconds (~2.75 min)
- **Throughput:** 3.376 samples/sec, 0.109 steps/sec

### Artifacts
- LoRA adapter: `models/dpo_checkpoint/` (409MB)
- Merged model: `models/dpo_checkpoint_merged/` (14GB, 3 safetensor shards)

### Observations
- Required instance-level forward patch to handle Unsloth/DPO 4D attention mask incompatibility.
- 84% reward accuracy indicates strong preference alignment after just 1 epoch.
- Reward margin of 2.821 shows clear separation between chosen and rejected outputs.
- Training was very fast (18 steps) due to small DPO dataset (560 pairs).

---

## Notes
- All training used bf16 precision (A6000 supports bfloat16 natively)
- All stages use Unsloth for 2x faster QLoRA training
- Network volume at `/workspace/` persists across pod restarts
