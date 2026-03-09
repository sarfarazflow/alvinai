# RunPod Fine-Tuning Guide

**GPU:** RTX A6000 (48GB VRAM)
**Pod SSH (TCP):** `ssh root@38.147.83.16 -p 41740 -i ~/.ssh/id_ed25519`

---

## SFT — Stage 1: Supervised Fine-Tuning

### Setup (first time only)

```bash
# Clone repo
cd /workspace
git clone https://github.com/sarfarazflow/alvinai.git
cd alvinai

# Install dependencies
pip install uv
cd backend && uv sync --extra training && cd ..

# Upgrade PyTorch to 2.6+ (required by Unsloth)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Unsloth (must be after torch)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Login to HuggingFace (for gated Mistral model)
python3 -c "from huggingface_hub import login; login()"

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Symlink models to network volume (persists across pod restarts)
mkdir -p /workspace/models
ln -sf /workspace/models ./models
```

### Run SFT Training

```bash
cd /workspace/alvinai

# Run in background (survives SSH disconnect)
nohup python -m scripts.run_sft --config configs/sft_config.yaml \
  > /workspace/sft_training.log 2>&1 &

# Monitor live
tail -f /workspace/sft_training.log

# Check VRAM usage
watch -n1 nvidia-smi
```

### Check Progress (after reconnecting)

```bash
tail -50 /workspace/sft_training.log
```

### SFT Config Summary

| Parameter | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Method | QLoRA (4-bit NF4) |
| LoRA r / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 8 (A6000 48GB) |
| Grad accumulation | 4 (effective batch = 32) |
| Dataset | 3,200 train / 800 val |
| Output | `models/sft_checkpoint/` |

### Expected Output

- Checkpoint saved to `/workspace/models/sft_checkpoint/`
- Merged model at `/workspace/models/sft_checkpoint_merged/`
- Training log at `/workspace/sft_training.log`

---

## RAFT — Stage 2: Retrieval-Augmented Fine-Tuning

### Run RAFT Training

```bash
cd /workspace/alvinai

# Pull latest code (RAFT scripts)
git pull

# Run in background (survives SSH disconnect)
nohup python -m scripts.run_raft --config configs/raft_config.yaml \
  > /workspace/raft_training.log 2>&1 &

# Monitor live
tail -f /workspace/raft_training.log

# Check VRAM usage
watch -n1 nvidia-smi
```

### Check Progress (after reconnecting)

```bash
tail -50 /workspace/raft_training.log
```

### RAFT Config Summary

| Parameter | Value |
|---|---|
| Base model | `models/sft_checkpoint` (from Stage 1) |
| Method | QLoRA (4-bit NF4) |
| LoRA r / alpha | 16 / 32 |
| Learning rate | 5e-5 |
| Epochs | 2 |
| Batch size | 2 (long RAFT sequences) |
| Grad accumulation | 16 (effective batch = 32) |
| Dataset | 3,200 train / 800 val |
| Output | `models/raft_checkpoint/` |

### Expected Output

- Checkpoint saved to `/workspace/models/raft_checkpoint/`
- Merged model at `/workspace/models/raft_checkpoint_merged/`
- Training log at `/workspace/raft_training.log`

### Quality Thresholds (must pass before DPO)

- Oracle citation rate >= 80%
- Distractor rejection rate >= 80%
- Abstention rate >= 80%

---

## DPO — Stage 3: Direct Preference Optimization

### Run DPO Training

```bash
cd /workspace/alvinai

# Pull latest code (DPO scripts)
git pull

# Run in background (survives SSH disconnect)
nohup python -m scripts.run_dpo --config configs/dpo_config.yaml \
  > /workspace/dpo_training.log 2>&1 &

# Monitor live
tail -f /workspace/dpo_training.log
```

### DPO Config Summary

| Parameter | Value |
|---|---|
| Base model | `models/raft_checkpoint` (from Stage 2) |
| Method | QLoRA + DPO (sigmoid loss) |
| Beta (KL penalty) | 0.1 |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Batch size | 2 (DPO uses 2x VRAM) |
| Grad accumulation | 16 (effective batch = 32) |
| Dataset | 560 train / 140 val |
| Output | `models/dpo_checkpoint/` |

### Expected Output

- Checkpoint saved to `/workspace/models/dpo_checkpoint/`
- Merged model at `/workspace/models/dpo_checkpoint_merged/`
- Training log at `/workspace/dpo_training.log`

---

## AWQ Export & HF Upload

*To be updated after DPO completes.*

---

## Important Rules

- **STOP the pod after training.** Idle GPU pods burn money.
- **Network volume** (`/workspace/`) persists. Container storage does not.
- **Never commit model weights to git.** Push datasets and configs only.
- **Monitor VRAM:** `watch -n1 nvidia-smi`
