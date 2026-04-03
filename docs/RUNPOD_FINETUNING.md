# RunPod Fine-Tuning Guide

**GPU:** RTX A5000 (24GB VRAM)
**Pod:** Check RunPod dashboard for current SSH details — pod IP and port change on each restart.

---

## First-Time Pod Setup

```bash
# SSH into the pod (get IP/port from RunPod dashboard)
ssh root@{pod_ip} -p {pod_port} -i ~/.ssh/id_ed25519

# Clone repo
cd /workspace
git clone https://github.com/sarfarazflow/alvinai.git
cd alvinai

# Install dependencies
pip install uv
cd backend && uv sync --extra training && cd ..

# Upgrade PyTorch to 2.6+ (required by Unsloth)
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Unsloth (must be AFTER torch — it detects CUDA at install time)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# Login to HuggingFace (for gated Mistral model)
python3 -c "from huggingface_hub import login; login()"

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA RTX A5000

# Symlink models to network volume (persists across pod restarts)
mkdir -p /workspace/models
ln -sf /workspace/models ./models

# (Optional) Set up W&B for experiment tracking
pip install wandb
wandb login
```

---

## Returning to an Existing Pod

```bash
ssh root@{pod_ip} -p {pod_port} -i ~/.ssh/id_ed25519
cd /workspace/alvinai
git pull origin main    # get latest code + datasets
```

---

## Training Workflow

### Option A: Using Default Configs (quick run)

```bash
# SFT — Stage 1
nohup python -m scripts.run_sft --config configs/sft_config.yaml \
  > /workspace/sft_training.log 2>&1 &
tail -f /workspace/sft_training.log

# RAFT — Stage 2 (after SFT completes)
nohup python -m scripts.run_raft --config configs/raft_config.yaml \
  > /workspace/raft_training.log 2>&1 &
tail -f /workspace/raft_training.log

# DPO — Stage 3 (after RAFT completes)
nohup python -m scripts.run_dpo --config configs/dpo_config.yaml \
  > /workspace/dpo_training.log 2>&1 &
tail -f /workspace/dpo_training.log
```

### Option B: Using Experiment Configs (recommended)

Each model experiment has its own configs under `experiments/`. This keeps results isolated and comparable.

```bash
# Scaffold a new experiment (first time only)
./scripts/new_experiment.sh mistral-nemo-12b v1 v1 \
  --base-model mistralai/Mistral-Nemo-Instruct-2407 --chat-template mistral

# SFT — Stage 1
nohup python -m scripts.run_sft \
  --config experiments/mistral-nemo-12b-v1/configs/sft_config.yaml \
  > /workspace/sft_training.log 2>&1 &
tail -f /workspace/sft_training.log

# RAFT — Stage 2 (after SFT completes)
nohup python -m scripts.run_raft \
  --config experiments/mistral-nemo-12b-v1/configs/raft_config.yaml \
  > /workspace/raft_training.log 2>&1 &
tail -f /workspace/raft_training.log

# DPO — Stage 3 (after RAFT completes)
nohup python -m scripts.run_dpo \
  --config experiments/mistral-nemo-12b-v1/configs/dpo_config.yaml \
  > /workspace/dpo_training.log 2>&1 &
tail -f /workspace/dpo_training.log
```

---

## Monitoring

```bash
# Live training log
tail -f /workspace/sft_training.log

# Check progress after reconnecting
tail -50 /workspace/raft_training.log

# Real-time VRAM usage (should stay under 24GB)
watch -n1 nvidia-smi

# Check if training is still running
ps aux | grep python
```

---

## Config Summary

### SFT — Stage 1

| Parameter | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Method | QLoRA (4-bit NF4) |
| LoRA r / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Batch size | 8 |
| Grad accumulation | 4 (effective batch = 32) |
| Dataset | 3,200 train / 800 val |
| Output | `models/sft_checkpoint/` |

### RAFT — Stage 2

| Parameter | Value |
|---|---|
| Base model | `models/sft_checkpoint` (from Stage 1) |
| Method | QLoRA (4-bit NF4) |
| LoRA r / alpha | 16 / 32 |
| Learning rate | 5e-5 |
| Epochs | 2 |
| Batch size | 2 (RAFT sequences are 3-4K tokens — OOM at higher values) |
| Grad accumulation | 16 (effective batch = 32) |
| Dataset | 3,200 train / 800 val |
| Output | `models/raft_checkpoint/` |

**Quality gates (must pass before DPO):**
- Oracle citation rate >= 0.80
- Distractor rejection rate >= 0.80
- Abstention rate >= 0.80

### DPO — Stage 3

| Parameter | Value |
|---|---|
| Base model | `models/raft_checkpoint` (from Stage 2) |
| Method | QLoRA + DPO (sigmoid loss) |
| Beta (KL penalty) | 0.1 |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Batch size | 2 (DPO runs 2 forward passes per example) |
| Grad accumulation | 16 (effective batch = 32) |
| Dataset | 560 train / 140 val |
| Output | `models/dpo_checkpoint/` |

---

## AWQ Export

After DPO completes, quantise the merged model for production serving:

```bash
python -m scripts.export_awq \
  --input models/dpo_checkpoint_merged/ \
  --output models/production/alvinai-7b-awq/

# For experiment-based workflow:
python -m scripts.export_awq \
  --input experiments/mistral-nemo-12b-v1/models/dpo_checkpoint_merged/ \
  --output experiments/mistral-nemo-12b-v1/models/production/
```

This produces a ~3.5GB AWQ 4-bit model. Verify it:

```bash
ls -lh models/production/alvinai-7b-awq/
# Should contain: model.safetensors, tokenizer files, config.json
```

---

## Upload to HuggingFace Hub

```bash
python -m scripts.upload_model_to_hf \
  --local-path models/production/alvinai-7b-awq/ \
  --repo-id tbqguy/alvinai-7b-awq-$(date +%Y%m%d) \
  --commit-message "DPO production model $(date +%Y-%m-%d)"

# For experiment-based workflow:
python -m scripts.upload_model_to_hf \
  --local-path experiments/mistral-nemo-12b-v1/models/production/ \
  --repo-id tbqguy/alvinai-nemo-12b-awq-$(date +%Y%m%d) \
  --commit-message "Nemo 12B DPO production model $(date +%Y-%m-%d)"

# Use --private for private repos
```

---

## Expected Output Per Stage

| Stage | Checkpoint | Merged Model | Log |
|---|---|---|---|
| SFT | `models/sft_checkpoint/` | `models/sft_checkpoint_merged/` | `/workspace/sft_training.log` |
| RAFT | `models/raft_checkpoint/` | `models/raft_checkpoint_merged/` | `/workspace/raft_training.log` |
| DPO | `models/dpo_checkpoint/` | `models/dpo_checkpoint_merged/` | `/workspace/dpo_training.log` |
| AWQ | — | `models/production/alvinai-7b-awq/` | — |

---

## Actual Training Results (A5000, March 2026)

| Stage | Duration | Steps | Final Loss |
|---|---|---|---|
| SFT (3 epochs, 3.2K pairs) | ~18 min | 300 | 0.1395 (eval) |
| RAFT (2 epochs, 3.2K pairs) | ~37 min | 200 | 0.0560 (eval) |
| DPO (1 epoch, 560 pairs) | ~3 min | 18 | 84.1% reward accuracy |
| **Total** | **~58 min** | **518** | |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| OOM during RAFT | Reduce `batch_size` to 1, increase `grad_accum` to 32. If still OOM, reduce `max_seq_length` to 3072. |
| OOM during DPO | Reduce `batch_size` to 1, increase `grad_accum` to 32. |
| `merge_and_unload()` fails | Use `model.save_pretrained_merged()` instead (Unsloth method). |
| DPO attention mask crash | Already handled by `_patch_model_for_dpo()` in `dpo_trainer.py`. If still failing, ensure `PatchDPOTrainer()` is called before training. |
| `xformers not found` warning | Safe to ignore on A5000. |
| SSH disconnects mid-training | Training continues in background (started with `nohup`). Reconnect and check log. |
| Pod restarted, models gone | Models on network volume (`/workspace/`) persist. Container storage (`/root/`, `/tmp/`) does not. |

---

## Important Rules

1. **STOP the pod after training completes.** Idle GPU pods burn money ($0.79/hr).
2. **Network volume** (`/workspace/`) persists across pod restarts. Container storage does not.
3. **Never commit model weights to git.** `.gitignore` covers `*.safetensors`, `*.bin`, `*.gguf`. Push datasets (JSONL) and configs (YAML) only.
4. **Monitor VRAM:** `watch -n1 nvidia-smi` — should stay under 24GB.
5. **Install order matters:** PyTorch 2.6+ first, then Unsloth, then `--no-deps trl peft accelerate bitsandbytes`.
6. **Never skip RAFT.** Models trained SFT → DPO (skipping RAFT) hallucinate from retrieved context.
