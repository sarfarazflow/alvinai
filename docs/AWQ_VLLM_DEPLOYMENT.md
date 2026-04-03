# AWQ Quantization & vLLM Deployment — Session Prompt

## Context

The Mistral Nemo 12B model has completed the full fine-tuning pipeline (SFT → RAFT → DPO).
The DPO merged 16-bit model is on HuggingFace: `sarfarazflow/alvinai-nemo-12b-dpo-20260403` (23GB, 5 shards).
All LoRA adapters and eval results are also backed up on HuggingFace.

This session covers: AWQ 4-bit quantization → upload to HF → deploy on vLLM → validate.

## Step 1: AWQ Quantization (on RTX PRO 6000 training pod)

```bash
# 1. Start RunPod pod: RTX PRO 6000 (96GB), 50GB volume
# 2. SSH in

# 3. Setup
cd /workspace
git clone https://github.com/sarfarazflow/alvinai.git
cd alvinai
pip install uv && cd backend && uv sync --extra training && cd ..

# 4. Install AutoAWQ
pip install autoawq

# 5. Download the DPO merged model from HuggingFace
export HF_TOKEN=<your_hf_token>
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'sarfarazflow/alvinai-nemo-12b-dpo-20260403',
    local_dir='models/dpo_merged',
    token='$HF_TOKEN'
)
print('Download complete')
"

# 6. Run AWQ quantization
python -m scripts.export_awq \
    --input models/dpo_merged \
    --output models/production/alvinai-nemo-12b-awq

# If export_awq.py needs updating, the core logic is:
#
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer
#
# model = AutoAWQForCausalLM.from_pretrained(input_path)
# tokenizer = AutoTokenizer.from_pretrained(input_path)
#
# quant_config = {
#     "zero_point": True,
#     "q_group_size": 128,
#     "w_bit": 4,
#     "version": "GEMM"
# }
#
# # Calibration dataset (use a sample of training data)
# model.quantize(tokenizer, quant_config=quant_config)
# model.save_quantized(output_path)
# tokenizer.save_pretrained(output_path)

# 7. Verify AWQ model size (should be ~6GB vs 23GB original)
du -sh models/production/alvinai-nemo-12b-awq/

# 8. Upload AWQ model to HuggingFace
python -m scripts.upload_model_to_hf \
    --local-path models/production/alvinai-nemo-12b-awq \
    --repo-id sarfarazflow/alvinai-nemo-12b-awq-20260403 \
    --commit-message "Nemo 12B AWQ 4-bit production model (SFT→RAFT→DPO)"

# 9. STOP the training pod
```

## Step 2: vLLM Inference Setup (on A5000 inference pod)

```bash
# 1. Start RunPod pod: A5000 (24GB), 20GB volume
#    Template: RunPod vLLM (or plain PyTorch)
#    Expose ports: 8080 (vLLM API)

# 2. SSH in

# 3. Install vLLM
pip install vllm

# 4. Start vLLM server with AWQ model
export HF_TOKEN=<your_hf_token>

python -m vllm.entrypoints.openai.api_server \
    --model sarfarazflow/alvinai-nemo-12b-awq-20260403 \
    --quantization awq \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8080 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --served-model-name alvinai-nemo-12b \
    --trust-remote-code

# vLLM will:
# - Download the AWQ model from HuggingFace (~6GB)
# - Load it into GPU memory
# - Serve OpenAI-compatible API on port 8080
```

## Step 3: Validate the Deployment

```bash
# From another terminal (or local machine), test the endpoint:

# Health check
curl http://<pod_ip>:8080/health

# List models
curl http://<pod_ip>:8080/v1/models

# Test chat completion
curl http://<pod_ip>:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "alvinai-nemo-12b",
        "messages": [
            {"role": "system", "content": "You are a helpful automotive assistant."},
            {"role": "user", "content": "What is a TSB in automotive engineering?"}
        ],
        "max_tokens": 256,
        "temperature": 0.3
    }'

# Run the validation script against the endpoint
python -m scripts.eval_model \
    --endpoint-url http://<pod_ip>:8080 \
    --api-key dummy \
    --direct

# Latency benchmark (run 10 queries, measure P95)
python3 -c "
import requests, time
url = 'http://<pod_ip>:8080/v1/chat/completions'
latencies = []
for i in range(10):
    start = time.time()
    r = requests.post(url, json={
        'model': 'alvinai-nemo-12b',
        'messages': [{'role': 'user', 'content': 'What warranty coverage does my vehicle have?'}],
        'max_tokens': 256,
        'temperature': 0.3,
    })
    latencies.append(time.time() - start)
    print(f'  Query {i+1}: {latencies[-1]:.2f}s')
latencies.sort()
print(f'\nP50: {latencies[4]:.2f}s  P95: {latencies[8]:.2f}s')
"
```

## Step 4: Production Configuration

For production deployment (Hetzner or cloud), update `.env`:

```bash
VLLM_BASE_URL=http://vllm_blue:8080/v1
VLLM_MODEL_ID=sarfarazflow/alvinai-nemo-12b-awq-20260403
VLLM_MODEL_NAME=alvinai-nemo-12b
```

Docker Compose vLLM service:

```yaml
vllm_blue:
    image: vllm/vllm-openai:latest
    command: >
        --model sarfarazflow/alvinai-nemo-12b-awq-20260403
        --quantization awq
        --dtype float16
        --host 0.0.0.0
        --port 8080
        --max-model-len 4096
        --gpu-memory-utilization 0.85
        --served-model-name alvinai-nemo-12b
    ports:
        - "8080:8080"
    deploy:
        resources:
            reservations:
                devices:
                    - capabilities: [gpu]
    environment:
        - HF_TOKEN=${HF_TOKEN}
```

## Expected Results

| Metric | Target |
|---|---|
| AWQ model size | ~6GB (vs 23GB fp16) |
| VRAM usage | ~8-10GB (model + KV cache) |
| Cold start (model load) | < 30s on A5000 |
| P95 latency (256 tokens) | < 8s |
| P95 latency (warm cache) | < 200ms |

## Troubleshooting

- **`--quantization awq` required** — do NOT use `gptq` flag for AWQ models
- **`--gpu-memory-utilization 0.85`** — don't go above 0.90, need headroom for KV cache
- **OOM on A5000** — reduce `--max-model-len` to 2048
- **Slow first request** — normal, vLLM builds CUDA graphs on first inference
- **Token errors** — ensure HF_TOKEN is set if repo is private
