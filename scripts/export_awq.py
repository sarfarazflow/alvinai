#!/usr/bin/env python3
"""Export a fine-tuned model to AWQ 4-bit quantization for vLLM serving.

Usage:
    python -m scripts.export_awq \
        --input experiments/mistral-nemo-12b-v1/models/dpo_checkpoint/ \
        --output experiments/mistral-nemo-12b-v1/models/production/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def main():
    parser = argparse.ArgumentParser(description="AlvinAI AWQ Export")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the merged model checkpoint (DPO output)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the AWQ quantized model",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="AWQ group size (default: 128)",
    )
    args = parser.parse_args()

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading model from {args.input}...")
    model = AutoAWQForCausalLM.from_pretrained(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.input)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": args.bits,
        "version": "GEMM",
    }

    print(f"Quantizing to AWQ {args.bits}-bit (group_size={args.group_size})...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving quantized model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_quantized(args.output)
    tokenizer.save_pretrained(args.output)

    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith((".safetensors", ".bin"))
    )
    print(f"Done. Model size: {total_size / 1e9:.1f} GB")
    print(f"Serve with: vllm serve {args.output} --quantization awq")


if __name__ == "__main__":
    main()
