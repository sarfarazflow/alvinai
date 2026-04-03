#!/usr/bin/env python3
"""Evaluate a RAFT checkpoint against validation data.

Tests the 3 RAFT validation metrics:
  1. Oracle citation rate: does the model cite the oracle doc?
  2. Distractor rejection rate: does it avoid citing distractors?
  3. Abstention rate: does it say "not found" for oracle-free examples?

Usage:
    python scripts/eval_raft_checkpoint.py \
        --checkpoint experiments/mistral-nemo-12b-v1/models/raft_checkpoint/checkpoint-200 \
        --val-data data/v2/raft/val.jsonl \
        --num-samples 100

    # Also works for SFT checkpoints (skip RAFT-specific metrics):
    python scripts/eval_raft_checkpoint.py \
        --checkpoint experiments/mistral-nemo-12b-v1/models/sft_checkpoint/checkpoint-200 \
        --val-data data/v2/sft/val.jsonl \
        --num-samples 50 \
        --stage sft
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def load_model(checkpoint_path, max_seq_length=4096):
    """Load model from checkpoint using Unsloth."""
    from unsloth import FastLanguageModel

    print(f"Loading model from {checkpoint_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.")
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response from the model given a list of messages."""
    # Build prompt from messages (system + user only, no assistant)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    # Use tokenizer's chat template
    input_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def evaluate_raft_example(response, metadata, oracle_answer):
    """Evaluate a single RAFT example against the 3 metrics."""
    result = {
        "type": metadata.get("type", "unknown"),
        "category": metadata.get("category", "unknown"),
    }

    response_lower = response.lower()

    if metadata.get("type") == "oracle":
        # Check oracle citation: does the response use ##begin_quote## or reference the oracle doc?
        oracle_doc = metadata.get("oracle_doc", "")
        has_citation = "##begin_quote##" in response or "##end_quote##" in response
        has_doc_ref = oracle_doc.lower() in response_lower if oracle_doc else False
        result["oracle_cited"] = has_citation or has_doc_ref

        # Check distractor rejection: response should NOT contain "not found" / "not available"
        abstention_phrases = [
            "not available in the provided",
            "not found in the provided",
            "cannot find",
            "not present in the",
            "do not have information",
            "not covered in the documents",
        ]
        incorrectly_abstained = any(p in response_lower for p in abstention_phrases)
        result["distractor_rejected"] = not incorrectly_abstained

    elif metadata.get("type") == "oracle_free":
        # Check abstention: model should say "not found" / "not available"
        abstention_phrases = [
            "not available",
            "not found",
            "cannot find",
            "not present",
            "do not have",
            "not covered",
            "not contain",
            "cannot confirm",
            "cannot locate",
            "not in the provided",
        ]
        correctly_abstained = any(p in response_lower for p in abstention_phrases)
        result["correctly_abstained"] = correctly_abstained

    # General quality: is the response non-empty and reasonable length?
    result["response_length"] = len(response)
    result["non_empty"] = len(response) > 10

    return result


def evaluate_sft_example(response, metadata, expected_answer):
    """Evaluate a single SFT example — basic quality checks."""
    return {
        "category": metadata.get("category", "unknown"),
        "response_length": len(response),
        "non_empty": len(response) > 10,
        "has_content": len(response) > 30,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAFT/SFT checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--val-data", required=True, help="Path to validation JSONL")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--stage", default="raft", choices=["sft", "raft", "dpo"], help="Training stage")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    args = parser.parse_args()

    # Load validation data
    print(f"Loading validation data from {args.val_data}...")
    with open(args.val_data) as f:
        val_data = [json.loads(line) for line in f]

    # Sample
    rng = random.Random(args.seed)
    if len(val_data) > args.num_samples:
        val_data = rng.sample(val_data, args.num_samples)
    print(f"Evaluating {len(val_data)} samples...")

    # Load model
    model, tokenizer = load_model(args.checkpoint, args.max_seq_length)

    # Run evaluation
    results = []
    start_time = time.time()

    for i, example in enumerate(val_data):
        messages = example.get("messages", [])
        metadata = example.get("metadata", {})
        expected = messages[-1]["content"] if messages else ""

        response = generate_response(model, tokenizer, messages)

        if args.stage == "raft":
            result = evaluate_raft_example(response, metadata, expected)
        else:
            result = evaluate_sft_example(response, metadata, expected)

        result["response_preview"] = response[:200]
        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(val_data)}] {rate:.1f} examples/s")

    elapsed = time.time() - start_time

    # Compute metrics
    print(f"\nCompleted in {elapsed:.0f}s ({len(results)/elapsed:.1f} examples/s)")
    print("=" * 60)

    if args.stage == "raft":
        # RAFT metrics
        oracle_examples = [r for r in results if r["type"] == "oracle"]
        oracle_free_examples = [r for r in results if r["type"] == "oracle_free"]

        oracle_cited = sum(1 for r in oracle_examples if r.get("oracle_cited", False))
        distractor_rejected = sum(1 for r in oracle_examples if r.get("distractor_rejected", False))
        correctly_abstained = sum(1 for r in oracle_free_examples if r.get("correctly_abstained", False))

        oracle_citation_rate = oracle_cited / max(len(oracle_examples), 1)
        distractor_rejection_rate = distractor_rejected / max(len(oracle_examples), 1)
        abstention_rate = correctly_abstained / max(len(oracle_free_examples), 1)

        print(f"RAFT Validation Results ({args.checkpoint})")
        print("-" * 60)
        print(f"  Oracle examples:          {len(oracle_examples)}")
        print(f"  Oracle-free examples:     {len(oracle_free_examples)}")
        print()
        print(f"  Oracle citation rate:     {oracle_citation_rate:.3f}  (target >= 0.80)  {'PASS' if oracle_citation_rate >= 0.80 else 'FAIL'}")
        print(f"  Distractor rejection:     {distractor_rejection_rate:.3f}  (target >= 0.80)  {'PASS' if distractor_rejection_rate >= 0.80 else 'FAIL'}")
        print(f"  Abstention rate:          {abstention_rate:.3f}  (target >= 0.80)  {'PASS' if abstention_rate >= 0.80 else 'FAIL'}")
        print()

        all_pass = all([
            oracle_citation_rate >= 0.80,
            distractor_rejection_rate >= 0.80,
            abstention_rate >= 0.80,
        ])
        print(f"  Overall: {'ALL PASS — proceed to DPO' if all_pass else 'FAIL — fix RAFT data before DPO'}")

        summary = {
            "stage": "raft",
            "checkpoint": args.checkpoint,
            "num_samples": len(results),
            "oracle_citation_rate": round(oracle_citation_rate, 4),
            "distractor_rejection_rate": round(distractor_rejection_rate, 4),
            "abstention_rate": round(abstention_rate, 4),
            "all_pass": all_pass,
            "elapsed_seconds": round(elapsed, 1),
        }

    else:
        # SFT/DPO basic metrics
        non_empty = sum(1 for r in results if r.get("non_empty", False))
        has_content = sum(1 for r in results if r.get("has_content", False))
        avg_length = sum(r.get("response_length", 0) for r in results) / max(len(results), 1)

        print(f"SFT Validation Results ({args.checkpoint})")
        print("-" * 60)
        print(f"  Samples evaluated:   {len(results)}")
        print(f"  Non-empty responses: {non_empty}/{len(results)} ({non_empty/len(results)*100:.0f}%)")
        print(f"  Has content (>30c):  {has_content}/{len(results)} ({has_content/len(results)*100:.0f}%)")
        print(f"  Avg response length: {avg_length:.0f} chars")

        # Category breakdown
        from collections import Counter
        cats = Counter(r["category"] for r in results)
        print(f"\n  By namespace:")
        for cat, count in sorted(cats.items()):
            cat_results = [r for r in results if r["category"] == cat]
            cat_non_empty = sum(1 for r in cat_results if r.get("non_empty", False))
            cat_avg_len = sum(r.get("response_length", 0) for r in cat_results) / max(len(cat_results), 1)
            print(f"    {cat:<20} {cat_non_empty}/{count} non-empty, avg {cat_avg_len:.0f} chars")

        summary = {
            "stage": args.stage,
            "checkpoint": args.checkpoint,
            "num_samples": len(results),
            "non_empty_rate": round(non_empty / len(results), 4),
            "avg_response_length": round(avg_length, 1),
            "elapsed_seconds": round(elapsed, 1),
        }

    # Save results
    output_path = args.output or f"eval_{args.stage}_results.json"
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
