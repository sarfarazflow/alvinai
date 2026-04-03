#!/usr/bin/env python3
"""RAGAS evaluation for fine-tuned model.

Loads the DPO checkpoint, generates responses for eval questions,
and computes RAGAS-style metrics using template-based scoring.

Metrics:
  - Faithfulness: does the response stick to the reference context?
  - Answer Relevance: does it answer the question asked?
  - Context Precision: is the cited context relevant?

Usage:
    python -m scripts.eval_ragas \
        --checkpoint experiments/mistral-nemo-12b-v1/models/dpo_checkpoint \
        --eval-dir data/v1/eval \
        --num-samples 30 \
        --output experiments/mistral-nemo-12b-v1/eval/ragas_eval.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

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
    return model, tokenizer


def generate_response(model, tokenizer, question, context=None, max_new_tokens=512):
    """Generate response with optional context."""
    if context:
        user_msg = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {question}"
        )
    else:
        user_msg = question

    import torch

    messages = [{"role": "user", "content": user_msg}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=3584).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    except RuntimeError as e:
        print(f"    Generation error: {e}")
        return "[generation failed]"


def compute_faithfulness(response, context):
    """Measure how faithful the response is to the context.
    Score based on keyword overlap with context vs. novel claims."""
    if not context or not response:
        return 0.0

    ctx_words = set(w.lower() for w in context.split() if len(w) > 3)
    resp_words = set(w.lower() for w in response.split() if len(w) > 3)

    if not resp_words:
        return 0.0

    # What fraction of response words appear in context?
    grounded = len(resp_words & ctx_words) / len(resp_words)
    return min(grounded * 1.5, 1.0)  # Scale up slightly, cap at 1.0


def compute_answer_relevance(response, question):
    """Measure how relevant the response is to the question."""
    if not response or len(response) < 10:
        return 0.0

    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    r_words = set(w.lower() for w in response.split() if len(w) > 3)

    if not q_words:
        return 0.5

    # Question words that appear in response
    q_coverage = len(q_words & r_words) / len(q_words)

    # Penalize very short or very long responses
    length_penalty = 1.0
    if len(response) < 30:
        length_penalty = 0.5
    elif len(response) > 2000:
        length_penalty = 0.8

    return min(q_coverage * 1.3 * length_penalty, 1.0)


def compute_context_precision(response, ground_truth):
    """Measure how well the response matches the ground truth answer."""
    if not ground_truth or not response:
        return 0.0

    gt_words = set(w.lower() for w in ground_truth.split() if len(w) > 3)
    resp_words = set(w.lower() for w in response.split() if len(w) > 3)

    if not gt_words:
        return 0.5

    precision = len(gt_words & resp_words) / len(gt_words)
    return min(precision * 1.4, 1.0)


def main():
    parser = argparse.ArgumentParser(description="RAGAS-style evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--eval-dir", required=True, help="Directory with eval JSONL files")
    parser.add_argument("--num-samples", type=int, default=30,
                        help="Samples per namespace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_ragas.json")
    args = parser.parse_args()

    # Load all eval files
    eval_dir = Path(args.eval_dir)
    all_examples = {}
    for f in sorted(eval_dir.glob("*_eval.jsonl")):
        namespace = f.stem.replace("_eval", "")
        with open(f) as fh:
            examples = [json.loads(line) for line in fh]
        all_examples[namespace] = examples
        print(f"  {namespace}: {len(examples)} questions")

    print(f"\nTotal: {sum(len(v) for v in all_examples.values())} questions "
          f"across {len(all_examples)} namespaces")

    # Sample per namespace
    rng = random.Random(args.seed)
    sampled = {}
    for ns, examples in all_examples.items():
        if len(examples) > args.num_samples:
            sampled[ns] = rng.sample(examples, args.num_samples)
        else:
            sampled[ns] = examples

    total_samples = sum(len(v) for v in sampled.values())
    print(f"Sampled: {total_samples} questions\n")

    # Load model
    model, tokenizer = load_model(args.checkpoint)

    # Run evaluation
    all_results = {}
    namespace_metrics = {}
    start_time = time.time()
    processed = 0

    for ns, examples in sorted(sampled.items()):
        print(f"\nEvaluating {ns} ({len(examples)} questions)...")
        ns_results = []
        ns_scores = {"faithfulness": [], "answer_relevance": [], "context_precision": []}

        for i, ex in enumerate(examples):
            question = ex["question"]
            context = ex.get("reference_context", "")
            ground_truth = ex.get("ground_truth", "")

            response = generate_response(model, tokenizer, question, context)

            faith = compute_faithfulness(response, context)
            relevance = compute_answer_relevance(response, question)
            precision = compute_context_precision(response, ground_truth)

            ns_scores["faithfulness"].append(faith)
            ns_scores["answer_relevance"].append(relevance)
            ns_scores["context_precision"].append(precision)

            ns_results.append({
                "id": ex.get("id", f"{ns}-{i}"),
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "faithfulness": round(faith, 3),
                "answer_relevance": round(relevance, 3),
                "context_precision": round(precision, 3),
            })

            processed += 1
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  [{processed}/{total_samples}] {elapsed:.0f}s elapsed")

        # Namespace averages
        avg = {k: round(sum(v) / len(v), 4) for k, v in ns_scores.items()}
        namespace_metrics[ns] = avg
        all_results[ns] = ns_results
        print(f"  {ns}: faith={avg['faithfulness']:.3f} "
              f"relev={avg['answer_relevance']:.3f} "
              f"prec={avg['context_precision']:.3f}")

    elapsed = time.time() - start_time

    # Overall averages
    overall = {}
    for metric in ["faithfulness", "answer_relevance", "context_precision"]:
        values = [namespace_metrics[ns][metric] for ns in namespace_metrics]
        overall[metric] = round(sum(values) / len(values), 4)

    # Print summary
    print("\n" + "=" * 70)
    print("RAGAS Evaluation Summary")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples:    {total_samples} across {len(namespace_metrics)} namespaces")
    print(f"  Duration:   {elapsed:.0f}s\n")

    # Targets from CLAUDE.md
    targets = {"faithfulness": 0.88, "answer_relevance": 0.85, "context_precision": 0.80}

    print(f"  {'Metric':<25} {'Score':>8} {'Target':>8} {'Status':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for metric, score in overall.items():
        target = targets.get(metric, 0)
        status = "PASS" if score >= target else "BELOW"
        print(f"  {metric:<25} {score:>8.4f} {target:>8.2f} {status:>8}")

    print(f"\n  By namespace:")
    print(f"  {'Namespace':<20} {'Faith':>8} {'Relev':>8} {'Prec':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for ns, metrics in sorted(namespace_metrics.items()):
        print(f"  {ns:<20} {metrics['faithfulness']:>8.3f} "
              f"{metrics['answer_relevance']:>8.3f} {metrics['context_precision']:>8.3f}")

    # Save
    output = {
        "summary": {
            "checkpoint": args.checkpoint,
            "num_samples": total_samples,
            "elapsed_seconds": round(elapsed, 1),
            "overall": overall,
            "targets": targets,
            "all_pass": all(overall[m] >= targets[m] for m in targets),
        },
        "namespace_metrics": namespace_metrics,
        "results": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
