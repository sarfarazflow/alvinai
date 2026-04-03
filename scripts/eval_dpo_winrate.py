#!/usr/bin/env python3
"""Evaluate DPO win rate: DPO model vs RAFT-only model.

Loads both checkpoints, generates responses for the same prompts,
and uses a template-based judge to determine which response is better.

Target: DPO win rate >= 15% over RAFT-only.

Usage:
    python -m scripts.eval_dpo_winrate \
        --raft-checkpoint experiments/mistral-nemo-12b-v1/models/raft_checkpoint \
        --dpo-checkpoint experiments/mistral-nemo-12b-v1/models/dpo_checkpoint \
        --eval-data data/v1/eval/engineering_eval.jsonl \
        --num-samples 50
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
    return model, tokenizer


def generate_response(model, tokenizer, question, system_prompt=None, max_new_tokens=512):
    """Generate a response from the model."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def template_judge(question, ground_truth, response_a, response_b):
    """Template-based judge: score responses on factual overlap, citation, length, and abstention.

    Returns: "A", "B", or "tie" with a reason.
    """
    score_a = 0
    score_b = 0
    reasons = []

    gt_lower = ground_truth.lower()
    a_lower = response_a.lower()
    b_lower = response_b.lower()

    # 1. Factual overlap with ground truth (keyword matching)
    gt_words = set(w for w in gt_lower.split() if len(w) > 4)
    if gt_words:
        overlap_a = len(gt_words & set(a_lower.split())) / len(gt_words)
        overlap_b = len(gt_words & set(b_lower.split())) / len(gt_words)
        if overlap_a > overlap_b + 0.1:
            score_a += 2
            reasons.append("A has better factual overlap")
        elif overlap_b > overlap_a + 0.1:
            score_b += 2
            reasons.append("B has better factual overlap")

    # 2. Citation quality (##begin_quote## markers)
    has_citation_a = "##begin_quote##" in response_a or "##end_quote##" in response_a
    has_citation_b = "##begin_quote##" in response_b or "##end_quote##" in response_b
    if has_citation_a and not has_citation_b:
        score_a += 1
        reasons.append("A cites sources")
    elif has_citation_b and not has_citation_a:
        score_b += 1
        reasons.append("B cites sources")

    # 3. Appropriate length (penalize very short or very long)
    len_a = len(response_a)
    len_b = len(response_b)
    if len_a < 20 and len_b >= 20:
        score_b += 1
        reasons.append("A is too short")
    elif len_b < 20 and len_a >= 20:
        score_a += 1
        reasons.append("B is too short")
    elif len_a > 1500 and len_b < 1500:
        score_b += 1
        reasons.append("A is too verbose")
    elif len_b > 1500 and len_a < 1500:
        score_a += 1
        reasons.append("B is too verbose")

    # 4. Coherence (no repeated sentences)
    def has_repetition(text):
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        return len(sentences) != len(set(sentences))

    if has_repetition(response_a) and not has_repetition(response_b):
        score_b += 1
        reasons.append("A has repetition")
    elif has_repetition(response_b) and not has_repetition(response_a):
        score_a += 1
        reasons.append("B has repetition")

    # 5. Non-empty and meaningful
    if len(response_a.strip()) < 5:
        score_b += 2
        reasons.append("A is empty/garbage")
    if len(response_b.strip()) < 5:
        score_a += 2
        reasons.append("B is empty/garbage")

    if score_a > score_b:
        return "A", "; ".join(reasons) if reasons else "A scored higher overall"
    elif score_b > score_a:
        return "B", "; ".join(reasons) if reasons else "B scored higher overall"
    else:
        return "tie", "scores equal"


def main():
    parser = argparse.ArgumentParser(description="DPO win rate evaluation")
    parser.add_argument("--raft-checkpoint", required=True)
    parser.add_argument("--dpo-checkpoint", required=True)
    parser.add_argument("--eval-data", required=True, nargs="+",
                        help="One or more eval JSONL files")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval_dpo_winrate.json")
    args = parser.parse_args()

    # Load eval data from all files
    all_examples = []
    for eval_file in args.eval_data:
        with open(eval_file) as f:
            for line in f:
                ex = json.loads(line)
                all_examples.append(ex)

    rng = random.Random(args.seed)
    if len(all_examples) > args.num_samples:
        all_examples = rng.sample(all_examples, args.num_samples)
    print(f"Evaluating {len(all_examples)} examples\n")

    # Load RAFT model
    print("=" * 60)
    raft_model, raft_tokenizer = load_model(args.raft_checkpoint)

    # Generate RAFT responses
    print(f"\nGenerating RAFT responses...")
    raft_responses = []
    for i, ex in enumerate(all_examples):
        resp = generate_response(raft_model, raft_tokenizer, ex["question"])
        raft_responses.append(resp)
        if (i + 1) % 10 == 0:
            print(f"  RAFT: {i+1}/{len(all_examples)}")

    # Free RAFT model memory
    import torch
    del raft_model
    torch.cuda.empty_cache()
    print("RAFT model unloaded.\n")

    # Load DPO model
    dpo_model, dpo_tokenizer = load_model(args.dpo_checkpoint)

    # Generate DPO responses
    print(f"\nGenerating DPO responses...")
    dpo_responses = []
    for i, ex in enumerate(all_examples):
        resp = generate_response(dpo_model, dpo_tokenizer, ex["question"])
        dpo_responses.append(resp)
        if (i + 1) % 10 == 0:
            print(f"  DPO: {i+1}/{len(all_examples)}")

    del dpo_model
    torch.cuda.empty_cache()
    print("DPO model unloaded.\n")

    # Judge: compare responses (randomize order to remove position bias)
    print("Judging responses...")
    wins_dpo = 0
    wins_raft = 0
    ties = 0
    results = []

    for i, ex in enumerate(all_examples):
        ground_truth = ex.get("ground_truth", "")
        raft_resp = raft_responses[i]
        dpo_resp = dpo_responses[i]

        # Randomize position
        if rng.random() < 0.5:
            winner, reason = template_judge(ex["question"], ground_truth, dpo_resp, raft_resp)
            if winner == "A":
                winner = "dpo"
            elif winner == "B":
                winner = "raft"
        else:
            winner, reason = template_judge(ex["question"], ground_truth, raft_resp, dpo_resp)
            if winner == "A":
                winner = "raft"
            elif winner == "B":
                winner = "dpo"

        if winner == "dpo":
            wins_dpo += 1
        elif winner == "raft":
            wins_raft += 1
        else:
            ties += 1

        results.append({
            "id": ex.get("id", f"q{i}"),
            "namespace": ex.get("namespace", "unknown"),
            "question": ex["question"],
            "ground_truth": ground_truth[:200],
            "raft_response": raft_resp[:300],
            "dpo_response": dpo_resp[:300],
            "winner": winner,
            "reason": reason,
        })

    # Summary
    total = len(all_examples)
    dpo_win_rate = wins_dpo / total * 100
    raft_win_rate = wins_raft / total * 100
    tie_rate = ties / total * 100
    net_win_rate = (wins_dpo - wins_raft) / total * 100

    print("\n" + "=" * 60)
    print("DPO Win Rate Evaluation")
    print("=" * 60)
    print(f"  Total examples:    {total}")
    print(f"  DPO wins:          {wins_dpo} ({dpo_win_rate:.1f}%)")
    print(f"  RAFT wins:         {wins_raft} ({raft_win_rate:.1f}%)")
    print(f"  Ties:              {ties} ({tie_rate:.1f}%)")
    print(f"  Net DPO win rate:  {net_win_rate:+.1f}%  (target: >= +15%)")
    print(f"  Verdict:           {'PASS' if net_win_rate >= 15 else 'BELOW TARGET'}")

    # Namespace breakdown
    from collections import Counter
    ns_wins = {}
    for r in results:
        ns = r["namespace"]
        if ns not in ns_wins:
            ns_wins[ns] = {"dpo": 0, "raft": 0, "tie": 0}
        ns_wins[ns][r["winner"]] += 1

    print(f"\n  By namespace:")
    for ns, counts in sorted(ns_wins.items()):
        total_ns = sum(counts.values())
        print(f"    {ns:<20} DPO:{counts['dpo']} RAFT:{counts['raft']} Tie:{counts['tie']} "
              f"(net: {(counts['dpo']-counts['raft'])/total_ns*100:+.0f}%)")

    summary = {
        "raft_checkpoint": args.raft_checkpoint,
        "dpo_checkpoint": args.dpo_checkpoint,
        "num_samples": total,
        "dpo_wins": wins_dpo,
        "raft_wins": wins_raft,
        "ties": ties,
        "dpo_win_rate_pct": round(dpo_win_rate, 1),
        "raft_win_rate_pct": round(raft_win_rate, 1),
        "net_dpo_win_rate_pct": round(net_win_rate, 1),
        "target_met": net_win_rate >= 15,
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
