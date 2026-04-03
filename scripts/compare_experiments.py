#!/usr/bin/env python3
"""
compare_experiments.py — Generate COMPARISON.md from experiment results.

Reads all experiments/*/results/*.yaml files and produces a cross-model
comparison table at experiments/COMPARISON.md.

Usage:
    python scripts/compare_experiments.py
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def fmt(val, suffix="", precision=4):
    """Format a value for the table — handle nulls gracefully."""
    if val is None:
        return "—"
    if isinstance(val, float):
        if suffix == "%":
            return f"{val * 100:.{precision}f}%"
        return f"{val:.{precision}f}{suffix}"
    return f"{val}{suffix}"


def fmt_time(minutes):
    if minutes is None:
        return "—"
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    return f"{minutes:.0f} min"


def main():
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("Error: experiments/ directory not found. Run from project root.")
        sys.exit(1)

    # Discover experiments
    experiments = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir() and (d / "experiment.yaml").exists()]
    )

    if not experiments:
        print("No experiments found.")
        sys.exit(0)

    # Load all data
    data = []
    for exp_dir in experiments:
        manifest = load_yaml(exp_dir / "experiment.yaml") or {}
        sft = load_yaml(exp_dir / "results" / "sft_results.yaml") or {}
        raft = load_yaml(exp_dir / "results" / "raft_results.yaml") or {}
        dpo = load_yaml(exp_dir / "results" / "dpo_results.yaml") or {}
        ragas = load_yaml(exp_dir / "results" / "ragas_eval.yaml") or {}
        latency = load_yaml(exp_dir / "results" / "latency_benchmark.yaml") or {}

        sft_metrics = sft.get("metrics", {})
        raft_metrics = raft.get("metrics", {})
        raft_thresh = raft.get("raft_thresholds", {})
        dpo_metrics = dpo.get("metrics", {})
        dpo_eval = dpo.get("dpo_eval", {})
        ragas_overall = ragas.get("overall", {})
        benchmarks = latency.get("benchmarks", {})

        data.append({
            "name": manifest.get("name", exp_dir.name),
            "base_model": manifest.get("base_model", "—"),
            "status": manifest.get("status", "pending"),
            "hardware": manifest.get("hardware", "—"),
            "cost": manifest.get("total_cost_usd"),
            # SFT
            "sft_loss": sft_metrics.get("eval_loss_final"),
            "sft_time": sft.get("duration_minutes"),
            # RAFT
            "raft_loss": raft_metrics.get("eval_loss_final"),
            "raft_time": raft.get("duration_minutes"),
            "oracle_citation": raft_thresh.get("oracle_citation_rate"),
            "distractor_rejection": raft_thresh.get("distractor_rejection_rate"),
            "abstention": raft_thresh.get("abstention_rate"),
            "raft_passed": raft_thresh.get("passed"),
            # DPO
            "dpo_reward_acc": dpo_metrics.get("rewards_accuracy"),
            "dpo_time": dpo.get("duration_minutes"),
            "dpo_win_rate": dpo_eval.get("win_rate"),
            # RAGAS
            "faithfulness": ragas_overall.get("faithfulness_avg"),
            "answer_relevance": ragas_overall.get("answer_relevance_avg"),
            "context_precision": ragas_overall.get("context_precision_avg"),
            "context_recall": ragas_overall.get("context_recall_avg"),
            # Latency
            "factual_p95": benchmarks.get("factual_lookup", {}).get("p95_ms"),
            "doc_warm_p95": benchmarks.get("document_search_warm", {}).get("p95_ms"),
            "doc_cold_p95": benchmarks.get("document_search_cold", {}).get("p95_ms"),
            "tokens_per_sec": latency.get("tokens_per_second"),
        })

    # Generate markdown
    lines = []
    lines.append("# AlvinAI Model Comparison\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                 f"Experiments: {len(data)}\n")

    # ─── Training Summary ───
    lines.append("## Training Summary\n")
    lines.append("| Experiment | Status | SFT Loss | SFT Time | RAFT Loss | RAFT Time | DPO Reward Acc | DPO Time | Cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in data:
        lines.append(
            f"| {d['name']} | {d['status']} | {fmt(d['sft_loss'])} | {fmt_time(d['sft_time'])} "
            f"| {fmt(d['raft_loss'])} | {fmt_time(d['raft_time'])} "
            f"| {fmt(d['dpo_reward_acc'], '%', 1)} | {fmt_time(d['dpo_time'])} "
            f"| {fmt(d['cost'], '$', 2)} |"
        )
    lines.append("")

    # ─── RAFT Quality Gates ───
    lines.append("## RAFT Quality Gates\n")
    lines.append("All thresholds must be >= 0.80 to proceed to DPO.\n")
    lines.append("| Experiment | Oracle Citation | Distractor Rejection | Abstention | Passed |")
    lines.append("|---|---|---|---|---|")
    for d in data:
        passed = d["raft_passed"]
        passed_str = "YES" if passed is True else ("NO" if passed is False else "—")
        lines.append(
            f"| {d['name']} | {fmt(d['oracle_citation'])} | {fmt(d['distractor_rejection'])} "
            f"| {fmt(d['abstention'])} | {passed_str} |"
        )
    lines.append("")

    # ─── RAGAS Evaluation ───
    lines.append("## RAGAS Evaluation\n")
    lines.append("Targets: Faithfulness >= 0.88, Answer Relevance >= 0.85, "
                 "Context Precision >= 0.80, Context Recall >= 0.82\n")
    lines.append("| Experiment | Faithfulness | Answer Relevance | Context Precision | Context Recall |")
    lines.append("|---|---|---|---|---|")
    for d in data:
        lines.append(
            f"| {d['name']} | {fmt(d['faithfulness'])} | {fmt(d['answer_relevance'])} "
            f"| {fmt(d['context_precision'])} | {fmt(d['context_recall'])} |"
        )
    lines.append("")

    # ─── Latency Benchmarks ───
    lines.append("## Latency Benchmarks (AWQ on vLLM)\n")
    lines.append("| Experiment | Factual P95 | Doc Search P95 (warm) | Doc Search P95 (cold) | Tokens/s |")
    lines.append("|---|---|---|---|---|")
    for d in data:
        lines.append(
            f"| {d['name']} | {fmt(d['factual_p95'], 'ms', 0)} | {fmt(d['doc_warm_p95'], 'ms', 0)} "
            f"| {fmt(d['doc_cold_p95'], 'ms', 0)} | {fmt(d['tokens_per_sec'], '', 0)} |"
        )
    lines.append("")

    # ─── Winner Selection ───
    lines.append("## Winner Selection\n")
    lines.append("```")
    lines.append("Tier 1 — 7-8B:   Mistral 7B (baseline)")
    lines.append("Tier 2 — 12B:    Best of {Nemo 12B, Gemma 3 12B}")
    lines.append("Tier 3 — 14B:    Best of {Phi-4 14B, Qwen3 14B}")
    lines.append("Tier 4 — 24B+:   Best of {Mistral Small 24B, Granite 4.0 Small}")
    lines.append("")
    lines.append("Decision: If 14B ≈ 12B (<2% improvement) → stay at 12B (cheaper inference)")
    lines.append("```")
    lines.append("")

    output = "\n".join(lines)
    output_path = experiments_dir / "COMPARISON.md"
    output_path.write_text(output)
    print(f"Written: {output_path} ({len(data)} experiments)")


if __name__ == "__main__":
    main()
