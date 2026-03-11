"""
AlvinAI RAGAS Evaluation Script
================================
Compares AlvinAI (fine-tuned, Hetzner vLLM) vs base Mistral-7B (HF Inference API)
on the HR eval set using RAGAS metrics.

Usage:
    # Full run — Claude as RAGAS judge (recommended)
    python scripts/run_evaluation.py \\
      --vllm-url http://<hetzner-ip>:8080/v1 \\
      --db-url postgresql+asyncpg://alvinai:<pass>@<hetzner-ip>:5432/alvinai \\
      --hf-token hf_xxx \\
      --anthropic-api-key sk-ant-xxx

    # OpenAI as judge alternative
    python scripts/run_evaluation.py \\
      --vllm-url http://<hetzner-ip>:8080/v1 \\
      --db-url postgresql+asyncpg://alvinai:<pass>@<hetzner-ip>:5432/alvinai \\
      --hf-token hf_xxx \\
      --openai-api-key sk-xxx

    # Smoke test — 5 questions, skip baseline, no judge API needed
    python scripts/run_evaluation.py \\
      --vllm-url http://<hetzner-ip>:8080/v1 \\
      --db-url postgresql+asyncpg://alvinai:<pass>@<hetzner-ip>:5432/alvinai \\
      --limit 5 --skip-baseline

Environment variable equivalents:
    EVAL_VLLM_URL, EVAL_DB_URL, HF_TOKEN,
    ANTHROPIC_API_KEY, OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing — must happen before any app.* imports
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation: AlvinAI vs base Mistral-7B on HR eval set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--eval-file",
        default="data/eval/hr_eval.jsonl",
        help="Path to JSONL eval dataset (default: data/eval/hr_eval.jsonl)",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("EVAL_VLLM_URL", ""),
        help="External vLLM URL (e.g. http://65.21.x.x:8080/v1). Overrides VLLM_BASE_URL.",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("EVAL_DB_URL", ""),
        help="External PostgreSQL URL (postgresql+asyncpg://user:pass@host:port/db)",
    )
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("EVAL_REDIS_URL", "redis://localhost:6379/0"),
        help="Redis URL (default: redis://localhost:6379/0). Not used with use_cache=False.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token for Inference API (base Mistral baseline)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key. If set, uses Claude as RAGAS judge (recommended).",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-6",
        help="Claude model to use as RAGAS judge (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key. Used as RAGAS judge if --anthropic-api-key not set.",
    )
    parser.add_argument(
        "--vllm-model-name",
        default=os.environ.get("VLLM_MODEL_NAME", "alvinai-7b-awq"),
        help="vLLM model name (default: alvinai-7b-awq)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval",
        help="Directory for output files (default: data/eval)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N questions (0 = all). Use 5 for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip HF baseline generation. Useful for testing AlvinAI pipeline only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Eval record loader
# ---------------------------------------------------------------------------

def load_eval_records(path: str, limit: int = 0) -> list:
    """Load EvalRecord objects from JSONL file."""
    # Import deferred: app.* imports only happen after env is patched
    from app.evaluation.ragas_eval import EvalRecord

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(EvalRecord(
                id=d["id"],
                question=d["question"],
                ground_truth=d["ground_truth"],
                reference_context=d["reference_context"],
                namespace=d["namespace"],
                source_policy=d["source_policy"],
                source_section=d["source_section"],
                question_type=d["question_type"],
                topic=d["topic"],
            ))

    if limit:
        records = records[:limit]
    return records


# ---------------------------------------------------------------------------
# Three-phase eval runner with progress
# ---------------------------------------------------------------------------

async def run_eval_with_progress(records: list, config, skip_baseline: bool) -> dict:
    """Drive the evaluation loop with tqdm progress bars.

    Phase 1: AlvinAI RAG pipeline (async, sequential)
    Phase 2: HF Baseline answers (sync, sequential)
    Phase 3: RAGAS evaluate() on both datasets
    """
    from app.evaluation.ragas_eval import (
        run_rag_pipeline,
        get_baseline_answer,
        configure_ragas_llm,
        build_ragas_dataset,
        _package_results,
        METRIC_NAMES,
    )
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )
    from sqlalchemy.ext.asyncio import (
        AsyncSession, create_async_engine, async_sessionmaker,
    )

    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback: simple counter if tqdm not installed
        class tqdm:  # type: ignore
            def __init__(self, iterable, **kwargs):
                self._it = iterable
                desc = kwargs.get("desc", "")
                total = kwargs.get("total", len(iterable) if hasattr(iterable, '__len__') else "?")
                print(f"{desc} ({total} items)...")

            def __iter__(self):
                return iter(self._it)

    configure_ragas_llm(config)

    engine = create_async_engine(config.db_url, echo=False, pool_pre_ping=True)
    SessionFactory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # ── Phase 1: AlvinAI RAG ─────────────────────────────────────────────
    print("\n[1/3] AlvinAI RAG pipeline...")
    empty_context_count = 0

    async with SessionFactory() as db:
        for record in tqdm(records, desc="  AlvinAI RAG", unit="q"):
            try:
                answer, contexts = await run_rag_pipeline(
                    record.question, record.namespace, db, top_k=config.top_k
                )
                record.alvinai_answer = answer
                record.alvinai_contexts = contexts
                if not contexts:
                    empty_context_count += 1
            except Exception as e:
                logging.warning("RAG failed for %s: %s", record.id, e)
                record.alvinai_answer = ""
                record.alvinai_contexts = []
                empty_context_count += 1

    await engine.dispose()

    if empty_context_count:
        print(
            f"\n  WARNING: {empty_context_count}/{len(records)} questions returned 0 "
            "context chunks. Check that HR docs are ingested into the employee_hr namespace."
        )

    # ── Phase 2: Baseline ────────────────────────────────────────────────
    if not skip_baseline:
        print("\n[2/3] Base Mistral-7B baseline (HF Inference API)...")
        for record in tqdm(records, desc="  Baseline", unit="q"):
            try:
                record.baseline_answer = get_baseline_answer(
                    record.question, config.hf_token, record.namespace
                )
            except Exception as e:
                logging.warning("Baseline failed for %s: %s", record.id, e)
                record.baseline_answer = ""
    else:
        print("\n[2/3] Skipping baseline (--skip-baseline)")

    # ── Phase 3: RAGAS scoring ───────────────────────────────────────────
    print("\n[3/3] Running RAGAS evaluation...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    alvinai_dataset = build_ragas_dataset(records, use_alvinai=True)
    print("  Scoring AlvinAI answers...")
    try:
        alvinai_result = evaluate(alvinai_dataset, metrics=metrics)
    except Exception as e:
        logging.error("RAGAS evaluation failed for AlvinAI: %s", e)
        raise

    baseline_result = None
    if not skip_baseline:
        baseline_dataset = build_ragas_dataset(records, use_alvinai=False)
        print("  Scoring baseline answers...")
        try:
            baseline_result = evaluate(baseline_dataset, metrics=metrics)
        except Exception as e:
            logging.warning("RAGAS evaluation failed for baseline: %s — skipping delta", e)

    return _package_results(records, alvinai_result, baseline_result)


# ---------------------------------------------------------------------------
# Output: console table
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    targets = results.get("targets", {})
    passing = results.get("pass", {})
    alvinai = results["alvinai"]["scores"]
    baseline = results.get("baseline", {}).get("scores", {})
    delta = results.get("delta", {})

    width = 76
    print("\n" + "=" * width)
    print("  RAGAS EVALUATION — AlvinAI vs Baseline Mistral-7B (employee_hr)")
    print("=" * width)
    print(f"  {'Metric':<26} {'AlvinAI':>9} {'Baseline':>10} {'Delta':>8} {'Target':>8} {'':>5}")
    print("-" * width)

    for m in METRICS:
        a = alvinai.get(m)
        b = baseline.get(m)
        d = delta.get(m)
        t = targets.get(m, 0)
        passed = passing.get(m, False)

        a_str = f"{a:.3f}" if a is not None else "  N/A "
        b_str = f"{b:.3f}" if b is not None else "  N/A "
        d_str = (f"+{d:.3f}" if d >= 0 else f"{d:.3f}") if d is not None else "  N/A "
        flag = "PASS" if passed else "FAIL"
        flag_fmt = f"\033[92m{flag}\033[0m" if passed else f"\033[91m{flag}\033[0m"

        print(f"  {m:<26} {a_str:>9} {b_str:>10} {d_str:>8} {t:>8.2f}  {flag_fmt}")

    print("=" * width)

    all_pass = all(passing.values())
    if all_pass:
        print("  \033[92mAll metrics PASS — model meets RAGAS targets.\033[0m")
    else:
        failing = [m for m, p in passing.items() if not p]
        print(f"  \033[91mFailing metrics: {', '.join(failing)}\033[0m")
    print()


# ---------------------------------------------------------------------------
# Output: files
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Full JSON
    json_path = out / "ragas_results_hr.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON → {json_path}")

    # Per-question CSV
    csv_path = out / "ragas_results_hr.csv"
    per_q = results["alvinai"]["per_question"]
    if per_q:
        fieldnames = list(per_q[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_q)
        print(f"  CSV  → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Validate required args ───────────────────────────────────────────
    errors = []
    if not args.vllm_url:
        errors.append("--vllm-url (or EVAL_VLLM_URL) is required")
    if not args.db_url:
        errors.append("--db-url (or EVAL_DB_URL) is required")
    if not args.skip_baseline and not args.hf_token:
        errors.append("--hf-token (or HF_TOKEN) is required unless --skip-baseline is set")
    if not args.anthropic_api_key and not args.openai_api_key:
        print(
            "WARNING: No judge API key provided (--anthropic-api-key or --openai-api-key). "
            "Falling back to vLLM as judge — scores may have self-judge bias."
        )
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)

    # ── Patch env BEFORE any app.* imports ──────────────────────────────
    # get_settings() uses @lru_cache; must see patched values on first call.
    os.environ["VLLM_BASE_URL"] = args.vllm_url
    os.environ["REDIS_URL"] = args.redis_url
    if args.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Add backend/ to sys.path so `app.*` imports resolve
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    sys.path.insert(0, str(backend_dir))

    # ── Deferred app.* imports ───────────────────────────────────────────
    from app.evaluation.ragas_eval import RagasConfig

    # ── Load eval data ───────────────────────────────────────────────────
    eval_file = project_root / args.eval_file
    print(f"Loading eval records from: {eval_file}")
    records = load_eval_records(str(eval_file), limit=args.limit)
    n = len(records)
    print(f"Loaded {n} questions")
    if args.limit:
        print(f"(Limited to first {args.limit} questions)")

    # ── Build config ─────────────────────────────────────────────────────
    config = RagasConfig(
        vllm_url=args.vllm_url,
        vllm_model_name=args.vllm_model_name,
        hf_token=args.hf_token,
        anthropic_api_key=args.anthropic_api_key,
        openai_api_key=args.openai_api_key,
        judge_model=args.judge_model,
        db_url=args.db_url,
        top_k=5,
    )

    # ── Run evaluation ───────────────────────────────────────────────────
    results = asyncio.run(
        run_eval_with_progress(records, config, skip_baseline=args.skip_baseline)
    )

    # ── Print results ────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save outputs ─────────────────────────────────────────────────────
    output_dir = project_root / args.output_dir
    print("Saving results...")
    save_results(results, str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
