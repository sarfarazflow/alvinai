"""
RAGAS evaluation core library for AlvinAI.

Provides:
- EvalRecord / RagasConfig dataclasses
- run_rag_pipeline()    — AlvinAI RAG pipeline (retrieve → rerank → generate)
- get_baseline_answer() — base Mistral-7B via HF Inference API (no retrieval)
- configure_ragas_llm() — wire Claude, OpenAI, or vLLM as RAGAS judge
- build_ragas_dataset() — build HuggingFace Dataset for ragas.evaluate()
- _package_results()    — structured results dict with scores + per-question rows

Judge options (in order of recommendation):
  1. Claude (claude-sonnet-4-6) via Anthropic API  — set ANTHROPIC_API_KEY
  2. OpenAI (gpt-4o-mini)                          — set OPENAI_API_KEY
  3. Self-hosted vLLM (alvinai-7b-awq)             — fallback, slight self-judge bias
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("alvinai.eval")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    id: str
    question: str
    ground_truth: str
    reference_context: str
    namespace: str
    source_policy: str
    source_section: str
    question_type: str
    topic: str
    # Populated during evaluation
    alvinai_answer: str = ""
    alvinai_contexts: list = field(default_factory=list)
    baseline_answer: str = ""


@dataclass
class RagasConfig:
    vllm_url: str           # external-facing vLLM URL, e.g. http://65.21.x.x:8080/v1
    vllm_model_name: str    # model name registered in vLLM, e.g. alvinai-7b-awq
    hf_token: str           # HuggingFace token for Inference API
    db_url: str             # asyncpg URL, e.g. postgresql+asyncpg://user:pass@host:5432/db
    anthropic_api_key: str = ""   # preferred judge: Claude via Anthropic API
    openai_api_key: str = ""      # fallback judge: OpenAI GPT-4o-mini
    judge_model: str = "claude-sonnet-4-6"  # used when anthropic_api_key is set
    top_k: int = 5


# ---------------------------------------------------------------------------
# AlvinAI RAG pipeline
# ---------------------------------------------------------------------------

async def run_rag_pipeline(
    question: str,
    namespace: str,
    db,
    top_k: int = 5,
) -> tuple[str, list[str]]:
    """Run the full RAG pipeline for one question.

    Deliberately bypasses pipeline.run_query() to avoid the query classifier
    routing HR policy questions to structured_lookup() (which skips retrieval
    and produces empty contexts, invalidating RAGAS context metrics).

    Returns:
        (answer, contexts)  where contexts is a list of chunk content strings.
    """
    from app.ai.retriever import retrieve
    from app.ai.reranker import rerank
    from app.ai.prompt import get_system_prompt, format_context
    from app.ai.llm_client import generate

    chunks = await retrieve(db, question, namespace, top_k=top_k * 2, use_cache=False)
    if chunks:
        chunks = rerank(question, chunks, top_k=top_k)

    system_prompt = get_system_prompt(namespace)
    context = format_context(chunks) if chunks else ""

    answer = await generate(
        query=question,
        system_prompt=system_prompt,
        context=context,
        max_tokens=512,
        temperature=0.3,
    )

    contexts = [c.content for c in chunks]
    return answer, contexts


# ---------------------------------------------------------------------------
# Baseline: base Mistral-7B via HuggingFace Inference API
# ---------------------------------------------------------------------------

def get_baseline_answer(
    question: str,
    hf_token: str,
    namespace: str,
    max_retries: int = 1,
) -> str:
    """Call HF Inference API for base Mistral-7B-Instruct-v0.3.

    Uses the same namespace system prompt as AlvinAI but provides NO context —
    testing what the base model knows from parametric memory alone.

    Retries once on HTTP 429 (rate limit) after a 60-second wait.
    """
    from huggingface_hub import InferenceClient
    from app.ai.prompt import get_system_prompt

    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token,
    )
    system_prompt = get_system_prompt(namespace)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for attempt in range(max_retries + 1):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < max_retries:
                logger.warning("HF rate limit hit, waiting 60s before retry...")
                time.sleep(60)
            else:
                logger.error("HF Inference API failed: %s", e)
                return ""

    return ""


# ---------------------------------------------------------------------------
# RAGAS judge configuration
# ---------------------------------------------------------------------------

def configure_ragas_llm(config: RagasConfig) -> None:
    """Configure the RAGAS LLM judge. Three options, tried in priority order:

    1. Claude (ANTHROPIC_API_KEY set) — uses langchain_anthropic.ChatAnthropic
       with claude-sonnet-4-6. Best results, no OpenAI dependency.

    2. OpenAI (OPENAI_API_KEY set) — RAGAS v0.1 picks it up automatically
       from os.environ; GPT-4o-mini is accurate and cheap (~$0.10 / 50 Qs).

    3. vLLM fallback — points RAGAS at the vLLM OpenAI-compatible endpoint.
       Acceptable for development; slight self-judge bias.
    """
    import os
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # --- Option 1: Claude via Anthropic API ---
    if config.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
        try:
            from langchain_anthropic import ChatAnthropic
            judge_llm = ChatAnthropic(
                model=config.judge_model,
                anthropic_api_key=config.anthropic_api_key,
                temperature=0,
                max_tokens=1024,
            )
            for metric in metrics:
                metric.llm = judge_llm
            logger.info("RAGAS judge: Claude (%s)", config.judge_model)
            return
        except ImportError:
            logger.warning(
                "langchain_anthropic not installed; falling back to OpenAI judge. "
                "Install with: pip install langchain-anthropic"
            )

    # --- Option 2: OpenAI (auto-detected from OPENAI_API_KEY) ---
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        logger.info("RAGAS judge: OpenAI GPT-4o-mini")
        return

    # --- Option 3: vLLM as OpenAI-compatible judge (fallback) ---
    try:
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(
            base_url=config.vllm_url,
            api_key="not-needed",
            model=config.vllm_model_name,
            temperature=0,
            max_tokens=1024,
        )
        for metric in metrics:
            metric.llm = judge_llm
        logger.info(
            "RAGAS judge: vLLM (%s) — self-judge bias present",
            config.vllm_model_name,
        )
    except ImportError:
        logger.error(
            "No judge LLM configured and langchain_openai not installed. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        )
        raise


# ---------------------------------------------------------------------------
# RAGAS dataset builder
# ---------------------------------------------------------------------------

def build_ragas_dataset(records: list, use_alvinai: bool = True):
    """Build a HuggingFace Dataset compatible with ragas.evaluate().

    RAGAS v0.1 expects columns: question, answer, contexts, ground_truth.

    AlvinAI  : alvinai_answer  + alvinai_contexts (live retrieved chunks).
    Baseline : baseline_answer + [reference_context] (gold-standard passage).
               Using reference_context for baseline contexts lets RAGAS compute
               context_precision/recall, giving a meaningful comparison even
               though the baseline model had no live retrieval.
    """
    from datasets import Dataset

    rows = []
    for rec in records:
        if use_alvinai:
            answer = rec.alvinai_answer
            contexts = rec.alvinai_contexts if rec.alvinai_contexts else [rec.reference_context]
        else:
            answer = rec.baseline_answer
            contexts = [rec.reference_context]

        rows.append({
            "question": rec.question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": rec.ground_truth,
        })

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Results packaging
# ---------------------------------------------------------------------------

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RAGAS_TARGETS = {
    "faithfulness": 0.88,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.82,
}


def _package_results(
    records: list,
    alvinai_result,
    baseline_result,        # may be None if --skip-baseline
) -> dict:
    """Convert RAGAS EvaluationResult objects into a structured dict.

    RAGAS v0.1:
      result[metric_name]  → float (aggregate mean)
      result.to_pandas()   → DataFrame with per-question rows
    """
    alvinai_df = alvinai_result.to_pandas()

    alvinai_scores = {}
    for m in METRIC_NAMES:
        try:
            alvinai_scores[m] = round(float(alvinai_result[m]), 4)
        except Exception:
            alvinai_scores[m] = None

    baseline_scores = {}
    baseline_df = None
    if baseline_result is not None:
        baseline_df = baseline_result.to_pandas()
        for m in METRIC_NAMES:
            try:
                baseline_scores[m] = round(float(baseline_result[m]), 4)
            except Exception:
                baseline_scores[m] = None

    delta = {}
    for m in METRIC_NAMES:
        a = alvinai_scores.get(m)
        b = baseline_scores.get(m)
        delta[m] = round(a - b, 4) if (a is not None and b is not None) else None

    per_question = []
    for i, rec in enumerate(records):
        row = {
            "id": rec.id,
            "question": rec.question,
            "question_type": rec.question_type,
            "topic": rec.topic,
            "source_policy": rec.source_policy,
            "source_section": rec.source_section,
            "alvinai_answer": rec.alvinai_answer,
            "baseline_answer": rec.baseline_answer,
            "contexts_retrieved": len(rec.alvinai_contexts),
        }
        for m in METRIC_NAMES:
            try:
                row[f"alvinai_{m}"] = round(float(alvinai_df.iloc[i][m]), 4)
            except Exception:
                row[f"alvinai_{m}"] = None
            if baseline_df is not None:
                try:
                    row[f"baseline_{m}"] = round(float(baseline_df.iloc[i][m]), 4)
                except Exception:
                    row[f"baseline_{m}"] = None
            else:
                row[f"baseline_{m}"] = None
        per_question.append(row)

    return {
        "alvinai": {"scores": alvinai_scores, "per_question": per_question},
        "baseline": {"scores": baseline_scores},
        "delta": delta,
        "targets": RAGAS_TARGETS,
        "pass": {
            m: (alvinai_scores.get(m) or 0.0) >= RAGAS_TARGETS[m]
            for m in METRIC_NAMES
        },
    }
