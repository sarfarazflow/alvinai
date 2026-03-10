import logging
from app.ai.llm_client import generate
from app.ai.prompt import get_system_prompt
from app.core.metrics import QueryMetrics

logger = logging.getLogger("alvinai")


async def run_query(query: str, namespace: str = "customer_support") -> dict:
    """Run query through the AI pipeline.

    Phase 1: Direct LLM call (no RAG).
    Phase 2 will add: retriever -> reranker -> context injection.
    """
    metrics = QueryMetrics()
    system_prompt = get_system_prompt(namespace)

    answer = await generate(
        query=query,
        system_prompt=system_prompt,
        max_tokens=512,
        temperature=0.3,
    )

    metrics.log(query, namespace)

    return {
        "answer": answer,
        "namespace": namespace,
        "sources": [],
        "latency_ms": metrics.elapsed_ms(),
    }
