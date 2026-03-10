import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm_client import generate
from app.ai.prompt import get_system_prompt, format_context
from app.ai.query_classifier import classify_query_type, extract_document_ref, is_greeting
from app.ai.retriever import retrieve
from app.ai.reranker import rerank
from app.ai.structured import structured_lookup
from app.core.config import get_settings
from app.core.metrics import QueryMetrics

logger = logging.getLogger("alvinai")
settings = get_settings()


async def run_query(
    query: str,
    namespace: str = "customer_support",
    db: AsyncSession | None = None,
) -> dict:
    """Run query through the full RAG pipeline.

    Flow:
    1. Classify query type (factual_lookup | document_search | general)
    2. Route accordingly:
       - factual_lookup → direct DB lookup, no LLM
       - document_search → retrieve → rerank → context inject → generate
       - general → LLM only, no retrieval
    """
    metrics = QueryMetrics()

    # --- greeting: instant response, no LLM ---
    if is_greeting(query):
        logger.info("Greeting detected, returning canned response")
        return {
            "answer": "Hello! I'm Alvin, your AI assistant. How can I help you today?",
            "namespace": namespace,
            "sources": [],
            "latency_ms": metrics.elapsed_ms(),
            "query_type": "greeting",
        }

    query_type = classify_query_type(query)
    doc_ref = extract_document_ref(query)
    logger.info("Query classified as '%s' for namespace '%s' (doc_ref=%s)", query_type, namespace, doc_ref)

    # --- factual_lookup: direct DB search, no LLM ---
    if query_type == "factual_lookup" and db is not None:
        result = await structured_lookup(db, query, namespace)
        result["latency_ms"] = metrics.elapsed_ms()
        metrics.log(query, namespace)
        return result

    # --- document_search: full RAG pipeline ---
    if query_type == "document_search" and db is not None:
        # 1. Retrieve (filter by document if referenced in query)
        chunks = await retrieve(db, query, namespace, top_k=settings.RAG_TOP_K, doc_title_filter=doc_ref)

        # 2. Rerank
        if chunks:
            chunks = rerank(query, chunks, top_k=5)

        # 3. Build context + generate
        system_prompt = get_system_prompt(namespace)
        context = format_context(chunks) if chunks else ""

        answer = await generate(
            query=query,
            system_prompt=system_prompt,
            context=context,
            max_tokens=512,
            temperature=0.3,
        )

        # 4. Build sources list
        sources = [
            {"title": c.document_title, "snippet": c.content[:200]}
            for c in chunks
        ]

        metrics.log(query, namespace)
        return {
            "answer": answer,
            "namespace": namespace,
            "sources": sources,
            "latency_ms": metrics.elapsed_ms(),
            "query_type": "document_search",
        }

    # --- general: LLM only, no retrieval ---
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
        "query_type": "general",
    }
