import logging
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.ingestion.embedder import embed_query

logger = logging.getLogger("alvinai")


async def structured_lookup(
    db: AsyncSession,
    query: str,
    namespace: str,
) -> dict:
    """Handle factual_lookup queries with direct DB search.

    For precise value lookups (torque specs, prices, part numbers),
    retrieve the most relevant chunk and return it directly without LLM generation.
    """
    query_embedding = embed_query(query)

    # Find the single most relevant chunk by cosine similarity
    from sqlalchemy import text as sql_text

    sql = sql_text("""
        SELECT dc.content, d.title,
               1 - (dc.embedding <=> :embedding::vector) AS similarity
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.namespace = :namespace
        ORDER BY dc.embedding <=> :embedding::vector
        LIMIT 1
    """)

    result = await db.execute(
        sql,
        {"embedding": str(query_embedding), "namespace": namespace},
    )
    row = result.fetchone()

    if row and row.similarity >= 0.72:
        return {
            "answer": row.content,
            "namespace": namespace,
            "sources": [{"title": row.title, "snippet": row.content[:200]}],
            "latency_ms": 0.0,
            "query_type": "factual_lookup",
        }

    return {
        "answer": "I couldn't find a specific answer for that in the available documents.",
        "namespace": namespace,
        "sources": [],
        "latency_ms": 0.0,
        "query_type": "factual_lookup",
    }
