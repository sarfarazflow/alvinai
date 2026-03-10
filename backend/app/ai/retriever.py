import hashlib
import json
import logging
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import cache_get, cache_set
from app.core.config import get_settings
from app.ingestion.embedder import embed_query
from app.models.document import Document, DocumentChunk

logger = logging.getLogger("alvinai")
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    namespace: str
    content: str
    chunk_index: int
    score: float
    source: str = ""  # "dense", "sparse", or "hybrid"


@dataclass
class BM25Index:
    corpus: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    doc_titles: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    namespaces: list[str] = field(default_factory=list)
    index: BM25Okapi | None = None


_bm25_cache: dict[str, BM25Index] = {}


async def _build_bm25_index(db: AsyncSession, namespace: str) -> BM25Index:
    """Build BM25 index for a namespace (in-memory, cached)."""
    if namespace in _bm25_cache:
        return _bm25_cache[namespace]

    logger.info("Building BM25 index for namespace: %s", namespace)
    stmt = (
        select(DocumentChunk, Document.title, Document.namespace)
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.namespace == namespace)
        .order_by(DocumentChunk.chunk_index)
    )
    result = await db.execute(stmt)
    rows = result.all()

    idx = BM25Index()
    for chunk, doc_title, doc_namespace in rows:
        idx.corpus.append(chunk.content)
        idx.chunk_ids.append(str(chunk.id))
        idx.doc_titles.append(doc_title)
        idx.doc_ids.append(str(chunk.document_id))
        idx.chunk_indices.append(chunk.chunk_index)
        idx.namespaces.append(doc_namespace)

    if idx.corpus:
        tokenized = [doc.lower().split() for doc in idx.corpus]
        idx.index = BM25Okapi(tokenized)

    _bm25_cache[namespace] = idx
    logger.info("BM25 index built for '%s': %d chunks", namespace, len(idx.corpus))
    return idx


def invalidate_bm25_cache(namespace: str | None = None):
    """Clear BM25 cache (call after ingesting new documents)."""
    if namespace:
        _bm25_cache.pop(namespace, None)
    else:
        _bm25_cache.clear()


async def _dense_search(
    db: AsyncSession,
    query: str,
    namespace: str,
    top_k: int = 10,
    doc_title_filter: str | None = None,
) -> list[RetrievedChunk]:
    """pgvector cosine similarity search with namespace filter."""
    query_embedding = embed_query(query)

    if doc_title_filter:
        sql = text("""
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                   d.title, d.namespace,
                   1 - (dc.embedding <=> cast(:embedding AS vector)) AS similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.namespace = :namespace
              AND UPPER(d.title) LIKE :title_filter
            ORDER BY dc.embedding <=> cast(:embedding AS vector)
            LIMIT :top_k
        """)
        params = {
            "embedding": str(query_embedding),
            "namespace": namespace,
            "title_filter": f"%{doc_title_filter}%",
            "top_k": top_k,
        }
    else:
        sql = text("""
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.content,
                   d.title, d.namespace,
                   1 - (dc.embedding <=> cast(:embedding AS vector)) AS similarity
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE d.namespace = :namespace
            ORDER BY dc.embedding <=> cast(:embedding AS vector)
            LIMIT :top_k
        """)
        params = {
            "embedding": str(query_embedding),
            "namespace": namespace,
            "top_k": top_k,
        }

    result = await db.execute(sql, params)
    rows = result.fetchall()

    return [
        RetrievedChunk(
            chunk_id=str(row.id),
            document_id=str(row.document_id),
            document_title=row.title,
            namespace=row.namespace,
            content=row.content,
            chunk_index=row.chunk_index,
            score=float(row.similarity),
            source="dense",
        )
        for row in rows
    ]


async def _sparse_search(
    db: AsyncSession,
    query: str,
    namespace: str,
    top_k: int = 10,
    doc_title_filter: str | None = None,
) -> list[RetrievedChunk]:
    """BM25 sparse search over chunk texts for a namespace."""
    idx = await _build_bm25_index(db, namespace)
    if not idx.index:
        return []

    tokenized_query = query.lower().split()
    scores = idx.index.get_scores(tokenized_query)

    scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scored_indices:
        if score <= 0:
            continue
        # Filter by document title if specified
        if doc_title_filter and doc_title_filter not in idx.doc_titles[i].upper():
            continue
        results.append(
            RetrievedChunk(
                chunk_id=idx.chunk_ids[i],
                document_id=idx.doc_ids[i],
                document_title=idx.doc_titles[i],
                namespace=idx.namespaces[i],
                content=idx.corpus[i],
                chunk_index=idx.chunk_indices[i],
                score=float(score),
                source="sparse",
            )
        )
        if len(results) >= top_k:
            break

    return results


def _rrf_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion combining dense and sparse ranked lists."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank + 1)
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores, key=scores.get, reverse=True)

    results = []
    for chunk_id in sorted_ids:
        chunk = chunk_map[chunk_id]
        chunk.score = scores[chunk_id]
        chunk.source = "hybrid"
        results.append(chunk)

    return results


def _cache_key(query: str, namespace: str, top_k: int) -> str:
    """Generate cache key for retrieval results."""
    raw = f"{query}:{namespace}:{top_k}"
    return f"rag:{hashlib.md5(raw.encode()).hexdigest()}"


async def retrieve(
    db: AsyncSession,
    query: str,
    namespace: str,
    top_k: int = 10,
    use_cache: bool = True,
    doc_title_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: dense + sparse + RRF fusion.

    Applies namespace-specific similarity threshold:
    - compliance: 0.80
    - all others: 0.72

    If doc_title_filter is provided, only retrieves chunks from documents
    whose title contains the filter string (case-insensitive).
    """
    # Check cache
    if use_cache:
        key = _cache_key(query, namespace, top_k)
        cached = await cache_get(key)
        if cached:
            logger.info("Cache hit for query in namespace '%s'", namespace)
            return [RetrievedChunk(**c) for c in cached]

    # Run dense + sparse searches
    dense_results = await _dense_search(db, query, namespace, top_k=top_k, doc_title_filter=doc_title_filter)
    sparse_results = await _sparse_search(db, query, namespace, top_k=top_k, doc_title_filter=doc_title_filter)

    # Capture dense cosine scores BEFORE fusion overwrites them
    dense_cosine_scores = {c.chunk_id: c.score for c in dense_results}

    # Fuse results
    fused = _rrf_fusion(dense_results, sparse_results)

    # Apply similarity threshold using original dense cosine scores
    threshold = (
        settings.RAG_COMPLIANCE_SIMILARITY_THRESHOLD
        if namespace == "compliance"
        else settings.RAG_SIMILARITY_THRESHOLD
    )

    filtered = []
    for chunk in fused:
        cosine_score = dense_cosine_scores.get(chunk.chunk_id, 0)
        if cosine_score >= threshold:
            filtered.append(chunk)

    results = filtered[:top_k]

    # Cache results
    if use_cache and results:
        key = _cache_key(query, namespace, top_k)
        cache_data = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "document_title": c.document_title,
                "namespace": c.namespace,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "score": c.score,
                "source": c.source,
            }
            for c in results
        ]
        await cache_set(key, cache_data, ttl=settings.RAG_CACHE_TTL)

    logger.info(
        "Retrieved %d chunks for query in namespace '%s' (dense=%d, sparse=%d, fused=%d)",
        len(results), namespace, len(dense_results), len(sparse_results), len(fused),
    )
    return results
