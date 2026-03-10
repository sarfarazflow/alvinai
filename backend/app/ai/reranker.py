import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger("alvinai")

_model: CrossEncoder | None = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker() -> CrossEncoder:
    """Lazy-load cross-encoder reranker (CPU, singleton)."""
    global _model
    if _model is None:
        logger.info("Loading reranker model: %s", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        logger.info("Reranker model loaded")
    return _model


def rerank(query: str, chunks: list, top_k: int = 5) -> list:
    """Rerank retrieved chunks using cross-encoder.

    Args:
        query: User query string
        chunks: List of RetrievedChunk objects from retriever
        top_k: Number of top results to return after reranking

    Returns:
        Top-k chunks re-scored and sorted by cross-encoder relevance
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        return chunks

    model = _get_reranker()

    # Prepare query-document pairs for cross-encoder
    pairs = [(query, chunk.content) for chunk in chunks]
    scores = model.predict(pairs)

    # Attach reranker scores and sort
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in scored[:top_k]:
        chunk.score = float(score)
        results.append(chunk)

    logger.info("Reranked %d chunks to top-%d", len(chunks), top_k)
    return results
