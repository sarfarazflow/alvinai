import logging
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger("alvinai")
settings = get_settings()

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (CPU, singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning L2-normalized vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string, returning L2-normalized vector."""
    model = get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
