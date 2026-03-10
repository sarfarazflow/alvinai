import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.document import Document, DocumentChunk
from app.ingestion.embedder import embed_texts

logger = logging.getLogger("alvinai")

BATCH_SIZE = 64


async def index_document(
    db: AsyncSession,
    title: str,
    source_path: str,
    namespace: str,
    doc_type: str,
    chunks: list[dict],
    metadata: dict | None = None,
) -> Document:
    """Index a parsed+chunked document into PostgreSQL + pgvector.

    1. Create Document record
    2. Embed all chunks in batches
    3. Create DocumentChunk records with embeddings
    4. Reindex if batch is large (>1000 chunks)
    """
    doc = Document(
        title=title,
        source_path=source_path,
        namespace=namespace,
        doc_type=doc_type,
        total_chunks=len(chunks),
        metadata_=metadata,
    )
    db.add(doc)
    await db.flush()

    chunk_texts = [c["content"] for c in chunks]

    for batch_start in range(0, len(chunk_texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(chunk_texts))
        batch_texts = chunk_texts[batch_start:batch_end]
        batch_embeddings = embed_texts(batch_texts)

        for chunk_data, embedding in zip(chunks[batch_start:batch_end], batch_embeddings):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["content"],
                embedding=embedding,
                token_count=chunk_data["token_count"],
            )
            db.add(chunk)

        logger.info(
            "Indexed batch %d-%d of %d chunks for doc '%s'",
            batch_start, batch_end, len(chunk_texts), title,
        )

    await db.flush()

    if len(chunks) > 1000:
        logger.info("Large batch — reindexing pgvector index")
        await db.execute(text("REINDEX INDEX IF EXISTS ix_document_chunks_embedding"))

    await db.commit()
    logger.info(
        "Indexed document '%s' (%d chunks) into namespace '%s'",
        title, len(chunks), namespace,
    )
    return doc
