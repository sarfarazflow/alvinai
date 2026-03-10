import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.ingestion.parser import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.indexer import index_document

logger = logging.getLogger("alvinai")


async def ingest_file(
    db: AsyncSession,
    file_path: str,
    namespace: str,
    metadata: dict | None = None,
) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → index.

    Returns summary dict with document id, title, chunk count.
    """
    logger.info("Ingesting file: %s (namespace=%s)", file_path, namespace)

    # 1. Parse
    parsed = parse_document(file_path)
    if not parsed["text"].strip():
        raise ValueError(f"No text extracted from {file_path}")

    # 2. Chunk
    chunks = chunk_text(parsed["text"], namespace)
    if not chunks:
        raise ValueError(f"No chunks generated from {file_path}")

    # Use original filename as title if available (for uploads via temp files)
    title = parsed["title"]
    if metadata and "original_filename" in metadata:
        from pathlib import Path
        title = Path(metadata["original_filename"]).stem

    # 3. Embed + Index
    doc = await index_document(
        db=db,
        title=title,
        source_path=file_path,
        namespace=namespace,
        doc_type=parsed["doc_type"],
        chunks=chunks,
        metadata={
            **(metadata or {}),
            "pages": parsed.get("pages", 0),
            "original_title": parsed["title"],
        },
    )

    return {
        "document_id": str(doc.id),
        "title": doc.title,
        "namespace": namespace,
        "chunks": len(chunks),
        "doc_type": parsed["doc_type"],
    }
