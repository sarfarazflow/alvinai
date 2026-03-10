#!/usr/bin/env python3
"""CLI script to ingest documents into the AlvinAI RAG pipeline.

Usage:
    python scripts/ingest_documents.py --namespace customer_support --path ./docs/manual.pdf
    python scripts/ingest_documents.py --namespace engineering --path ./docs/engineering/
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.db import async_session
from app.ingestion.pipeline import ingest_file
from app.ai.retriever import invalidate_bm25_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("alvinai.ingest")

VALID_NAMESPACES = {
    "customer_support", "engineering", "dealer_sales",
    "compliance", "employee_hr", "vendor",
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def find_files(path: Path) -> list[Path]:
    """Find all supported document files in a path."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        else:
            logger.warning("Unsupported file type: %s", path)
            return []

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(path.rglob(f"*{ext}"))
    return sorted(files)


async def main(namespace: str, path: str):
    target = Path(path)
    if not target.exists():
        logger.error("Path does not exist: %s", path)
        sys.exit(1)

    files = find_files(target)
    if not files:
        logger.error("No supported documents found in: %s", path)
        sys.exit(1)

    logger.info("Found %d documents to ingest into namespace '%s'", len(files), namespace)

    success = 0
    errors = 0
    total_chunks = 0

    async with async_session() as db:
        for i, file_path in enumerate(files, 1):
            try:
                logger.info("[%d/%d] Ingesting: %s", i, len(files), file_path.name)
                result = await ingest_file(
                    db=db,
                    file_path=str(file_path),
                    namespace=namespace,
                )
                success += 1
                total_chunks += result["chunks"]
                logger.info(
                    "  -> %s: %d chunks (%s)",
                    result["title"], result["chunks"], result["doc_type"],
                )
            except Exception as e:
                errors += 1
                logger.error("  -> FAILED: %s — %s", file_path.name, e)

    # Invalidate BM25 cache after ingestion
    invalidate_bm25_cache(namespace)

    # Summary
    print("\n" + "=" * 50)
    print(f"Ingestion Summary — namespace: {namespace}")
    print(f"  Files processed: {success}")
    print(f"  Chunks created:  {total_chunks}")
    print(f"  Errors:          {errors}")
    print("=" * 50)

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into AlvinAI RAG pipeline")
    parser.add_argument(
        "--namespace",
        required=True,
        choices=sorted(VALID_NAMESPACES),
        help="Target namespace for ingestion",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to file or directory to ingest",
    )
    args = parser.parse_args()
    asyncio.run(main(args.namespace, args.path))
