# SOTA RAG Pipeline v2 — Detailed Implementation Plan

## Context

AlvinAI's current RAG pipeline uses dated components (MiniLM-L6 384-dim embeddings, fixed-size chunking, basic cross-encoder). Upgrading to 2025-2026 SOTA for 12B-14B models (Mistral Nemo 12B, Gemma 3 12B, Phi-4 14B, Qwen3 14B) with 128K context windows. All RAG components stay on Hetzner CPU; only final LLM inference hits GPU.

---

## Phase 1: Embedding Upgrade (nomic-embed-text-v1.5)

**Highest ROI — biggest single retrieval quality gain**

### 1.1 Update `backend/app/core/config.py`

Add new settings, change defaults:

```python
# Change these defaults:
EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"   # was: all-MiniLM-L6-v2
EMBEDDING_DIM: int = 768                                     # was: 384

# Add new:
EMBEDDING_QUERY_PREFIX: str = "search_query: "    # nomic requires task prefixes
EMBEDDING_DOC_PREFIX: str = "search_document: "
```

**File:** [config.py](backend/app/core/config.py) — lines 32-33, add lines after 38

### 1.2 Update `backend/app/ingestion/embedder.py`

Add task-prefix support for nomic-embed (improves retrieval by distinguishing query vs document embeddings):

```python
# embedder.py — full rewrite

import logging
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger("alvinai")
settings = get_settings()

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)
        logger.info("Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document texts with document prefix for asymmetric retrieval."""
    model = get_embedding_model()
    prefixed = [f"{settings.EMBEDDING_DOC_PREFIX}{t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a query with query prefix for asymmetric retrieval."""
    model = get_embedding_model()
    prefixed = f"{settings.EMBEDDING_QUERY_PREFIX}{query}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()
```

**File:** [embedder.py](backend/app/ingestion/embedder.py) — full rewrite (33 lines → ~35 lines)

### 1.3 Update `backend/app/models/document.py`

Change vector dimension:

```python
# Line 30: change Vector(384) to Vector(768)
embedding = mapped_column(Vector(768), nullable=True)
```

**File:** [document.py](backend/app/models/document.py) — line 30

### 1.4 Database Migration

```bash
cd backend
uv run alembic revision --autogenerate -m "upgrade embedding dim 384 to 768"
```

The migration will need manual editing to:
1. Drop the existing ivfflat index on `document_chunks.embedding`
2. Alter column type from `vector(384)` to `vector(768)`
3. Create new HNSW index (Phase 6 — can do both in same migration)

**Manual migration SQL:**
```sql
-- Drop old ivfflat index
DROP INDEX IF EXISTS ix_document_chunks_embedding;

-- Alter embedding column dimension
ALTER TABLE document_chunks
  ALTER COLUMN embedding TYPE vector(768);

-- Create HNSW index (better than ivfflat for accuracy + no tuning needed)
CREATE INDEX ix_document_chunks_embedding_hnsw
  ON document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

### 1.5 Re-embed All Existing Documents

Write a one-time script `scripts/reembed_all.py`:

```python
"""Re-embed all document chunks with the new embedding model.

Usage: uv run python -m scripts.reembed_all
"""
import asyncio
from sqlalchemy import select, update
from app.core.db import get_async_session
from app.models.document import DocumentChunk
from app.ingestion.embedder import embed_texts

BATCH_SIZE = 64

async def main():
    async for db in get_async_session():
        # Fetch all chunks
        result = await db.execute(select(DocumentChunk.id, DocumentChunk.content))
        rows = result.all()
        print(f"Re-embedding {len(rows)} chunks...")

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            texts = [r.content for r in batch]
            embeddings = embed_texts(texts)

            for row, emb in zip(batch, embeddings):
                await db.execute(
                    update(DocumentChunk)
                    .where(DocumentChunk.id == row.id)
                    .values(embedding=emb)
                )
            await db.flush()
            print(f"  Re-embedded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

        await db.commit()
        print("Done.")

asyncio.run(main())
```

### 1.6 Update `backend/pyproject.toml`

Add `einops` dependency (required by nomic-embed):
```toml
# Under [project] dependencies, add:
"einops>=0.7",
```

---

## Phase 2: Parent-Child Chunking

**Second highest ROI — richer context for LLM**

### 2.1 Add `ParentChunk` Model to `backend/app/models/document.py`

```python
class ParentChunk(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "parent_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    heading: Mapped[str | None] = mapped_column(String(500), nullable=True)

    document = relationship("Document", back_populates="parent_chunks")
    children = relationship("DocumentChunk", back_populates="parent_chunk", lazy="selectin")


# Update Document class — add relationship:
parent_chunks = relationship("ParentChunk", back_populates="document", lazy="selectin")


# Update DocumentChunk class — add FK:
parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
    UUID(as_uuid=True), ForeignKey("parent_chunks.id"), nullable=True, index=True
)
parent_chunk = relationship("ParentChunk", back_populates="children")
```

**File:** [document.py](backend/app/models/document.py) — add class after `Document`, modify `DocumentChunk`

### 2.2 Rewrite `backend/app/ingestion/chunker.py`

```python
import logging
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("alvinai")

# Parent chunk: large semantic sections for LLM context
PARENT_CONFIGS = {
    "customer_support": {"chunk_size": 2000, "chunk_overlap": 200},
    "engineering":      {"chunk_size": 1500, "chunk_overlap": 200},
    "dealer_sales":     {"chunk_size": 1500, "chunk_overlap": 150},
    "compliance":       {"chunk_size": 1200, "chunk_overlap": 150},
    "employee_hr":      {"chunk_size": 1500, "chunk_overlap": 200},
    "vendor":           {"chunk_size": 1500, "chunk_overlap": 200},
}

# Child chunk: small precise chunks for embedding retrieval
CHILD_CONFIGS = {
    "customer_support": {"chunk_size": 400, "chunk_overlap": 50},
    "engineering":      {"chunk_size": 300, "chunk_overlap": 50},
    "dealer_sales":     {"chunk_size": 300, "chunk_overlap": 50},
    "compliance":       {"chunk_size": 250, "chunk_overlap": 50},
    "employee_hr":      {"chunk_size": 300, "chunk_overlap": 50},
    "vendor":           {"chunk_size": 300, "chunk_overlap": 50},
}

# Heading patterns for semantic boundary detection
HEADING_PATTERN = re.compile(
    r"^(?:"
    r"#{1,4}\s+.+|"                          # Markdown headings
    r"(?:Section|Article|Chapter)\s+\d+|"    # Legal/regulatory sections
    r"\d+\.\d*\s+[A-Z].+|"                  # Numbered sections (4.2 Brake System)
    r"[A-Z][A-Z\s]{4,}$"                    # ALL CAPS headings
    r")",
    re.MULTILINE,
)


def _split_by_semantic_boundaries(text: str) -> list[dict]:
    """Split text into semantic sections based on headings.

    Returns list of {"heading": str|None, "content": str}.
    """
    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        return [{"heading": None, "content": text}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group().strip()
        content = text[start:end].strip()
        if content:
            sections.append({"heading": heading, "content": content})

    # Capture any text before the first heading
    if matches and matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.insert(0, {"heading": None, "content": preamble})

    return sections


def chunk_text_parent_child(text: str, namespace: str) -> dict:
    """Two-tier semantic chunking: parent (LLM context) + child (retrieval).

    Returns:
        {
            "parents": [{"content", "chunk_index", "token_count", "heading"}],
            "children": [{"content", "chunk_index", "token_count", "parent_index"}],
        }
    """
    parent_config = PARENT_CONFIGS.get(namespace, PARENT_CONFIGS["customer_support"])
    child_config = CHILD_CONFIGS.get(namespace, CHILD_CONFIGS["customer_support"])

    # Step 1: Split into semantic sections
    sections = _split_by_semantic_boundaries(text)

    # Step 2: Create parent chunks from semantic sections
    # If a section is too large, split it further with RecursiveCharacterTextSplitter
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_config["chunk_size"],
        chunk_overlap=parent_config["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_config["chunk_size"],
        chunk_overlap=child_config["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents = []
    children = []
    parent_idx = 0

    for section in sections:
        # Split section into parent-sized chunks if needed
        parent_texts = parent_splitter.split_text(section["content"])

        for p_text in parent_texts:
            parents.append({
                "content": p_text,
                "chunk_index": parent_idx,
                "token_count": len(p_text.split()),
                "heading": section["heading"],
            })

            # Split each parent into child chunks
            child_texts = child_splitter.split_text(p_text)
            for c_text in child_texts:
                children.append({
                    "content": c_text,
                    "chunk_index": len(children),
                    "token_count": len(c_text.split()),
                    "parent_index": parent_idx,
                })

            parent_idx += 1

    logger.info(
        "Chunked text into %d parents + %d children (namespace=%s)",
        len(parents), len(children), namespace,
    )
    return {"parents": parents, "children": children}


# Keep the old function for backward compatibility during migration
def chunk_text(text: str, namespace: str) -> list[dict]:
    """Legacy flat chunking — DEPRECATED, use chunk_text_parent_child."""
    result = chunk_text_parent_child(text, namespace)
    # Return children only (flat list, same interface as before)
    return [
        {"content": c["content"], "chunk_index": c["chunk_index"], "token_count": c["token_count"]}
        for c in result["children"]
    ]
```

**File:** [chunker.py](backend/app/ingestion/chunker.py) — full rewrite

### 2.3 Update `backend/app/ingestion/indexer.py`

Update to store both parent and child chunks:

```python
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.models.document import Document, DocumentChunk, ParentChunk
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
    parent_chunks: list[dict] | None = None,
    metadata: dict | None = None,
) -> Document:
    """Index a parsed+chunked document into PostgreSQL + pgvector.

    If parent_chunks is provided, creates parent-child relationships.
    Otherwise, falls back to flat indexing (backward compatible).
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

    # --- Index parent chunks (no embeddings — used for LLM context only) ---
    parent_id_map = {}  # parent_index -> ParentChunk.id
    if parent_chunks:
        for p in parent_chunks:
            parent = ParentChunk(
                document_id=doc.id,
                chunk_index=p["chunk_index"],
                content=p["content"],
                token_count=p["token_count"],
                heading=p.get("heading"),
            )
            db.add(parent)
            await db.flush()
            parent_id_map[p["chunk_index"]] = parent.id

    # --- Index child chunks (with embeddings — used for retrieval) ---
    chunk_texts = [c["content"] for c in chunks]

    for batch_start in range(0, len(chunk_texts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(chunk_texts))
        batch_texts = chunk_texts[batch_start:batch_end]
        batch_embeddings = embed_texts(batch_texts)

        for chunk_data, embedding in zip(chunks[batch_start:batch_end], batch_embeddings):
            parent_idx = chunk_data.get("parent_index")
            parent_id = parent_id_map.get(parent_idx) if parent_idx is not None else None

            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["content"],
                embedding=embedding,
                token_count=chunk_data["token_count"],
                parent_chunk_id=parent_id,
            )
            db.add(chunk)

        logger.info(
            "Indexed batch %d-%d of %d chunks for doc '%s'",
            batch_start, batch_end, len(chunk_texts), title,
        )

    await db.flush()

    if len(chunks) > 1000:
        logger.info("Large batch — reindexing pgvector index")
        await db.execute(text("REINDEX INDEX IF EXISTS ix_document_chunks_embedding_hnsw"))

    await db.commit()
    logger.info(
        "Indexed document '%s' (%d parents + %d children) into namespace '%s'",
        title, len(parent_chunks or []), len(chunks), namespace,
    )
    return doc
```

**File:** [indexer.py](backend/app/ingestion/indexer.py) — rewrite

### 2.4 Update `backend/app/ingestion/pipeline.py`

Use parent-child chunking:

```python
async def ingest_file(
    db: AsyncSession,
    file_path: str,
    namespace: str,
    metadata: dict | None = None,
) -> dict:
    logger.info("Ingesting file: %s (namespace=%s)", file_path, namespace)

    parsed = parse_document(file_path)
    if not parsed["text"].strip():
        raise ValueError(f"No text extracted from {file_path}")

    # Parent-child chunking
    from app.ingestion.chunker import chunk_text_parent_child
    chunk_result = chunk_text_parent_child(parsed["text"], namespace)

    if not chunk_result["children"]:
        raise ValueError(f"No chunks generated from {file_path}")

    title = parsed["title"]
    if metadata and "original_filename" in metadata:
        from pathlib import Path
        title = Path(metadata["original_filename"]).stem

    doc = await index_document(
        db=db,
        title=title,
        source_path=file_path,
        namespace=namespace,
        doc_type=parsed["doc_type"],
        chunks=chunk_result["children"],
        parent_chunks=chunk_result["parents"],
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
        "chunks": len(chunk_result["children"]),
        "parent_chunks": len(chunk_result["parents"]),
        "doc_type": parsed["doc_type"],
    }
```

**File:** [pipeline.py](backend/app/ingestion/pipeline.py) — rewrite `ingest_file`

### 2.5 Update `backend/app/ai/retriever.py` — Parent Resolution

Add parent chunk resolution after retrieval. The key change: retrieve by child (precise embedding match), but return parent content for prompt assembly.

Add to `RetrievedChunk` dataclass:
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    namespace: str
    content: str           # child content (for reranking)
    chunk_index: int
    score: float
    source: str = ""
    parent_content: str = ""   # NEW: parent content (for prompt assembly)
    parent_heading: str = ""   # NEW: section heading
```

Add new function after `_rrf_fusion`:
```python
async def _resolve_parents(
    db: AsyncSession, chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Resolve parent chunks for retrieved children.

    Deduplicates parents (multiple children may share a parent).
    Falls back to child content if no parent exists.
    """
    chunk_ids = [c.chunk_id for c in chunks]
    if not chunk_ids:
        return chunks

    sql = text("""
        SELECT dc.id AS chunk_id, pc.content AS parent_content, pc.heading
        FROM document_chunks dc
        LEFT JOIN parent_chunks pc ON dc.parent_chunk_id = pc.id
        WHERE dc.id = ANY(:chunk_ids)
    """)
    result = await db.execute(sql, {"chunk_ids": chunk_ids})
    parent_map = {
        str(row.chunk_id): (row.parent_content, row.heading)
        for row in result.fetchall()
    }

    # Deduplicate: track seen parent contents to avoid sending duplicate parents
    seen_parents = set()
    deduped = []
    for chunk in chunks:
        parent_content, heading = parent_map.get(chunk.chunk_id, (None, None))
        if parent_content:
            if parent_content not in seen_parents:
                seen_parents.add(parent_content)
                chunk.parent_content = parent_content
                chunk.parent_heading = heading or ""
                deduped.append(chunk)
        else:
            # No parent — use child content as fallback
            chunk.parent_content = chunk.content
            deduped.append(chunk)

    logger.info("Resolved %d children → %d unique parents", len(chunks), len(deduped))
    return deduped
```

Update `retrieve()` — add parent resolution before return:
```python
# After: results = filtered[:top_k]
# Add:
results = await _resolve_parents(db, results)
```

**File:** [retriever.py](backend/app/ai/retriever.py) — add `parent_content`/`parent_heading` fields, add `_resolve_parents()`, update `retrieve()`

### 2.6 Alembic Migration

```bash
uv run alembic revision --autogenerate -m "add parent chunks and upgrade embeddings to 768"
```

**Manual edits to generated migration:**
```python
def upgrade():
    # 1. Create parent_chunks table
    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id"), index=True),
        sa.Column("chunk_index", sa.Integer()),
        sa.Column("content", sa.Text()),
        sa.Column("token_count", sa.Integer(), default=0),
        sa.Column("heading", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 2. Add parent_chunk_id FK to document_chunks
    op.add_column("document_chunks",
        sa.Column("parent_chunk_id", sa.UUID(), sa.ForeignKey("parent_chunks.id"), nullable=True)
    )
    op.create_index("ix_document_chunks_parent_chunk_id", "document_chunks", ["parent_chunk_id"])

    # 3. Drop old ivfflat index
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding")

    # 4. Alter embedding dimension 384 → 768
    # Must set all embeddings to NULL first (dimension mismatch)
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(768)")

    # 5. Create HNSW index
    op.execute("""
        CREATE INDEX ix_document_chunks_embedding_hnsw
        ON document_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

def downgrade():
    op.drop_index("ix_document_chunks_embedding_hnsw")
    op.execute("UPDATE document_chunks SET embedding = NULL")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(384)")
    op.drop_index("ix_document_chunks_parent_chunk_id")
    op.drop_column("document_chunks", "parent_chunk_id")
    op.drop_table("parent_chunks")
```

---

## Phase 3: Structured Prompt Assembly + CoT

### 3.1 Rewrite `backend/app/ai/prompt.py`

Update `format_context()` to use parent content with structured layout:

```python
CONTEXT_TEMPLATE = """## Retrieved Documents

{context_blocks}

---

Instructions:
1. Analyze the retrieved documents above to answer the question.
2. Cite specific documents by name when referencing information.
3. If documents contain conflicting information, note the discrepancy.
4. If the answer is not fully covered by the documents, state what is missing.
5. Do not reproduce the reference material verbatim — synthesize and explain."""


def format_context(chunks: list) -> str:
    """Format retrieved chunks with structured metadata for the LLM."""
    if not chunks:
        return ""

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        title = getattr(chunk, "document_title", "Unknown")
        heading = getattr(chunk, "parent_heading", "")
        score = getattr(chunk, "score", 0.0)

        # Use parent content if available (richer context), else child content
        content = getattr(chunk, "parent_content", "") or chunk.content

        header = f"### Document {i}: {title}"
        if heading:
            header += f" ({heading})"
        header += f"\nRelevance: {score:.2f}"

        blocks.append(f"{header}\n---\n{content}")

    context_blocks = "\n\n".join(blocks)
    return CONTEXT_TEMPLATE.format(context_blocks=context_blocks)
```

**File:** [prompt.py](backend/app/ai/prompt.py) — rewrite `CONTEXT_TEMPLATE` and `format_context()`

---

## Phase 4: Query Processor (Rewriting + HyDE)

### 4.1 Create `backend/app/ai/query_processor.py` (NEW FILE)

```python
"""Query processing: rewrite, HyDE, and sub-question decomposition.

All processing uses the serving LLM via llm_client.generate().
Results are cached in Redis to avoid redundant LLM calls.
"""
import hashlib
import logging
import re

from app.ai.llm_client import generate
from app.core.cache import cache_get, cache_set

logger = logging.getLogger("alvinai")

REWRITE_PROMPT = """Rewrite this user question into a clear, specific search query for an automotive knowledge base. Keep it concise (under 30 words). Do not answer the question — only rewrite it.

User question: {query}
Rewritten query:"""

HYDE_PROMPT = """Write a short paragraph (3-4 sentences) that would be the ideal answer to this automotive question. Write as if you are quoting from a technical manual. Do not hedge or say "I don't know."

Question: {query}
Ideal answer paragraph:"""

DECOMPOSE_KEYWORDS = re.compile(
    r"\b(compare|difference|between|versus|vs\.?|both|each)\b", re.IGNORECASE
)


async def rewrite_query(query: str, namespace: str = "") -> str:
    """Rewrite a vague/colloquial query into a precise search query.

    Returns the original query if rewriting fails or query is already specific.
    """
    # Skip short, specific queries (likely already precise)
    if len(query.split()) <= 4:
        return query

    cache_key = f"qr:{hashlib.md5(query.encode()).hexdigest()}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    try:
        rewritten = await generate(
            query=REWRITE_PROMPT.format(query=query),
            max_tokens=60,
            temperature=0.1,
        )
        rewritten = rewritten.strip().strip('"')
        if rewritten and len(rewritten) > 5:
            await cache_set(cache_key, rewritten, ttl=3600)
            logger.info("Query rewritten: '%s' → '%s'", query[:50], rewritten[:50])
            return rewritten
    except Exception as e:
        logger.warning("Query rewriting failed: %s", e)

    return query


async def generate_hyde(query: str, namespace: str = "") -> str:
    """Generate a Hypothetical Document Embedding (HyDE).

    Creates a hypothetical ideal answer that is closer in embedding space
    to real documents than the short user query.
    """
    cache_key = f"hyde:{hashlib.md5(query.encode()).hexdigest()}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    try:
        hypothetical = await generate(
            query=HYDE_PROMPT.format(query=query),
            max_tokens=150,
            temperature=0.3,
        )
        if hypothetical and len(hypothetical) > 20:
            await cache_set(cache_key, hypothetical, ttl=3600)
            logger.info("HyDE generated for: '%s' (%d chars)", query[:50], len(hypothetical))
            return hypothetical
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)

    return query


async def decompose_query(query: str) -> list[str]:
    """Decompose comparison queries into sub-questions.

    Only triggers for queries with comparison keywords.
    Returns [original_query] if no decomposition needed.
    """
    if not DECOMPOSE_KEYWORDS.search(query):
        return [query]

    try:
        result = await generate(
            query=(
                f"Break this comparison question into 2-3 separate lookup questions. "
                f"Return one question per line, no numbering.\n\n"
                f"Question: {query}\n\nSub-questions:"
            ),
            max_tokens=100,
            temperature=0.1,
        )
        sub_questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
        if len(sub_questions) >= 2:
            logger.info("Decomposed query into %d sub-questions", len(sub_questions))
            return sub_questions
    except Exception as e:
        logger.warning("Query decomposition failed: %s", e)

    return [query]
```

### 4.2 Update `backend/app/ai/pipeline.py`

Insert query processing between classification and retrieval:

```python
# Add import at top:
from app.ai.query_processor import rewrite_query, generate_hyde, decompose_query

# In the document_search branch of run_query(), BEFORE retrieve():

    if query_type == "document_search" and db is not None:
        # --- Query processing (NEW) ---
        search_query = await rewrite_query(query, namespace)

        # For complex queries, decompose and retrieve for each sub-question
        sub_queries = await decompose_query(search_query)

        all_chunks = []
        for sq in sub_queries:
            # Use HyDE embedding for dense search
            hyde_text = await generate_hyde(sq, namespace)

            chunks = await retrieve(
                db, hyde_text, namespace,
                top_k=settings.RAG_TOP_K,
                doc_title_filter=doc_ref,
            )
            all_chunks.extend(chunks)

        # Deduplicate chunks by chunk_id
        seen = set()
        chunks = []
        for c in all_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                chunks.append(c)

        # Rerank the merged set
        if chunks:
            chunks = rerank(query, chunks, top_k=7)  # increased from 5 to 7

        # ... rest of pipeline unchanged
```

**File:** [pipeline.py](backend/app/ai/pipeline.py) — modify `document_search` branch

---

## Phase 5: Reranker Upgrade

### 5.1 Update `backend/app/ai/reranker.py`

One-line change:

```python
# Line 7: change model name
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # was: L-6-v2
```

And update the default top_k in the `rerank()` signature:

```python
def rerank(query: str, chunks: list, top_k: int = 7) -> list:  # was: 5
```

**File:** [reranker.py](backend/app/ai/reranker.py) — lines 7 and 20

---

## Phase 6: Adaptive Retrieval Depth

### 6.1 Update `backend/app/core/config.py`

```python
# Add after RAG_CACHE_TTL:
RAG_RERANK_DEPTH: int = 15           # retrieve this many, rerank to top_k
RAG_TOP_K_SIMPLE: int = 5            # factual queries
RAG_TOP_K_COMPLEX: int = 15          # comparison/multi-hop queries
RAG_TOP_K_COMPLIANCE: int = 10       # high-recall for compliance
```

### 6.2 Update retrieval depth in `backend/app/ai/pipeline.py`

```python
# In run_query(), compute adaptive top_k based on query type and namespace:
if query_type == "document_search" and db is not None:
    # Adaptive retrieval depth
    if namespace == "compliance":
        retrieve_depth = settings.RAG_TOP_K_COMPLIANCE
    elif len(sub_queries) > 1:  # complex/comparison query
        retrieve_depth = settings.RAG_TOP_K_COMPLEX
    else:
        retrieve_depth = settings.RAG_TOP_K

    # Use retrieve_depth instead of settings.RAG_TOP_K in retrieve() calls
```

---

## Phase 7: Multi-Model Chat Templates

### 7.1 Update `backend/app/core/config.py`

```python
# Add:
CHAT_TEMPLATE: str = "mistral"  # mistral | gemma | phi | qwen
```

### 7.2 Update `backend/app/ai/llm_client.py`

Replace `_format_mistral_prompt` with a template router:

```python
CHAT_TEMPLATES = {
    "mistral": {
        "bos": "<s>",
        "system_start": "[INST] ", "system_end": "\n\n",
        "user_start": "[INST] ", "user_end": " [/INST]",
        "assistant_start": " ", "assistant_end": "</s>",
    },
    "gemma": {
        "bos": "",
        "system_start": "<start_of_turn>user\n", "system_end": "\n",
        "user_start": "<start_of_turn>user\n", "user_end": "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n", "assistant_end": "<end_of_turn>\n",
    },
    "phi": {
        "bos": "",
        "system_start": "<|system|>\n", "system_end": "<|end|>\n",
        "user_start": "<|user|>\n", "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n", "assistant_end": "<|end|>\n",
    },
    "qwen": {
        "bos": "",
        "system_start": "<|im_start|>system\n", "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n", "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n", "assistant_end": "<|im_end|>\n",
    },
}


def _format_prompt(messages: list[dict]) -> str:
    """Format messages using the configured chat template."""
    template = CHAT_TEMPLATES.get(settings.CHAT_TEMPLATE, CHAT_TEMPLATES["mistral"])
    prompt = template["bos"]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"{template['system_start']}{content}{template['system_end']}"
        elif role == "user":
            prompt += f"{template['user_start']}{content}{template['user_end']}"
        elif role == "assistant":
            prompt += f"{template['assistant_start']}{content}{template['assistant_end']}"

    # Open the assistant turn for generation
    prompt += template["assistant_start"]
    return prompt
```

Replace `_format_mistral_prompt` calls with `_format_prompt`.

**Note:** When using vLLM's OpenAI-compatible `/chat/completions` endpoint, the template is applied server-side. This custom formatting is only needed for the RunPod serverless `/run` endpoint.

**File:** [llm_client.py](backend/app/ai/llm_client.py) — replace `_format_mistral_prompt`, update references

---

## Phase 8: Online Faithfulness Check

### 8.1 Create `backend/app/ai/verifier.py` (NEW FILE)

```python
"""Online faithfulness verification — catches hallucinations before response.

Mandatory for compliance namespace, sampled (10%) for others.
"""
import logging
import random

from app.ai.llm_client import generate

logger = logging.getLogger("alvinai")

VERIFY_PROMPT = """You are a fact-checker. Given an answer and its source documents, determine if the answer is FAITHFUL — meaning every claim in the answer is supported by the source documents.

Source documents:
{contexts}

Answer to verify:
{answer}

Reply with ONLY one word: FAITHFUL or UNFAITHFUL"""


async def verify_faithfulness(answer: str, context: str) -> bool:
    """Check if the answer is grounded in the provided context."""
    try:
        result = await generate(
            query=VERIFY_PROMPT.format(contexts=context, answer=answer),
            max_tokens=10,
            temperature=0.0,
        )
        is_faithful = "faithful" in result.strip().lower() and "unfaithful" not in result.strip().lower()
        logger.info("Faithfulness check: %s", "PASS" if is_faithful else "FAIL")
        return is_faithful
    except Exception as e:
        logger.warning("Faithfulness check failed: %s — allowing response", e)
        return True  # fail-open for non-compliance


async def verify_and_maybe_retry(
    query: str,
    answer: str,
    context: str,
    system_prompt: str,
    namespace: str,
) -> str:
    """Verify faithfulness. For compliance, always check. For others, sample 10%.

    If unfaithful and compliance: regenerate with stricter prompt.
    If unfaithful and non-compliance: log warning, return original.
    """
    is_compliance = namespace == "compliance"

    # Only check 10% of non-compliance queries
    if not is_compliance and random.random() > 0.10:
        return answer

    is_faithful = await verify_faithfulness(answer, context)

    if is_faithful:
        return answer

    if is_compliance:
        # Regenerate with stricter instructions
        logger.warning("Compliance answer failed faithfulness — regenerating")
        stricter = (
            f"{system_prompt}\n\n"
            "CRITICAL: You MUST only state facts that appear in the provided documents. "
            "If a fact is not explicitly stated in the documents, do NOT include it. "
            "It is better to say 'I cannot confirm this' than to state something unsupported."
        )
        return await generate(
            query=query,
            system_prompt=stricter,
            context=context,
            max_tokens=512,
            temperature=0.1,
        )
    else:
        logger.warning("Non-compliance answer failed faithfulness check (logged only)")
        return answer
```

### 8.2 Update `backend/app/ai/pipeline.py`

Add verification after generation in `document_search` branch:

```python
# Add import:
from app.ai.verifier import verify_and_maybe_retry

# After: answer = await generate(...)
# Add:
answer = await verify_and_maybe_retry(
    query=query,
    answer=answer,
    context=context,
    system_prompt=system_prompt,
    namespace=namespace,
)
```

---

## Phase 9: Update pyproject.toml Dependencies

```toml
# Add to [project] dependencies:
"einops>=0.7",          # required by nomic-embed-text-v1.5
```

No other new dependencies needed — `sentence-transformers`, `rank-bm25`, `langchain-text-splitters` already present.

---

## Summary: Files Changed

| File | Change Type | Phase |
|------|------------|-------|
| `backend/app/core/config.py` | Edit (add settings) | 1, 6, 7 |
| `backend/app/ingestion/embedder.py` | Rewrite | 1 |
| `backend/app/models/document.py` | Edit (add ParentChunk, modify DocumentChunk) | 1, 2 |
| `backend/app/ingestion/chunker.py` | Rewrite | 2 |
| `backend/app/ingestion/indexer.py` | Rewrite | 2 |
| `backend/app/ingestion/pipeline.py` | Edit | 2 |
| `backend/app/ai/retriever.py` | Edit (add parent resolution, adaptive depth) | 2, 6 |
| `backend/app/ai/prompt.py` | Rewrite `format_context` + `CONTEXT_TEMPLATE` | 3 |
| `backend/app/ai/query_processor.py` | **NEW FILE** | 4 |
| `backend/app/ai/pipeline.py` | Edit (add query processing, verification) | 4, 6, 8 |
| `backend/app/ai/reranker.py` | Edit (2 lines) | 5 |
| `backend/app/ai/llm_client.py` | Edit (multi-template) | 7 |
| `backend/app/ai/verifier.py` | **NEW FILE** | 8 |
| `backend/pyproject.toml` | Edit (add einops) | 9 |
| `scripts/reembed_all.py` | **NEW FILE** (one-time migration script) | 1 |
| `alembic/versions/xxx_upgrade_embeddings.py` | **NEW FILE** (migration) | 1, 2 |

---

## Verification Plan

| Phase | Test | Command |
|-------|------|---------|
| 1 | Embedding model loads, produces 768-dim vectors | `uv run python -c "from app.ingestion.embedder import embed_query; v = embed_query('test'); print(len(v))"` |
| 2 | Parent-child chunking produces correct structure | `uv run pytest tests/ingestion/test_chunker.py -v` |
| 2 | DB migration runs cleanly | `uv run alembic upgrade head` |
| 2 | Re-embedding script completes | `uv run python -m scripts.reembed_all` |
| 3 | Prompt format includes structured headers | `uv run pytest tests/ai/test_prompt.py -v` |
| 4 | Query rewriting produces better search terms | Manual test with 10 vague queries |
| 5 | Reranker loads new model | `uv run python -c "from app.ai.reranker import _get_reranker; _get_reranker()"` |
| ALL | Full RAGAS eval — compare v1 vs v2 | `uv run python -m scripts.run_evaluation --eval-file data/eval/engineering_eval.jsonl` |
| ALL | Latency benchmark stays within targets | P95 < 8s cold, < 200ms cached |
| ALL | Lint passes | `uv run ruff check . && uv run ruff format .` |
