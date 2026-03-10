import os
import tempfile
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.ingestion.pipeline import ingest_file
from app.ai.retriever import invalidate_bm25_cache

logger = logging.getLogger("alvinai")
router = APIRouter(prefix="/ingest", tags=["ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
VALID_NAMESPACES = {
    "customer_support", "engineering", "dealer_sales",
    "compliance", "employee_hr", "vendor",
}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = Form(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload and ingest a document into the RAG pipeline.

    Accepts PDF, DOCX, TXT, MD, HTML files.
    Requires admin role.
    """
    # Validate namespace
    if namespace not in VALID_NAMESPACES:
        raise HTTPException(400, f"Invalid namespace. Must be one of: {VALID_NAMESPACES}")

    # Check admin role
    if user.role != "admin":
        raise HTTPException(403, "Only admins can upload documents")

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Save to temp file and ingest
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await ingest_file(
            db=db,
            file_path=tmp_path,
            namespace=namespace,
            metadata={"original_filename": file.filename},
        )

        # Invalidate BM25 cache for this namespace
        invalidate_bm25_cache(namespace)

        return {
            "status": "success",
            "document_id": result["document_id"],
            "title": result["title"],
            "namespace": result["namespace"],
            "chunks": result["chunks"],
            "doc_type": result["doc_type"],
        }

    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file.filename, e)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
