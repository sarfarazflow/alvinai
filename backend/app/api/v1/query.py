import uuid
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.schemas.query import QueryRequest, QueryResponse
from app.ai.pipeline import run_query

logger = logging.getLogger("alvinai")
router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Get or create conversation
    if req.conversation_id:
        conv_id = uuid.UUID(req.conversation_id)
    else:
        conv = Conversation(user_id=user.id, namespace=req.namespace)
        db.add(conv)
        await db.flush()
        conv_id = conv.id

    # Save user message
    user_msg = Message(conversation_id=conv_id, role="user", content=req.query)
    db.add(user_msg)

    # Run AI pipeline (never crashes — returns error message on failure)
    try:
        result = await run_query(query=req.query, namespace=req.namespace, db=db)
    except Exception as e:
        logger.error("Pipeline error: %s", e)
        await db.rollback()
        result = {
            "answer": "I'm sorry, I couldn't process your request. The inference service may be warming up — please try again in a moment.",
            "namespace": req.namespace,
            "sources": [],
            "latency_ms": 0.0,
        }

    # Save assistant message
    try:
        assistant_msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=result["answer"],
            metadata_={"sources": result["sources"], "latency_ms": result["latency_ms"]},
        )
        db.add(assistant_msg)
        await db.commit()
    except Exception as e:
        logger.error("Failed to save message: %s", e)
        await db.rollback()

    return QueryResponse(
        answer=result["answer"],
        namespace=result["namespace"],
        conversation_id=str(conv_id),
        sources=result["sources"],
        latency_ms=result["latency_ms"],
    )
