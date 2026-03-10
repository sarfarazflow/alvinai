from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.conversation import Conversation

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/")
async def list_conversations(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    convs = result.scalars().all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "namespace": c.namespace,
            "created_at": c.created_at.isoformat(),
        }
        for c in convs
    ]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": str(conv.id),
        "title": conv.title,
        "namespace": conv.namespace,
        "created_at": conv.created_at.isoformat(),
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in conv.messages
        ],
    }
