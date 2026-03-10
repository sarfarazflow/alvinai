from pydantic import BaseModel
from datetime import datetime


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: str
    title: str
    namespace: str
    created_at: datetime
    messages: list[MessageOut] = []

    model_config = {"from_attributes": True}
