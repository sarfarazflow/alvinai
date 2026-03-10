import uuid
from sqlalchemy import String, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base, UUIDMixin, TimestampMixin


class Conversation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    namespace: Mapped[str] = mapped_column(String(100), default="customer_support")

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", lazy="selectin", order_by="Message.created_at")


class Message(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
