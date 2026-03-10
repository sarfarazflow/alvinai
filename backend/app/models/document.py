import uuid
from sqlalchemy import String, Text, Integer, Float, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base, UUIDMixin, TimestampMixin


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(500))
    source_path: Mapped[str] = mapped_column(String(1000))
    namespace: Mapped[str] = mapped_column(String(100), index=True)
    doc_type: Mapped[str] = mapped_column(String(50))  # pdf | docx | txt
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document", lazy="selectin")


class DocumentChunk(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(384), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    document = relationship("Document", back_populates="chunks")
