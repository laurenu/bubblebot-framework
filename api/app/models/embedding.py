"""
Database models for embeddings and vector storage.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Document(Base):
    """Document metadata storage."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    tenant_id = Column(String(100), nullable=False, index=True)
    document_type = Column(String(50), nullable=False)
    total_chunks = Column(Integer, nullable=False)
    total_words = Column(Integer, nullable=False)
    processing_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document")
    
    # Index for efficient tenant queries
    __table_args__ = (
        Index('idx_tenant_created', 'tenant_id', 'created_at'),
    )


class DocumentChunk(Base):
    """Individual document chunks with embeddings."""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    
    # Embedding data
    embedding_vector = Column(ARRAY(Float), nullable=True)  # 1536 dimensions for OpenAI
    embedding_model = Column(String(100), nullable=True)
    embedding_created_at = Column(DateTime, nullable=True)
    
    # Metadata
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    document = relationship("Document", back_populates="chunks")
    
    # Indexes for efficient similarity search
    __table_args__ = (
        Index('idx_tenant_embedding', 'tenant_id', 'embedding_created_at'),
        Index('idx_document_chunk', 'document_id', 'chunk_index'),
    )


class EmbeddingStats(Base):
    """Track embedding generation statistics."""
    __tablename__ = "embedding_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    total_chunks_processed = Column(Integer, nullable=False)
    total_api_calls = Column(Integer, nullable=False)
    total_tokens_used = Column(Integer, nullable=False)
    processing_time_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
