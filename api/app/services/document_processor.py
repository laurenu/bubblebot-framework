"""
Bubblebot Framework - Document Processing Pipeline

This module handles document ingestion, processing, and text extraction
for the AI chatbot knowledge base.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from pypdf import PdfReader
from docx import Document as DocxDocument
import aiofiles

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx" 
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """Represents a processed chunk of document text."""
    content: str
    chunk_index: int
    source_file: str
    document_type: DocumentType
    metadata: Dict[str, Any]
    word_count: int
    
    def __post_init__(self):
        """Calculate word count if not provided."""
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass 
class ProcessingResult:
    """Result of document processing operation."""
    success: bool
    chunks: List[DocumentChunk]
    total_chunks: int
    total_words: int
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class DocumentProcessor:
    """
    Handles document processing for the Bubblebot chatbot framework.
    
    Features:
    - Multi-format support (PDF, DOCX, TXT)
    - Intelligent text chunking
    - Metadata extraction
    - Async processing for performance
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_file_size_mb: int = 10
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Target size for text chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            max_file_size_mb: Maximum allowed file size in MB
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
    async def process_file(
        self, 
        file_path: Path, 
        tenant_id: str
    ) -> ProcessingResult:
        """
        Process a document file and extract text chunks.
        
        Args:
            file_path: Path to the document file
            tenant_id: ID of the tenant uploading the document
            
        Returns:
            ProcessingResult with chunks and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate file
            if not await self._validate_file(file_path):
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    total_words=0,
                    error_message="File validation failed"
                )
            
            # Determine document type
            doc_type = self._get_document_type(file_path)
            
            # Extract text based on type
            raw_text = await self._extract_text(file_path, doc_type)
            
            if not raw_text.strip():
                return ProcessingResult(
                    success=False,
                    chunks=[],
                    total_chunks=0,
                    total_words=0,
                    error_message="No text content found in document"
                )
            
            # Create chunks
            chunks = self._create_chunks(
                raw_text, 
                file_path.name, 
                doc_type,
                tenant_id
            )
            
            processing_time = time.time() - start_time
            total_words = sum(chunk.word_count for chunk in chunks)
            
            logger.info(
                f"Processed {file_path.name}: {len(chunks)} chunks, "
                f"{total_words} words in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                success=True,
                chunks=chunks,
                total_chunks=len(chunks),
                total_words=total_words,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                chunks=[],
                total_chunks=0,
                total_words=0,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time
            )
    
    async def _validate_file(self, file_path: Path) -> bool:
        """Validate file size and accessibility."""
        try:
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
                
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.error(f"File too large: {file_size} bytes")
                return False
                
            return True
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def _get_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension."""
        suffix = file_path.suffix.lower()
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.txt': DocumentType.TXT,
        }
        return type_mapping.get(suffix, DocumentType.UNKNOWN)
    
    async def _extract_text(self, file_path: Path, doc_type: DocumentType) -> str:
        """Extract text content based on document type."""
        if doc_type == DocumentType.PDF:
            return await self._extract_pdf_text(file_path)
        elif doc_type == DocumentType.DOCX:
            return await self._extract_docx_text(file_path)
        elif doc_type == DocumentType.TXT:
            return await self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        def _extract():
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
                
                return "\n\n".join(text_parts)
        
        # Run in thread pool to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        def _extract():
            doc = DocxDocument(file_path)
            paragraphs = []
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    paragraphs.append(text)
            
            return "\n\n".join(paragraphs)
        
        return await asyncio.get_event_loop().run_in_executor(None, _extract)
    
    async def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from plain text file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()
    
    def _create_chunks(
        self, 
        text: str, 
        source_file: str, 
        doc_type: DocumentType,
        tenant_id: str
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks for better context retrieval."""
        if len(text) <= self.chunk_size:
            # Text is small enough for a single chunk
            return [DocumentChunk(
                content=text,
                chunk_index=0,
                source_file=source_file,
                document_type=doc_type,
                metadata={
                    "tenant_id": tenant_id,
                    "total_chunks": 1,
                    "is_complete_document": True
                },
                word_count=len(text.split())
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary near the end
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                best_break = end
                
                for i in range(max(0, end - 100), end):
                    if any(text[i:i+2] == ending for ending in sentence_endings):
                        best_break = i + 2
                        break
                
                end = best_break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    source_file=source_file,
                    document_type=doc_type,
                    metadata={
                        "tenant_id": tenant_id,
                        "total_chunks": 0,  # Will be updated after processing
                        "char_start": start,
                        "char_end": end
                    },
                    word_count=len(chunk_text.split())
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Generate processing statistics for multiple documents."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        return {
            "total_documents": len(results),
            "successful_documents": len(successful_results),
            "failed_documents": len(results) - len(successful_results),
            "total_chunks": sum(r.total_chunks for r in successful_results),
            "total_words": sum(r.total_words for r in successful_results),
            "average_processing_time": sum(r.processing_time_seconds for r in successful_results) / len(successful_results) if successful_results else 0,
            "success_rate": len(successful_results) / len(results) * 100
        }
