"""
Tests for the Bubblebot document processing pipeline.
"""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.document_processor import (
    DocumentProcessor, 
    DocumentType, 
    DocumentChunk, 
    ProcessingResult
)


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance with test settings."""
        return DocumentProcessor(
            chunk_size=500,
            chunk_overlap=100,
            max_file_size_mb=5
        )
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Real Estate FAQ
        
        Q: How long does it take to buy a house?
        A: The typical home buying process takes 30-45 days from contract to closing.
        
        Q: What is a pre-approval letter?
        A: A pre-approval letter is a document from a lender stating how much they're willing to lend you.
        
        Q: Do I need a real estate agent?
        A: While not required, a good agent can save you time and money in the process.
        """.strip()
    
    @pytest.fixture
    def temp_txt_file(self, sample_text):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_text)
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink()  # Clean up
    
    def test_document_type_detection(self, processor):
        """Test document type detection from file extensions."""
        test_cases = [
            ("document.pdf", DocumentType.PDF),
            ("file.docx", DocumentType.DOCX), 
            ("readme.txt", DocumentType.TXT),
            ("unknown.xyz", DocumentType.UNKNOWN),
            ("NO_EXTENSION", DocumentType.UNKNOWN)
        ]
        
        for filename, expected_type in test_cases:
            path = Path(filename)
            assert processor._get_document_type(path) == expected_type
    
    @pytest.mark.asyncio
    async def test_text_file_processing(self, processor, temp_txt_file, sample_text):
        """Test processing of a text file."""
        result = await processor.process_file(temp_txt_file, "tenant_123")
        
        assert result.success
        assert result.total_chunks > 0
        assert result.total_words > 0
        assert result.processing_time_seconds > 0
        
        # Check that chunks contain the original content
        all_content = " ".join(chunk.content for chunk in result.chunks)
        assert "Real Estate FAQ" in all_content
        assert "pre-approval letter" in all_content
    
    def test_chunk_creation_small_text(self, processor):
        """Test chunking for text smaller than chunk size."""
        small_text = "This is a small document."
        chunks = processor._create_chunks(
            small_text, 
            "test.txt", 
            DocumentType.TXT,
            "tenant_123"
        )
        
        assert len(chunks) == 1
        assert chunks[0].content == small_text
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["is_complete_document"] is True
        assert chunks[0].word_count == 5
    
    def test_chunk_creation_large_text(self, processor):
        """Test chunking for text larger than chunk size."""
        # Create text larger than chunk_size (500 chars)
        large_text = "This is a sentence. " * 50  # ~1000 characters
        
        chunks = processor._create_chunks(
            large_text,
            "large.txt",
            DocumentType.TXT, 
            "tenant_456"
        )
        
        assert len(chunks) > 1
        assert all(chunk.chunk_index == i for i, chunk in enumerate(chunks))
        assert all(chunk.source_file == "large.txt" for chunk in chunks)
        assert all(chunk.metadata["tenant_id"] == "tenant_456" for chunk in chunks)
    
    def test_chunk_overlap(self, processor):
        """Test that chunks have proper overlap."""
        # Text that will definitely need chunking
        sentences = ["This is sentence one. "] * 20
        text = "".join(sentences)
        
        chunks = processor._create_chunks(
            text,
            "overlap_test.txt", 
            DocumentType.TXT,
            "tenant_789"
        )
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            chunk1_end = chunks[0].content[-50:]  # Last 50 chars of first chunk
            chunk2_start = chunks[1].content[:50]  # First 50 chars of second chunk
            
            # There should be some common words due to overlap
            words1 = set(chunk1_end.split())
            words2 = set(chunk2_start.split())
            assert len(words1.intersection(words2)) > 0
    
    @pytest.mark.asyncio
    async def test_file_validation_nonexistent(self, processor):
        """Test validation of non-existent file."""
        fake_path = Path("nonexistent_file.txt")
        is_valid = await processor._validate_file(fake_path)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_file_validation_too_large(self, processor):
        """Test validation of oversized file."""
        # Create processor with very small max file size
        small_processor = DocumentProcessor(max_file_size_mb=0.001)  # 1KB limit
        
        # Create a file larger than the limit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("x" * 2000)  # 2KB file
            large_file = Path(f.name)
        
        try:
            is_valid = await small_processor._validate_file(large_file)
            assert not is_valid
        finally:
            large_file.unlink()
    
    @pytest.mark.asyncio
    async def test_txt_extraction(self, processor, temp_txt_file, sample_text):
        """Test text extraction from TXT file."""
        extracted = await processor._extract_txt_text(temp_txt_file)
        assert extracted.strip() == sample_text
    
    @pytest.mark.asyncio
    async def test_processing_failure_handling(self, processor):
        """Test handling of processing failures."""
        # Try to process a non-existent file
        fake_path = Path("does_not_exist.txt")
        result = await processor.process_file(fake_path, "tenant_error")
        
        assert not result.success
        assert result.error_message is not None
        assert result.total_chunks == 0
        assert result.total_words == 0
    
    def test_processing_stats_empty(self, processor):
        """Test stats generation with empty results."""
        stats = processor.get_processing_stats([])
        assert stats == {}
    
    def test_processing_stats_mixed_results(self, processor):
        """Test stats generation with mixed success/failure results."""
        results = [
            ProcessingResult(
                success=True,
                chunks=[Mock(word_count=100), Mock(word_count=150)],
                total_chunks=2,
                total_words=250,
                processing_time_seconds=1.5
            ),
            ProcessingResult(
                success=False,
                chunks=[],
                total_chunks=0,
                total_words=0,
                error_message="Test error",
                processing_time_seconds=0.5
            ),
            ProcessingResult(
                success=True,
                chunks=[Mock(word_count=75)],
                total_chunks=1, 
                total_words=75,
                processing_time_seconds=0.8
            )
        ]
        
        stats = processor.get_processing_stats(results)
        
        assert stats["total_documents"] == 3
        assert stats["successful_documents"] == 2
        assert stats["failed_documents"] == 1
        assert stats["total_chunks"] == 3
        assert stats["total_words"] == 325
        assert stats["success_rate"] == pytest.approx(66.67, rel=1e-2)
        assert stats["average_processing_time"] == pytest.approx(1.15, rel=1e-2)
    
    def test_document_chunk_dataclass(self):
        """Test DocumentChunk dataclass functionality."""
        chunk = DocumentChunk(
            content="This is test content with five words.",
            chunk_index=0,
            source_file="test.txt",
            document_type=DocumentType.TXT,
            metadata={"tenant_id": "test_tenant"},
            word_count=0  # Should be calculated automatically
        )
        
        # word_count should be calculated in __post_init__
        assert chunk.word_count == 7  # "This is test content with five words"
    
    @pytest.mark.asyncio
    async def test_empty_file_handling(self, processor):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            empty_file = Path(f.name)
        
        try:
            result = await processor.process_file(empty_file, "tenant_empty")
            assert not result.success
            assert "No text content found" in result.error_message
        finally:
            empty_file.unlink()
    
    @pytest.mark.asyncio
    async def test_whitespace_only_file(self, processor):
        """Test handling of files with only whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("   \n\n\t  \n  ")  # Only whitespace
            whitespace_file = Path(f.name)
        
        try:
            result = await processor.process_file(whitespace_file, "tenant_whitespace")
            assert not result.success
            assert "No text content found" in result.error_message
        finally:
            whitespace_file.unlink()


if __name__ == "__main__":
    # Run tests with: python -m pytest api/tests/test_document_processor.py -v
    pytest.main([__file__, "-v"])
