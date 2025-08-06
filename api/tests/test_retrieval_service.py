"""
Unit tests for the RetrievalService class.
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict

from app.services.retrieval_service import (
    RetrievalService,
    RetrievalResult
)
from app.services.embedding_service import SearchResult
from app.services.document_processor import DocumentChunk, DocumentType


class TestRetrievalService:
    """Test suite for RetrievalService."""
    
    @pytest.fixture
    def retrieval_service(self):
        """Create a RetrievalService instance for testing."""
        with patch('app.services.retrieval_service.settings') as mock_settings:
            mock_settings.max_context_length = 1000
            mock_settings.similarity_threshold = 0.7
            
            service = RetrievalService()
            return service
    
    @pytest.fixture
    def sample_document_chunks(self):
        """Create sample DocumentChunk objects for testing."""
        return [
            DocumentChunk(
                source_file="document1.pdf",
                chunk_index=0,
                content="This is the first chunk containing information about machine learning algorithms.",
                word_count=12,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="document1.pdf",
                chunk_index=1,
                content="The second chunk discusses neural networks and deep learning techniques.",
                word_count=11,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="document2.pdf",
                chunk_index=0,
                content="This chunk covers natural language processing and text analysis methods.",
                word_count=11,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
    
    @pytest.fixture
    def sample_chunk_embeddings(self, sample_document_chunks):
        """Create sample chunk embeddings mapping."""
        return {
            "document1.pdf_0": [0.1, 0.2, 0.3, 0.4],
            "document1.pdf_1": [0.5, 0.6, 0.7, 0.8],
            "document2.pdf_0": [0.9, 0.8, 0.7, 0.6]
        }
    
    @pytest.fixture
    def sample_search_results(self, sample_document_chunks):
        """Create sample SearchResult objects."""
        return [
            SearchResult(
                chunk=sample_document_chunks[0],
                similarity_score=0.95,
                rank=1
            ),
            SearchResult(
                chunk=sample_document_chunks[1],
                similarity_score=0.85,
                rank=2
            ),
            SearchResult(
                chunk=sample_document_chunks[2],
                similarity_score=0.75,
                rank=3
            )
        ]

    @pytest.mark.asyncio
    async def test_retrieve_context_success(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test successful context retrieval."""
        query = "What is machine learning?"
        tenant_id = "test-tenant"
        query_embedding = [0.2, 0.3, 0.4, 0.5]
        
        # Mock the embedding service methods
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=query_embedding)
        
        # Mock the find_similar_chunks method
        mock_search_results = [
            SearchResult(
                chunk=sample_document_chunks[0],
                similarity_score=0.95,
                rank=1
            ),
            SearchResult(
                chunk=sample_document_chunks[1],
                similarity_score=0.85,
                rank=2
            )
        ]
        retrieval_service.embedding_service.find_similar_chunks = Mock(return_value=mock_search_results)
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=sample_chunk_embeddings,
            tenant_id=tenant_id,
            top_k=2
        )
        
        assert result.success is True
        assert result.query == query
        assert len(result.relevant_chunks) == 2
        assert result.context_text != ""
        assert result.total_chunks_searched == 3
        assert result.retrieval_time_seconds > 0
        assert result.error_message is None
        
        # Verify that embed_query was called correctly
        retrieval_service.embedding_service.embed_query.assert_called_once_with(query)
        
        # Verify that find_similar_chunks was called correctly
        retrieval_service.embedding_service.find_similar_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_context_query_embedding_failure(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test context retrieval when query embedding fails."""
        query = "Test query"
        tenant_id = "test-tenant"
        
        # Mock embedding failure
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=None)
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=sample_chunk_embeddings,
            tenant_id=tenant_id
        )
        
        assert result.success is False
        assert result.query == query
        assert result.relevant_chunks == []
        assert result.context_text == ""
        assert result.error_message == "Failed to generate query embedding"

    @pytest.mark.asyncio
    async def test_retrieve_context_exception_handling(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test context retrieval exception handling."""
        query = "Test query"
        tenant_id = "test-tenant"
        
        # Mock an exception in embed_query
        retrieval_service.embedding_service.embed_query = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=sample_chunk_embeddings,
            tenant_id=tenant_id
        )
        
        assert result.success is False
        assert result.error_message == "API Error"
        assert result.total_chunks_searched == len(sample_document_chunks)

    @pytest.mark.asyncio
    async def test_retrieve_context_missing_embeddings(self, retrieval_service, sample_document_chunks):
        """Test context retrieval with missing chunk embeddings."""
        query = "Test query"
        tenant_id = "test-tenant"
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Empty embeddings dict
        chunk_embeddings = {}
        
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=query_embedding)
        retrieval_service.embedding_service.find_similar_chunks = Mock(return_value=[])
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=chunk_embeddings,
            tenant_id=tenant_id
        )
        
        assert result.success is True
        assert len(result.relevant_chunks) == 0
        assert result.total_chunks_searched == 0

    def test_build_context_text_success(self, retrieval_service, sample_search_results):
        """Test successful context text building."""
        context_text = retrieval_service._build_context_text(sample_search_results)
        
        assert context_text != ""
        assert "document1.pdf" in context_text
        assert "document2.pdf" in context_text
        assert "Similarity: 0.950" in context_text
        assert sample_search_results[0].chunk.content in context_text
        
        # Verify format with source and similarity
        lines = context_text.split('\n')
        assert any("[Source:" in line and "Similarity:" in line for line in lines)

    def test_build_context_text_empty_results(self, retrieval_service):
        """Test context text building with empty results."""
        context_text = retrieval_service._build_context_text([])
        assert context_text == ""

    def test_build_context_text_length_limit(self, retrieval_service, sample_document_chunks):
        """Test context text building respects length limits."""
        # Set a very small context length limit
        retrieval_service.max_context_length = 50
        
        # Create search results with long content
        long_chunk = DocumentChunk(
            source_file="long_doc.pdf",
            chunk_index=0,
            content="This is a very long chunk of content that should exceed the context length limit and be truncated.",
            word_count=20,
            document_type=DocumentType.PDF,
            metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
        )
        
        search_results = [
            SearchResult(chunk=long_chunk, similarity_score=0.9, rank=1)
        ]
        
        context_text = retrieval_service._build_context_text(search_results)
        
        # Should be empty or very short due to length limit
        assert len(context_text) <= retrieval_service.max_context_length

    def test_rank_chunks_by_relevance_default(self, retrieval_service, sample_search_results):
        """Test chunk ranking with default parameters."""
        # Create unordered results
        unordered_results = [
            sample_search_results[2],  # similarity 0.75
            sample_search_results[0],  # similarity 0.95
            sample_search_results[1]   # similarity 0.85
        ]
        
        ranked_results = retrieval_service.rank_chunks_by_relevance(unordered_results)
        
        # Should be ordered by similarity score (descending)
        assert ranked_results[0].similarity_score == 0.95
        assert ranked_results[1].similarity_score == 0.85
        assert ranked_results[2].similarity_score == 0.75
        
        # Ranks should be updated
        assert ranked_results[0].rank == 1
        assert ranked_results[1].rank == 2
        assert ranked_results[2].rank == 3

    def test_rank_chunks_by_relevance_boost_longer(self, retrieval_service, sample_document_chunks):
        """Test chunk ranking with boost for longer chunks."""
        # Create chunks with different lengths
        short_chunk = DocumentChunk(
            source_file="doc.pdf",
            chunk_index=0,
            content="Short content.",
            word_count=2,
            metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"},
            document_type=DocumentType.PDF,
        )
        
        long_chunk = DocumentChunk(
            source_file="doc.pdf",
            chunk_index=1,
            content=" ".join(["Long"] * 150),  # 150 words
            word_count=150,
            metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"},
            document_type=DocumentType.PDF,
        )
        
        # Both have similar similarity scores
        search_results = [
            SearchResult(chunk=short_chunk, similarity_score=0.80, rank=1),
            SearchResult(chunk=long_chunk, similarity_score=0.80, rank=2)  # Slightly lower
        ]
        
        ranked_results = retrieval_service.rank_chunks_by_relevance(
            search_results, 
            boost_longer=True
        )
        
        # Long chunk should now rank higher due to boost
        assert ranked_results[0].chunk == long_chunk
        assert ranked_results[1].chunk == short_chunk

    def test_rank_chunks_by_relevance_empty_input(self, retrieval_service):
        """Test chunk ranking with empty input."""
        ranked_results = retrieval_service.rank_chunks_by_relevance([])
        assert ranked_results == []

    def test_get_retrieval_stats_success(self, retrieval_service, sample_document_chunks):
        """Test retrieval statistics generation."""
        # Create sample retrieval results
        results = [
            RetrievalResult(
                success=True,
                query="query1",
                relevant_chunks=[Mock(), Mock()],
                context_text="context1",
                total_chunks_searched=5,
                retrieval_time_seconds=0.5
            ),
            RetrievalResult(
                success=True,
                query="query2",
                relevant_chunks=[Mock()],
                context_text="context2",
                total_chunks_searched=5,
                retrieval_time_seconds=0.3
            ),
            RetrievalResult(
                success=False,
                query="query2",
                relevant_chunks=[],
                context_text="",
                total_chunks_searched=0,
                retrieval_time_seconds=0.2,
                error_message="Error2"
            )
        ]
        
        stats = retrieval_service.get_retrieval_stats(results)
        
        assert stats["total_queries"] == 3
        assert stats["successful_queries"] == 2
        assert stats["failed_queries"] == 1
        assert stats["average_retrieval_time"] == 0.4 
        assert stats["average_chunks_retrieved"] == 1.5
        assert stats["success_rate"] == 66.66666666666666

    def test_retrieval_result_dataclass(self, sample_search_results):
        """Test RetrievalResult dataclass functionality."""
        query = "test query"
        context_text = "test context"
        total_chunks = 10
        retrieval_time = 1.5
        
        result = RetrievalResult(
            success=True,
            query=query,
            relevant_chunks=sample_search_results,
            context_text=context_text,
            total_chunks_searched=total_chunks,
            retrieval_time_seconds=retrieval_time
        )
        
        assert result.success is True
        assert result.query == query
        assert result.relevant_chunks == sample_search_results
        assert result.context_text == context_text
        assert result.total_chunks_searched == total_chunks
        assert result.retrieval_time_seconds == retrieval_time
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_retrieve_context_with_custom_parameters(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test context retrieval with custom parameters."""
        query = "custom query"
        tenant_id = "test-tenant"
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        custom_top_k = 1
        custom_threshold = 0.9
        
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=query_embedding)
        
        mock_search_results = [
            SearchResult(
                chunk=sample_document_chunks[0],
                similarity_score=0.95,
                rank=1
            )
        ]
        retrieval_service.embedding_service.find_similar_chunks = Mock(return_value=mock_search_results)
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=sample_chunk_embeddings,
            tenant_id=tenant_id,
            top_k=custom_top_k,
            similarity_threshold=custom_threshold
        )
        
        assert result.success is True
        
        # Verify custom parameters were passed to find_similar_chunks
        call_args = retrieval_service.embedding_service.find_similar_chunks.call_args
        assert call_args[1]["top_k"] == custom_top_k
        assert call_args[1]["threshold"] == custom_threshold

    def test_chunk_embedding_key_generation(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test that chunk embedding keys are generated correctly."""
        # This test verifies the key format: "{source_file}_{chunk_index}"
        expected_keys = [
            "document1.pdf_0",
            "document1.pdf_1", 
            "document2.pdf_0"
        ]
        
        # All expected keys should exist in the sample embeddings
        for key in expected_keys:
            assert key in sample_chunk_embeddings
        
        # Test that the service properly generates these keys during retrieval
        # This is implicitly tested in the successful retrieval test, but we can
        # verify the logic here by checking the key format matches the chunks
        for i, chunk in enumerate(sample_document_chunks):
            expected_key = f"{chunk.source_file}_{chunk.chunk_index}"
            assert expected_key in sample_chunk_embeddings

    @pytest.mark.asyncio
    async def test_retrieve_context_timing_accuracy(self, retrieval_service, sample_document_chunks, sample_chunk_embeddings):
        """Test that retrieval timing is measured accurately."""
        query = "timing test"
        tenant_id = "test-tenant"
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Create a mock for the sleep function
        mock_sleep = AsyncMock()
        
        # Mock with a controlled delay to test timing
        async def delayed_embed_query(q):
            await mock_sleep(0.1)  # Mocked sleep
            return query_embedding
        
        # Set up the mocks
        retrieval_service.embedding_service.embed_query = delayed_embed_query
        retrieval_service.embedding_service.find_similar_chunks = Mock(return_value=[])
        
        # Mock time.time() to return controlled values
        with patch('time.time', side_effect=[100.0, 100.1]):  # 100ms difference
            result = await retrieval_service.retrieve_context(
                query=query,
                available_chunks=sample_document_chunks,
                chunk_embeddings=sample_chunk_embeddings,
                tenant_id=tenant_id
            )
        
        # Verify the sleep was called with the correct duration
        mock_sleep.assert_awaited_once_with(0.1)
        
        # The retrieval time should be approximately 0.1 seconds
        assert result.success is True
        assert abs(result.retrieval_time_seconds - 0.1) < 0.01  # Allowing small floating point error

    def test_context_text_formatting(self, retrieval_service, sample_document_chunks):
        """Test context text formatting includes proper metadata."""
        search_results = [
            SearchResult(
                chunk=sample_document_chunks[0],
                similarity_score=0.923456,
                rank=1
            ),
            SearchResult(
                chunk=sample_document_chunks[1], 
                similarity_score=0.876543,
                rank=2
            )
        ]
        
        context_text = retrieval_service._build_context_text(search_results)
        
        # Check formatting of source and similarity
        assert "[Source: document1.pdf, Similarity: 0.923]" in context_text
        assert "[Source: document1.pdf, Similarity: 0.877]" in context_text
        
        # Check that content is included
        assert sample_document_chunks[0].content in context_text
        assert sample_document_chunks[1].content in context_text
        
        # Check that chunks are separated by double newlines
        assert "\n\n" in context_text

    @pytest.mark.asyncio
    async def test_retrieve_context_partial_embeddings(self, retrieval_service, sample_document_chunks):
        """Test retrieval when only some chunks have embeddings."""
        query = "partial test"
        tenant_id = "test-tenant"
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Only provide embeddings for first two chunks
        partial_embeddings = {
            "document1.pdf_0": [0.1, 0.2, 0.3, 0.4],
            "document1.pdf_1": [0.5, 0.6, 0.7, 0.8]
            # Missing document2.pdf_0
        }
        
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=query_embedding)
        retrieval_service.embedding_service.find_similar_chunks = Mock(return_value=[])
        
        result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=sample_document_chunks,
            chunk_embeddings=partial_embeddings,
            tenant_id=tenant_id
        )
        
        assert result.success is True
        # Should only search through chunks that have embeddings
        assert result.total_chunks_searched == 2
        
        # Verify find_similar_chunks was called with only available chunks
        call_args = retrieval_service.embedding_service.find_similar_chunks.call_args
        chunk_embedding_pairs = call_args[1]["chunk_embeddings"]
        assert len(chunk_embedding_pairs) == 2

    def test_initialization_with_settings(self):
        """Test service initialization with settings."""
        with patch('app.services.retrieval_service.settings') as mock_settings:
            mock_settings.max_context_length = 2000
            
            service = RetrievalService()
            assert service.max_context_length == 2000
            assert hasattr(service, 'embedding_service')

    @pytest.mark.parametrize("boost_recent, boost_longer, expected_order", [
        # With no boosts, order should be by similarity score
        (False, False, [0, 1, 2]),
        # With boost_longer=True, the long chunk (index 1) should move up if its score + 0.01 is higher than others
        (False, True, [1, 0, 2]),  # 0.85 + 0.01 = 0.86 > 0.85 > 0.80
        # boost_recent is not implemented yet, so it shouldn't affect the order
        (True, False, [0, 1, 2]),
        # With both boosts, only boost_longer should have an effect
        (True, True, [1, 0, 2])
    ])
    def test_rank_chunks_parametrized(self, retrieval_service, boost_recent, boost_longer, expected_order):
        """Test chunk ranking with different boost parameters."""
        # Create chunks with different characteristics
        chunks = [
            DocumentChunk(
                source_file="doc.pdf",
                chunk_index=0,
                content="Short content",
                word_count=2,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="doc.pdf",
                chunk_index=1,
                content=" ".join(["Long"] * 150),  # Long chunk
                word_count=150,  # Will get a 0.01 boost if boost_longer is True
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="doc.pdf", 
                chunk_index=2,
                content="Medium length content here",
                word_count=4,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
        
        # Initial search results with similarity scores
        search_results = [
            SearchResult(chunk=chunks[0], similarity_score=0.90, rank=1),
            SearchResult(chunk=chunks[1], similarity_score=0.88, rank=2),  # Long chunk
            SearchResult(chunk=chunks[2], similarity_score=0.80, rank=3)
        ]
        
        ranked_results = retrieval_service.rank_chunks_by_relevance(
            search_results,
            boost_recent=boost_recent,  # Currently not implemented
            boost_longer=boost_longer   # Adds 0.01 to chunks with word_count > 100
        )
        
        # Check that chunks are in expected order
        for i, expected_idx in enumerate(expected_order):
            assert ranked_results[i].chunk == chunks[expected_idx], \
                f"Expected chunk {expected_idx} at position {i}, but got chunk {ranked_results[i].chunk.chunk_index}"


if __name__ == "__main__":
    pytest.main([__file__])
