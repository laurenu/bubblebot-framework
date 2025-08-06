"""
Unit tests for the EmbeddingService class.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Tuple

import numpy as np
from openai import AsyncOpenAI

from app.services.embedding_service import (
    EmbeddingService, 
    EmbeddingResult, 
    SearchResult
)
from app.services.document_processor import DocumentChunk, DocumentType


class TestEmbeddingService:
    """Test suite for EmbeddingService."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance for testing."""
        with patch('app.services.embedding_service.settings') as mock_settings:
            mock_settings.openai_api_key = "test-api-key"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 1536
            mock_settings.embedding_batch_size = 100
            mock_settings.similarity_threshold = 0.7
            
            service = EmbeddingService()
            return service
    
    @pytest.fixture
    def sample_document_chunks(self):
        """Create sample DocumentChunk objects for testing."""
        return [
            DocumentChunk(
                content="This is the first chunk of content.",
                chunk_index=0,
                source_file="doc1.pdf",
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"},
                word_count=8
            ),
            DocumentChunk(
                content="This is the second chunk with different content.",
                chunk_index=1,
                source_file="doc1.pdf",
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"},
                word_count=9
            ),
            DocumentChunk(
                content="Another document with completely different information.",
                chunk_index=0,
                source_file="doc2.pdf",
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"},
                word_count=8
            )
        ]

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service):
        """Test successful embedding generation."""
        texts = ["Hello world", "How are you?", "Test text"]
        tenant_id = "test-tenant"
        
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
            Mock(embedding=[0.7, 0.8, 0.9])
        ]
        
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        # Mock tokenizer
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(side_effect=lambda x: [1] * len(x.split()))
        
        result = await embedding_service.generate_embeddings(texts, tenant_id)
        
        assert result.success is True
        assert len(result.embeddings) == 3
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.token_count == 7  # Sum of word counts (2 + 3 + 2)
        assert result.processing_time_seconds > 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_texts(self, embedding_service):
        """Test embedding generation with empty text list."""
        result = await embedding_service.generate_embeddings([], "test-tenant")
        
        assert result.success is False
        assert result.embeddings == []
        assert result.token_count == 0
        assert result.error_message == "No texts provided"

    @pytest.mark.asyncio
    async def test_generate_embeddings_api_error(self, embedding_service):
        """Test embedding generation with API error."""
        texts = ["Hello world"]
        tenant_id = "test-tenant"
        
        embedding_service.client.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await embedding_service.generate_embeddings(texts, tenant_id)
        
        assert result.success is False
        assert result.embeddings == []
        assert result.error_message == "API Error"

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, embedding_service):
        """Test batch embedding generation."""
        texts = ["text1", "text2"]
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await embedding_service._generate_batch_embeddings(texts)
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        embedding_service.client.embeddings.create.assert_called_once_with(
            model=embedding_service.model,
            input=texts,
            dimensions=embedding_service.dimensions
        )

    def test_count_tokens_with_tokenizer(self, embedding_service):
        """Test token counting with tokenizer available."""
        texts = ["Hello world", "How are you?"]
        
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(side_effect=lambda x: [1] * len(x.split()))
        
        token_count = embedding_service._count_tokens(texts)
        
        assert token_count == 5  # 2 + 3 words
        assert embedding_service.tokenizer.encode.call_count == 2

    def test_count_tokens_without_tokenizer(self, embedding_service):
        """Test token counting without tokenizer (estimation)."""
        texts = ["Hello world", "Test"]
        embedding_service.tokenizer = None
        
        token_count = embedding_service._count_tokens(texts)
        
        # Rough estimation: total chars / 4
        expected = (len("Hello world") + len("Test")) // 4
        assert token_count == expected

    def test_find_similar_chunks_success(self, embedding_service, sample_document_chunks):
        """Test successful similarity search."""
        query_embedding = [0.5, 0.5, 0.5]
        chunk_embeddings = [
            (sample_document_chunks[0], [0.6, 0.4, 0.5]),  # similarity ≈ 0.92
            (sample_document_chunks[1], [0.1, 0.2, 0.3]),  # similarity ≈ 0.58
            (sample_document_chunks[2], [0.8, 0.9, 0.7])   # similarity ≈ 0.98
        ]
        
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            top_k=3,
            threshold=0.5
        )
        
        assert len(results) == 3
        # Results should be sorted by similarity (descending)
        assert results[0].chunk == sample_document_chunks[2]  # Highest similarity
        assert results[0].rank == 1
        assert results[1].chunk == sample_document_chunks[0]
        assert results[1].rank == 2
        assert all(r.similarity_score >= 0.5 for r in results)

    def test_find_similar_chunks_with_threshold(self, embedding_service, sample_document_chunks):
        """Test similarity search with threshold filtering."""
        query_embedding = [0.5, 0.5, 0.5]
        chunk_embeddings = [
            (sample_document_chunks[0], [0.6, 0.4, 0.5]),  # similarity ≈ 0.99
            (sample_document_chunks[1], [0.1, 0.2, 0.3]),  # similarity ≈ 0.93
            (sample_document_chunks[2], [0.0, 0.1, 0.0])   # similarity ≈ 0.58
        ]
        
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            top_k=3,
            threshold=0.6
        )
        
        # Only chunks with similarity >= 0.6 should be returned
        assert len(results) == 2
        assert results[0].chunk == sample_document_chunks[0]

    def test_find_similar_chunks_empty_input(self, embedding_service):
        """Test similarity search with empty chunk list."""
        query_embedding = [0.5, 0.5, 0.5]
        
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=[],
            top_k=5
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_embed_query_success(self, embedding_service):
        """Test successful query embedding."""
        query = "What is the weather like?"
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        
        result = await embedding_service.embed_query(query)
        
        assert result == [0.1, 0.2, 0.3]
        embedding_service.client.embeddings.create.assert_called_once_with(
            model=embedding_service.model,
            input=[query],
            dimensions=embedding_service.dimensions
        )

    @pytest.mark.asyncio
    async def test_embed_query_failure(self, embedding_service):
        """Test query embedding failure."""
        query = "Test query"
        
        embedding_service.client.embeddings.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await embedding_service.embed_query(query)
        
        assert result is None

    def test_calculate_embedding_cost(self, embedding_service):
        """Test embedding cost calculation."""
        token_count = 1000
        expected_cost = token_count * (0.02 / 1_000_000)  # $0.02 per 1M tokens
        
        cost = embedding_service.calculate_embedding_cost(token_count)
        
        assert cost == expected_cost

    def test_get_provider_info(self, embedding_service):
        """Test provider information retrieval."""
        info = embedding_service.get_provider_info()
        
        expected_info = {
            "provider": "OpenAI",
            "model": embedding_service.model,
            "dimensions": embedding_service.dimensions,
            "batch_size": embedding_service.batch_size,
            "max_tokens_per_request": 8192
        }
        
        assert info == expected_info

    @pytest.mark.asyncio
    async def test_batch_processing_with_delay(self, embedding_service):
        """Test that batch processing includes delays between batches."""
        # Create texts that will require multiple batches
        embedding_service.batch_size = 2
        texts = ["text1", "text2", "text3", "text4", "text5"]
        
        # Create a side_effect that returns the correct number of embeddings for each batch
        def mock_embeddings_create(*args, **kwargs):
            batch_size = len(kwargs['input'])
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2]) for _ in range(batch_size)]
            return mock_response
            
        embedding_service.client.embeddings.create = AsyncMock(side_effect=mock_embeddings_create)
        
        # Mock tokenizer
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(return_value=[1])
        
        with patch('asyncio.sleep') as mock_sleep:
            start_time = time.time()
            result = await embedding_service.generate_embeddings(texts, "tenant")
            end_time = time.time()
            
            # Should have called sleep between batches (3 batches total, 2 sleep calls)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_called_with(0.1)
            
        assert result.success is True
        assert len(result.embeddings) == 5  # Should match the number of input texts

    def test_search_result_dataclass(self, sample_document_chunks):
        """Test SearchResult dataclass functionality."""
        chunk = sample_document_chunks[0]
        similarity_score = 0.85
        rank = 1
        
        result = SearchResult(
            chunk=chunk,
            similarity_score=similarity_score,
            rank=rank
        )
        
        assert result.chunk == chunk
        assert result.similarity_score == similarity_score
        assert result.rank == rank

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass functionality."""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        token_count = 10
        processing_time = 1.5
        
        result = EmbeddingResult(
            success=True,
            embeddings=embeddings,
            token_count=token_count,
            processing_time_seconds=processing_time
        )
        
        assert result.success is True
        assert result.embeddings == embeddings
        assert result.token_count == token_count
        assert result.processing_time_seconds == processing_time
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_initialization_without_tokenizer(self):
        """Test service initialization when tokenizer fails to load."""
        with patch('app.services.embedding_service.settings') as mock_settings:
            mock_settings.openai_api_key = "test-api-key"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 1536
            mock_settings.embedding_batch_size = 100
            
            with patch('tiktoken.get_encoding', side_effect=Exception("Tokenizer error")):
                service = EmbeddingService()
                assert service.tokenizer is None

    def test_cosine_similarity_calculation(self, embedding_service):
        """Test that cosine similarity is calculated correctly."""
        # Known vectors with expected similarity
        query_embedding = [1.0, 0.0, 0.0]
        chunk_embeddings = [
            (Mock(), [1.0, 0.0, 0.0]),  # Same vector, similarity = 1.0
            (Mock(), [0.0, 1.0, 0.0]),  # Orthogonal, similarity = 0.0
            (Mock(), [-1.0, 0.0, 0.0])  # Opposite, similarity = -1.0
        ]
        
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding,
            chunk_embeddings=chunk_embeddings,
            top_k=3,
            threshold=-1.0  # Include all results
        )
        
        # Check that similarities are calculated correctly
        assert abs(results[0].similarity_score - 1.0) < 1e-6  # First result should be 1.0
        assert abs(results[1].similarity_score - 0.0) < 1e-6  # Second result should be 0.0
        assert abs(results[2].similarity_score - (-1.0)) < 1e-6  # Third result should be -1.0


if __name__ == "__main__":
    pytest.main([__file__])
