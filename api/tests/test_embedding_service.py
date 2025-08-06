"""
Unit tests for the EmbeddingService class using a provider-based architecture.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock, MagicMock
from typing import List, Tuple, Dict, Any

import numpy as np

from app.services.embedding_service import (
    EmbeddingService, 
    EmbeddingResult, 
    SearchResult
)
from app.services.document_processor import DocumentChunk, DocumentType
from app.services.providers.base import EmbeddingProvider, EmbeddingResponse, ProviderAPIError


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create a mock EmbeddingProvider."""
    mock_provider = MagicMock(spec=EmbeddingProvider)
    mock_provider.embed_texts = AsyncMock()
    mock_provider.embed_query = AsyncMock()
    mock_provider.calculate_cost = MagicMock()
    mock_provider.get_model_info = MagicMock()
    mock_provider.provider_name = "mock_provider"
    return mock_provider


@pytest.fixture
def embedding_service(mock_embedding_provider: MagicMock) -> EmbeddingService:
    """Create an EmbeddingService instance with a mock provider."""
    with patch('app.services.embedding_service.settings') as mock_settings:
        mock_settings.similarity_threshold = 0.7
        # The service is initialized with the mock provider directly
        return EmbeddingService(provider=mock_embedding_provider)


@pytest.fixture
def sample_document_chunks() -> List[DocumentChunk]:
    """Create sample DocumentChunk objects for testing."""
    return [
        DocumentChunk(
            content="This is the first chunk.",
            chunk_index=0, source_file="doc1.pdf", document_type=DocumentType.PDF,
            metadata={"page": 1}, word_count=5
        ),
        DocumentChunk(
            content="This is the second chunk.",
            chunk_index=1, source_file="doc1.pdf", document_type=DocumentType.PDF,
            metadata={"page": 1}, word_count=5
        ),
        DocumentChunk(
            content="A completely different document.",
            chunk_index=0, source_file="doc2.pdf", document_type=DocumentType.PDF,
            metadata={"page": 1}, word_count=4
        )
    ]


class TestEmbeddingServiceWithProvider:
    """Test suite for EmbeddingService with a mocked provider."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test successful embedding generation via the provider."""
        texts = ["Hello world", "How are you?"]
        tenant_id = "test-tenant"
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        # Mock the provider's response
        mock_provider_response = EmbeddingResponse(
            embeddings=mock_embeddings,
            model="mock-model",
            usage={'total_tokens': 5},
            processing_time_seconds=0.5
        )
        mock_embedding_provider.embed_texts.return_value = mock_provider_response
        
        result = await embedding_service.generate_embeddings(texts, tenant_id)
        
        # Assert that the service called the provider correctly
        mock_embedding_provider.embed_texts.assert_called_once_with(texts)
        
        # Assert that the service correctly returns data from the provider's response
        assert result.success is True
        assert result.embeddings == mock_embeddings
        assert result.token_count == 5
        assert result.processing_time_seconds == 0.5
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_texts(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test that the service handles empty text lists before calling the provider."""
        result = await embedding_service.generate_embeddings([], "test-tenant")
        
        assert result.success is False
        assert result.error_message == "No texts provided"
        mock_embedding_provider.embed_texts.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_provider_error(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test how the service handles errors from the provider."""
        texts = ["Hello world"]
        error_message = "Provider API Error"
        mock_embedding_provider.embed_texts.side_effect = ProviderAPIError(error_message)
        
        result = await embedding_service.generate_embeddings(texts, "test-tenant")
        
        assert result.success is False
        assert result.embeddings == []
        assert result.error_message == error_message

    @pytest.mark.asyncio
    async def test_embed_query_success(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test successful query embedding via the provider."""
        query = "What is the weather?"
        mock_embedding = [0.1, 0.2, 0.3]
        mock_embedding_provider.embed_query.return_value = mock_embedding
        
        result = await embedding_service.embed_query(query)
        
        mock_embedding_provider.embed_query.assert_called_once_with(query)
        assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_embed_query_failure(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test query embedding failure from the provider."""
        query = "Test query"
        mock_embedding_provider.embed_query.side_effect = ProviderAPIError("API Error")
        
        result = await embedding_service.embed_query(query)
        
        assert result is None

    def test_find_similar_chunks_success(self, embedding_service: EmbeddingService, sample_document_chunks: List[DocumentChunk]):
        """Test successful similarity search logic, which remains in the service."""
        query_embedding = [1.0, 0.0, 0.0]
        chunk_embeddings = [
            (sample_document_chunks[0], [0.9, 0.1, 0.1]),  # High similarity
            (sample_document_chunks[1], [0.0, 1.0, 0.0]),  # Low similarity (orthogonal)
            (sample_document_chunks[2], [0.8, 0.2, 0.1])   # Medium similarity
        ]
    
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding, chunk_embeddings=chunk_embeddings, top_k=3, threshold=0.5
        )
    
        assert len(results) == 2  # The orthogonal chunk should be filtered out
        assert results[0].chunk == sample_document_chunks[0]
        assert results[0].rank == 1
        assert results[1].chunk == sample_document_chunks[2]
        assert results[1].rank == 2

    def test_find_similar_chunks_with_threshold(self, embedding_service: EmbeddingService, sample_document_chunks: List[DocumentChunk]):
        """Test similarity search with a higher threshold."""
        query_embedding = [1.0, 0.0, 0.0]
        chunk_embeddings = [
            (sample_document_chunks[0], [1.0, 0.0, 0.0]),  # Perfect similarity
            (sample_document_chunks[1], [0.8, 0.2, 0.0]),  # High similarity, but below threshold
        ]
    
        results = embedding_service.find_similar_chunks(
            query_embedding=query_embedding, chunk_embeddings=chunk_embeddings, top_k=2, threshold=0.99
        )
    
        assert len(results) == 1 # Only the perfect match should be returned
        assert results[0].chunk == sample_document_chunks[0]

    def test_find_similar_chunks_empty_input(self, embedding_service: EmbeddingService):
        """Test find_similar_chunks with empty inputs."""
        assert embedding_service.find_similar_chunks([0.1], []) == []
        assert embedding_service.find_similar_chunks(None, [Mock()]) == []

    def test_calculate_embedding_cost(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test that the service delegates cost calculation to the provider."""
        token_count = 1000
        expected_cost = 0.0002
        mock_embedding_provider.calculate_cost.return_value = expected_cost
        
        cost = embedding_service.calculate_embedding_cost(token_count)
        
        mock_embedding_provider.calculate_cost.assert_called_once_with(token_count)
        assert cost == expected_cost

    def test_get_provider_info(self, embedding_service: EmbeddingService, mock_embedding_provider: MagicMock):
        """Test that the service delegates provider info retrieval."""
        expected_info = {"provider": "mock_provider", "model": "mock-model"}
        mock_embedding_provider.get_model_info.return_value = expected_info
        
        info = embedding_service.get_provider_info()
        
        mock_embedding_provider.get_model_info.assert_called_once()
        assert info == expected_info

    def test_initialization_from_settings(self):
        """Test that the service can be initialized from settings using the factory."""
        with patch('app.services.embedding_service.create_provider_from_settings') as mock_create_provider:
            with patch('app.services.embedding_service.settings') as mock_settings:
                # Set up mock settings
                mock_settings.embedding_provider = "openai"
                mock_settings.embedding_provider_api_key = "test_key"
                mock_settings.embedding_provider_model = "test_model"
                mock_settings.embedding_provider_dimensions = 128
                mock_settings.embedding_batch_size = 50

                # Create the service, which should trigger the factory
                service = EmbeddingService(provider=None)

                # Verify the factory was called with the correct config
                expected_config = {
                    "api_key": "test_key",
                    "model": "test_model",
                    "dimensions": 128,
                    "max_batch_size": 50,
                }
                mock_create_provider.assert_called_once_with(
                    provider_name="openai",
                    **expected_config
                )
                assert service.provider == mock_create_provider.return_value


if __name__ == "__main__":
    pytest.main([__file__])
