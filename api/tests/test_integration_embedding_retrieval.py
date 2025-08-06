"""
Integration tests for embedding and retrieval services using a provider-based architecture.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from typing import List, Dict
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from app.services.document_processor import DocumentProcessor, DocumentChunk, DocumentType
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.providers.base import EmbeddingProvider, EmbeddingResponse, ProviderAPIError


@pytest_asyncio.fixture
async def mock_embedding_provider() -> MagicMock:
    """Create a mock EmbeddingProvider for integration tests."""
    mock_provider = MagicMock(spec=EmbeddingProvider)
    mock_provider.embed_texts = AsyncMock()
    mock_provider.embed_query = AsyncMock()
    return mock_provider


@pytest_asyncio.fixture
async def integration_setup(mock_embedding_provider):
    """Set up integration test environment with mocked provider."""
    with patch('app.services.retrieval_service.settings') as retrieval_settings:
        retrieval_settings.max_context_length = 2000
        retrieval_settings.similarity_threshold = 0.7

        # Initialize services with the mock provider
        doc_processor = DocumentProcessor()
        embedding_service = EmbeddingService(provider=mock_embedding_provider)
        retrieval_service = RetrievalService(embedding_service=embedding_service)
        
        yield {
            'doc_processor': doc_processor,
            'embedding_service': embedding_service,
            'retrieval_service': retrieval_service,
            'mock_provider': mock_embedding_provider
        }


@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Create sample documents for testing."""
    return {
        'ml_basics.txt': "Machine learning is a subset of AI.",
        'neural_networks.txt': "Neural networks are computing systems.",
        'real_estate.txt': "This property features 3 bedrooms."
    }


@pytest.mark.asyncio
class TestEmbeddingRetrievalIntegrationWithProvider:
    """Integration tests for the pipeline with a mocked provider."""

    async def test_complete_pipeline_end_to_end(self, integration_setup, sample_documents):
        """Test the complete pipeline from documents to semantic search."""
        services = integration_setup
        doc_processor = services['doc_processor']
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        mock_provider = services['mock_provider']
        
        tenant_id = "integration-test-tenant"
        
        # 1. Process documents into chunks
        all_chunks = []
        for filename, content in sample_documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            try:
                result = await doc_processor.process_file(Path(temp_file_path), tenant_id)
                assert result.success
                all_chunks.extend(result.chunks)
            finally:
                os.unlink(temp_file_path)
        
        assert len(all_chunks) >= 3

        # 2. Mock provider response for chunk embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        mock_embeddings = [[0.1 + i*0.1, 0.2] for i in range(len(chunk_texts))]
        mock_provider.embed_texts.return_value = EmbeddingResponse(
            embeddings=mock_embeddings, model="mock-model", usage={'total_tokens': 30}, processing_time_seconds=0.1
        )

        # 3. Generate embeddings via EmbeddingService
        embedding_result = await embedding_service.generate_embeddings(chunk_texts, tenant_id)
        assert embedding_result.success
        
        # 4. Create chunk embeddings mapping for RetrievalService
        chunk_embeddings_map = {
            f"{chunk.source_file}_{chunk.chunk_index}": emb
            for chunk, emb in zip(all_chunks, embedding_result.embeddings)
        }
        
        # 5. Test semantic search
        query = "What is machine learning?"
        mock_provider.embed_query.return_value = [0.1, 0.2]  # Similar to the first chunk
        
        retrieval_result = await retrieval_service.retrieve_context(
            query=query,
            available_chunks=all_chunks,
            chunk_embeddings=chunk_embeddings_map,
            tenant_id=tenant_id,
            top_k=1
        )
        
        # 6. Verify results
        assert retrieval_result.success
        assert len(retrieval_result.relevant_chunks) == 1
        assert "Machine learning" in retrieval_result.relevant_chunks[0].chunk.content
        mock_provider.embed_texts.assert_called_once_with(chunk_texts)
        mock_provider.embed_query.assert_called_once_with(query)

    async def test_error_propagation_from_provider(self, integration_setup):
        """Test that errors from the provider are handled gracefully."""
        services = integration_setup
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        mock_provider = services['mock_provider']
        
        tenant_id = "error-test"
        chunks = [DocumentChunk("content", 0, "file.txt", DocumentType.TXT, {}, 1)]
        chunk_embeddings_map = {"file.txt_0": [0.1, 0.2]}

        # Test failure in generate_embeddings
        mock_provider.embed_texts.side_effect = ProviderAPIError("Provider Down")
        embedding_result = await embedding_service.generate_embeddings(["text"], tenant_id)
        assert not embedding_result.success
        assert embedding_result.error_message == "Provider Down"

        # Reset mock side_effect
        mock_provider.embed_texts.side_effect = None

        # Test failure in retrieve_context
        mock_provider.embed_query.side_effect = ProviderAPIError("Query Embedding Failed")
        retrieval_result = await retrieval_service.retrieve_context(
            query="test",
            available_chunks=chunks,
            chunk_embeddings=chunk_embeddings_map,
            tenant_id=tenant_id
        )
        assert not retrieval_result.success
        assert "Failed to generate query embedding" in retrieval_result.error_message

    async def test_retrieval_with_no_matching_chunks(self, integration_setup):
        """Test retrieval when no chunks meet the similarity threshold."""
        services = integration_setup
        retrieval_service = services['retrieval_service']
        mock_provider = services['mock_provider']

        chunks = [DocumentChunk("content", 0, "f.txt", DocumentType.TXT, {}, 1)]
        chunk_embeddings_map = {"f.txt_0": [0.9, 0.9]} # High vector
        
        # Mock a query embedding that is very dissimilar
        mock_provider.embed_query.return_value = [0.1, 0.2]

        retrieval_result = await retrieval_service.retrieve_context(
            query="unrelated query",
            available_chunks=chunks,
            chunk_embeddings=chunk_embeddings_map,
            tenant_id="test",
            similarity_threshold=0.95
        )

        assert retrieval_result.success
        assert len(retrieval_result.relevant_chunks) == 0


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
