"""
Integration tests for embedding and retrieval services.
Tests the complete pipeline from document chunks to semantic search.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from typing import List, Dict
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from app.services.document_processor import DocumentProcessor, DocumentChunk, DocumentType
from app.services.embedding_service import EmbeddingService, EmbeddingResult
from app.services.retrieval_service import RetrievalService, RetrievalResult


@pytest.mark.asyncio
class TestEmbeddingRetrievalIntegration:
    """Integration tests for the complete embedding and retrieval pipeline."""

    @pytest_asyncio.fixture
    async def integration_setup(self):
        """Set up integration test environment."""
        with patch('app.services.embedding_service.settings') as embed_settings, \
             patch('app.services.retrieval_service.settings') as retrieval_settings:
            
            # Configure embedding service settings
            embed_settings.openai_api_key = "test-api-key"
            embed_settings.openai_embedding_model = "text-embedding-3-small"
            embed_settings.openai_embedding_dimensions = 1536
            embed_settings.embedding_batch_size = 100
            embed_settings.similarity_threshold = 0.7
            
            # Configure retrieval service settings
            retrieval_settings.max_context_length = 2000
            retrieval_settings.similarity_threshold = 0.7
            
            # Create services
            doc_processor = DocumentProcessor()
            embedding_service = EmbeddingService()
            retrieval_service = RetrievalService()
            
            yield {
                'doc_processor': doc_processor,
                'embedding_service': embedding_service,
                'retrieval_service': retrieval_service
            }

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return {
            'ml_basics.txt': """
            Machine learning is a subset of artificial intelligence that focuses on algorithms 
            that can learn from data. It includes supervised learning, unsupervised learning, 
            and reinforcement learning approaches.
            """,
            'neural_networks.txt': """
            Neural networks are computing systems inspired by biological neural networks. 
            They consist of layers of interconnected nodes that process information through 
            weighted connections and activation functions.
            """,
            'real_estate.txt': """
            This property features 3 bedrooms, 2 bathrooms, and a spacious kitchen. 
            The house has a large backyard perfect for families and includes a two-car garage.
            Located in a quiet neighborhood with excellent schools nearby.
            """
        }

    @pytest.mark.asyncio
    async def test_complete_pipeline_end_to_end(self, integration_setup, sample_documents):
        """Test the complete pipeline from documents to semantic search."""
        services = integration_setup
        doc_processor = services['doc_processor']
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        
        tenant_id = "integration-test-tenant"
        
        # Step 1: Process documents into chunks
        all_chunks = []
        for filename, content in sample_documents.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Pass only the required arguments to process_file
                result = await doc_processor.process_file(
                    file_path=Path(temp_file_path),
                    tenant_id=tenant_id
                )
                assert result.success
                all_chunks.extend(result.chunks)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Verify we have chunks from all documents
        assert len(all_chunks) >= 3
        source_files = {chunk.source_file for chunk in all_chunks}
        assert len(source_files) == 3
        
        # Step 2: Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        
        # Mock OpenAI embedding API response
        mock_embeddings = [[0.1 + i*0.1, 0.2 + i*0.1, 0.3 + i*0.1, 0.4 + i*0.1] 
                          for i in range(len(chunk_texts))]
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(side_effect=lambda x: [1] * len(x.split()))
        
        embedding_result = await embedding_service.generate_embeddings(chunk_texts, tenant_id)
        
        assert embedding_result.success
        assert len(embedding_result.embeddings) == len(chunk_texts)
        
        # Step 3: Create chunk embeddings mapping
        chunk_embeddings = {}
        for chunk, embedding in zip(all_chunks, embedding_result.embeddings):
            chunk_key = f"{chunk.source_file}_{chunk.chunk_index}"
            chunk_embeddings[chunk_key] = embedding
        
        # Step 4: Test semantic search queries
        test_queries = [
            "What is machine learning?",
            "Tell me about neural networks",
            "Find properties with bedrooms"
        ]
        
        for query in test_queries:
            # Mock query embedding
            query_embedding = [0.5, 0.5, 0.5, 0.5]
            embedding_service.embed_query = AsyncMock(return_value=query_embedding)
            
            # Perform retrieval
            retrieval_result = await retrieval_service.retrieve_context(
                query=query,
                available_chunks=all_chunks,
                chunk_embeddings=chunk_embeddings,
                tenant_id=tenant_id,
                top_k=3
            )
            
            # Verify retrieval succeeded
            assert retrieval_result.success
            assert retrieval_result.query == query
            assert len(retrieval_result.relevant_chunks) <= 3
            assert retrieval_result.context_text != ""
            assert retrieval_result.total_chunks_searched == len(all_chunks)

    @pytest.mark.asyncio
    async def test_multi_tenant_data_isolation(self, integration_setup, sample_documents):
        """Test that different tenants have isolated embeddings and retrieval."""
        services = integration_setup
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        
        tenant1_id = "tenant-1"
        tenant2_id = "tenant-2"
        
        # Create different chunks for each tenant
        tenant1_chunks = [
            DocumentChunk(
                source_file="tenant1_doc.pdf",
                chunk_index=0,
                content="Tenant 1 confidential business data about project Alpha.",
                word_count=9,
                document_type=DocumentType.PDF,
                metadata={"tenant_id": tenant1_id, "page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
        
        tenant2_chunks = [
            DocumentChunk(
                source_file="tenant2_doc.pdf", 
                chunk_index=0,
                content="Tenant 2 private information about project Beta.",
                word_count=8,
                document_type=DocumentType.PDF,
                metadata={"tenant_id": tenant2_id, "page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
        
        # Generate separate embeddings for each tenant
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
        
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(return_value=[1, 2, 3])
        
        # Tenant 1 embeddings
        tenant1_result = await embedding_service.generate_embeddings(
            [chunk.content for chunk in tenant1_chunks], 
            tenant1_id
        )
        
        # Tenant 2 embeddings  
        tenant2_result = await embedding_service.generate_embeddings(
            [chunk.content for chunk in tenant2_chunks],
            tenant2_id
        )
        
        assert tenant1_result.success
        assert tenant2_result.success
        
        # Create isolated chunk embeddings
        tenant1_embeddings = {
            f"{tenant1_chunks[0].source_file}_{tenant1_chunks[0].chunk_index}": tenant1_result.embeddings[0]
        }
        tenant2_embeddings = {
            f"{tenant2_chunks[0].source_file}_{tenant2_chunks[0].chunk_index}": tenant2_result.embeddings[0]
        }
        
        # Test that tenant 1 query only searches tenant 1 data
        embedding_service.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        
        tenant1_retrieval = await retrieval_service.retrieve_context(
            query="project information",
            available_chunks=tenant1_chunks,  # Only tenant 1 chunks
            chunk_embeddings=tenant1_embeddings,  # Only tenant 1 embeddings
            tenant_id=tenant1_id
        )
        
        # Verify tenant isolation
        assert tenant1_retrieval.success
        assert len(tenant1_retrieval.relevant_chunks) <= len(tenant1_chunks)
        assert all("tenant1_doc.pdf" in result.chunk.source_file 
                  for result in tenant1_retrieval.relevant_chunks)

    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, integration_setup):
        """Test performance characteristics of the embedding and retrieval pipeline."""
        services = integration_setup
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        
        tenant_id = "performance-test"
        
        # Create a larger dataset for performance testing
        large_chunks = []
        for i in range(20):  # Create 20 chunks
            chunk = DocumentChunk(
                source_file=f"perf_doc_{i//5}.pdf",  # 4 docs with 5 chunks each
                chunk_index=i % 5,
                content=f"Performance test chunk {i} with various content about topic {i//5}. " * 10,  # ~100 words
                word_count=100,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
            large_chunks.append(chunk)
        
        # Mock embeddings generation with timing
        chunk_texts = [chunk.content for chunk in large_chunks]
        
        mock_embeddings = [[i*0.01 + j*0.001 for j in range(4)] for i in range(len(chunk_texts))]
        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]
        
        embedding_service.client.embeddings.create = AsyncMock(return_value=mock_response)
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(side_effect=lambda x: [1] * len(x.split()))
        
        # Test embedding generation performance
        import time
        start_time = time.time()
        embedding_result = await embedding_service.generate_embeddings(chunk_texts, tenant_id)
        embedding_time = time.time() - start_time
        
        assert embedding_result.success
        assert len(embedding_result.embeddings) == 20
        
        # Performance assertions
        assert embedding_time < 2.0  # Should complete within 2 seconds (mocked)
        assert embedding_result.processing_time_seconds > 0
        
        # Create embeddings mapping
        chunk_embeddings = {}
        for chunk, embedding in zip(large_chunks, embedding_result.embeddings):
            chunk_key = f"{chunk.source_file}_{chunk.chunk_index}"
            chunk_embeddings[chunk_key] = embedding
        
        # Test retrieval performance
        embedding_service.embed_query = AsyncMock(return_value=[0.1, 0.1, 0.1, 0.1])
        
        start_time = time.time()
        retrieval_result = await retrieval_service.retrieve_context(
            query="performance test query",
            available_chunks=large_chunks,
            chunk_embeddings=chunk_embeddings,
            tenant_id=tenant_id,
            top_k=5
        )
        retrieval_time = time.time() - start_time
        
        assert retrieval_result.success
        assert retrieval_time < 1.0  # Should complete within 1 second
        assert retrieval_result.retrieval_time_seconds > 0

    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, integration_setup):
        """Test error handling across the complete pipeline."""
        services = integration_setup
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        
        tenant_id = "error-test"
        
        # Test embedding service error propagation
        embedding_service.client.embeddings.create = AsyncMock(
            side_effect=Exception("OpenAI API Error")
        )
        
        embedding_result = await embedding_service.generate_embeddings(
            ["test content"], 
            tenant_id
        )
        
        assert embedding_result.success is False
        assert "OpenAI API Error" in embedding_result.error_message
        
        # Test retrieval service error handling when embedding fails
        chunks = [
            DocumentChunk(
                source_file="error_doc.pdf",
                chunk_index=0,
                content="Test content for error handling.",
                word_count=6,
                document_type=DocumentType.PDF,
                metadata={"page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
        
        # Retrieval should fail gracefully when query embedding fails
        retrieval_service.embedding_service.embed_query = AsyncMock(return_value=None)
        
        retrieval_result = await retrieval_service.retrieve_context(
            query="test query",
            available_chunks=chunks,
            chunk_embeddings={},
            tenant_id=tenant_id
        )
        
        assert retrieval_result.success is False
        assert "Failed to generate query embedding" in retrieval_result.error_message

    @pytest.mark.asyncio
    async def test_search_relevance_accuracy(self, integration_setup):
        """Test that semantic search returns relevant results."""
        services = integration_setup
        embedding_service = services['embedding_service']
        retrieval_service = services['retrieval_service']
        
        tenant_id = "relevance-test"
        
        # Create chunks with clearly different topics
        test_chunks = [
            DocumentChunk(
                source_file="ml_doc.pdf",
                chunk_index=0,
                content="Machine learning algorithms include decision trees, neural networks, and support vector machines.",
                word_count=13,
                document_type=DocumentType.PDF,
                metadata={"topic": "machine_learning", "page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="cooking_doc.pdf",
                chunk_index=0,
                content="Recipe for chocolate chip cookies: mix flour, sugar, eggs, and chocolate chips.",
                word_count=13,
                document_type=DocumentType.PDF,
                metadata={"topic": "cooking", "page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            ),
            DocumentChunk(
                source_file="sports_doc.pdf",
                chunk_index=0,
                content="Basketball is played with two teams of five players each on a rectangular court.",
                word_count=14,
                document_type=DocumentType.PDF,
                metadata={"topic": "sports", "page_number": 1, "created_at": "2024-01-01T00:00:00Z"}
            )
        ]
        
        # Create embeddings that reflect content similarity
        # ML query should be most similar to ML chunk
        ml_query_embedding = [0.9, 0.1, 0.1, 0.1]
        ml_chunk_embedding = [0.95, 0.05, 0.05, 0.05]  # Very similar to ML query
        cooking_chunk_embedding = [0.1, 0.9, 0.05, 0.05]  # Different from ML query
        sports_chunk_embedding = [0.1, 0.05, 0.9, 0.05]  # Different from ML query
        
        chunk_embeddings = {
            "ml_doc.pdf_0": ml_chunk_embedding,
            "cooking_doc.pdf_0": cooking_chunk_embedding,
            "sports_doc.pdf_0": sports_chunk_embedding
        }
        
        # Test ML query
        embedding_service.embed_query = AsyncMock(return_value=ml_query_embedding)
        
        retrieval_result = await retrieval_service.retrieve_context(
            query="What are machine learning algorithms?",
            available_chunks=test_chunks,
            chunk_embeddings=chunk_embeddings,
            tenant_id=tenant_id,
            top_k=3
        )
        
        assert retrieval_result.success
        assert len(retrieval_result.relevant_chunks) == 3
        
        # The ML chunk should be ranked first (highest similarity)
        top_result = retrieval_result.relevant_chunks[0]
        assert top_result.chunk.source_file == "ml_doc.pdf"
        assert top_result.similarity_score > 0.8  # Should be highly similar

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, integration_setup):
        """Test that batch processing works efficiently with rate limiting."""
        services = integration_setup
        embedding_service = services['embedding_service']
        
        # Set small batch size to test batching
        embedding_service.batch_size = 2
        tenant_id = "batch-test"
        
        # Create 5 texts (will require 3 batches: 2 + 2 + 1)
        texts = [f"Batch test text number {i}" for i in range(5)]
        
        call_count = 0
        batch_sizes = []
        
        def mock_create_embeddings(*args, **kwargs):
            nonlocal call_count, batch_sizes
            call_count += 1
            batch_size = len(kwargs['input'])
            batch_sizes.append(batch_size)
            
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1*i, 0.2*i]) for i in range(batch_size)]
            return mock_response
        
        embedding_service.client.embeddings.create = AsyncMock(side_effect=mock_create_embeddings)
        embedding_service.tokenizer = Mock()
        embedding_service.tokenizer.encode = Mock(return_value=[1, 2])
        
        with patch('asyncio.sleep') as mock_sleep:
            result = await embedding_service.generate_embeddings(texts, tenant_id)
        
        # Verify batching worked correctly
        assert result.success
        assert len(result.embeddings) == 5
        assert call_count == 3  # Should have made 3 API calls
        assert batch_sizes == [2, 2, 1]  # Batch sizes should be 2, 2, 1
        
        # Should have called sleep between batches (2 times for 3 batches)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.1)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
