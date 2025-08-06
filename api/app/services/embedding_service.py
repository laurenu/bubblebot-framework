"""
Embedding service for generating and managing vector embeddings.
This service uses a provider-based architecture to support multiple embedding APIs.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.document_processor import DocumentChunk
from app.services.providers.base import EmbeddingProvider
from app.services.providers.factory import create_provider_from_settings

# Import provider modules to ensure they are registered with the factory
from app.services.providers import openai_provider, gemini_provider

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation operation."""
    success: bool
    embeddings: List[List[float]]
    token_count: int
    processing_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class SearchResult:
    """Result of similarity search operation."""
    chunk: DocumentChunk
    similarity_score: float
    rank: int


class EmbeddingService:
    """
    Service for generating and managing vector embeddings using a provider model.
    """

    def __init__(self, provider: EmbeddingProvider = None):
        """
        Initialize the embedding service.

        Args:
            provider: An optional pre-configured embedding provider. If not provided,
                      one will be created from application settings.
        """
        if provider:
            self.provider = provider
        else:
            # Create provider from settings
            provider_config = {
                "api_key": settings.embedding_provider_api_key,
                "model": settings.embedding_provider_model,
                "dimensions": settings.embedding_provider_dimensions,
                "max_batch_size": settings.embedding_batch_size,
            }
            self.provider = create_provider_from_settings(
                provider_name=settings.embedding_provider,
                **provider_config,
            )

        logger.info(f"EmbeddingService initialized with provider: {self.provider.provider_name}")

    async def generate_embeddings(
        self, texts: List[str], tenant_id: str
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using the configured provider.
        """
        if not texts:
            return EmbeddingResult(
                success=False,
                embeddings=[],
                token_count=0,
                processing_time_seconds=0,
                error_message="No texts provided",
            )

        try:
            response = await self.provider.embed_texts(texts)

            logger.info(
                f"Generated {len(response.embeddings)} embeddings for tenant {tenant_id} "
                f"using {self.provider.provider_name}: "
                f"{response.usage.get('total_tokens', 0)} tokens in {response.processing_time_seconds:.2f}s"
            )

            return EmbeddingResult(
                success=True,
                embeddings=response.embeddings,
                token_count=response.usage.get("total_tokens", 0),
                processing_time_seconds=response.processing_time_seconds,
            )

        except Exception as e:
            logger.error(f"Embedding generation failed with {self.provider.provider_name}: {str(e)}")
            return EmbeddingResult(
                success=False,
                embeddings=[],
                token_count=0,
                processing_time_seconds=0,  # This might not be accurate if it fails mid-way
                error_message=str(e),
            )

    async def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a single query string using the configured provider.
        """
        try:
            return await self.provider.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed with {self.provider.provider_name}: {str(e)}")
            return None

    def find_similar_chunks(
        self, 
        query_embedding: List[float], 
        chunk_embeddings: List[Tuple[DocumentChunk, List[float]]],
        top_k: int = 5,
        threshold: float = None
    ) -> List[SearchResult]:
        """
        Find similar chunks using cosine similarity.
        
        Args:
            query_embedding: Embedding vector for the query
            chunk_embeddings: List of (chunk, embedding) tuples
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects ranked by similarity
        """
        if not chunk_embeddings or query_embedding is None:
            return []
        
        threshold = threshold or settings.similarity_threshold
        
        # Extract embeddings and chunks
        chunks, embeddings = zip(*chunk_embeddings)
        
        # Calculate similarities
        query_vec = np.array(query_embedding).reshape(1, -1)
        embedding_matrix = np.array(embeddings)
        
        similarities = cosine_similarity(query_vec, embedding_matrix)[0]
        
        # Create results with similarity scores
        results = []
        for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
            if similarity >= threshold:
                results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=float(similarity),
                    rank=i
                ))
        
        # Sort by similarity (descending) and take top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]

    def calculate_embedding_cost(self, token_count: int) -> float:
        """
        Calculate approximate cost for embedding generation using the provider.
        """
        return self.provider.calculate_cost(token_count)

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current embedding provider."""
        return self.provider.get_model_info()
