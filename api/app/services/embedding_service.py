"""
Embedding service for generating and managing vector embeddings.
"""

import asyncio
import logging
import time
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.services.document_processor import DocumentChunk

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
    Service for generating and managing vector embeddings using OpenAI API.
    
    Features:
    - Batch processing for efficiency
    - Token counting and cost optimization
    - Similarity search with configurable thresholds
    - Provider abstraction for future flexibility
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimensions = settings.openai_embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        
        # Initialize tokenizer for cost calculation
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {e}")
            self.tokenizer = None
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        tenant_id: str
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            tenant_id: Tenant identifier for logging
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        
        try:
            if not texts:
                return EmbeddingResult(
                    success=False,
                    embeddings=[],
                    token_count=0,
                    processing_time_seconds=0,
                    error_message="No texts provided"
                )
            
            # Count tokens for cost tracking
            total_tokens = self._count_tokens(texts)
            
            # Process in batches to avoid rate limits
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = await self._generate_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Generated {len(all_embeddings)} embeddings for tenant {tenant_id}: "
                f"{total_tokens} tokens in {processing_time:.2f}s"
            )
            
            return EmbeddingResult(
                success=True,
                embeddings=all_embeddings,
                token_count=total_tokens,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return EmbeddingResult(
                success=False,
                embeddings=[],
                token_count=0,
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions
        )
        
        return [item.embedding for item in response.data]
    
    def _count_tokens(self, texts: List[str]) -> int:
        """Count total tokens in texts for cost estimation."""
        if not self.tokenizer:
            # Rough estimation: ~4 chars per token
            return sum(len(text) for text in texts) // 4
        
        return sum(len(self.tokenizer.encode(text)) for text in texts)
    
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
        if not chunk_embeddings:
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
    
    async def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=[query],
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            return None
    
    def calculate_embedding_cost(self, token_count: int) -> float:
        """
        Calculate approximate cost for embedding generation.
        
        Args:
            token_count: Number of tokens processed
            
        Returns:
            Estimated cost in USD
        """
        # OpenAI text-embedding-3-small pricing: $0.02 per 1M tokens
        cost_per_token = 0.02 / 1_000_000
        return token_count * cost_per_token
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current embedding provider."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "max_tokens_per_request": 8192
        }
