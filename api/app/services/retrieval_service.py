"""
Context retrieval service for finding relevant document chunks.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.services.embedding_service import EmbeddingService, SearchResult
from app.services.document_processor import DocumentChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of context retrieval operation."""
    success: bool
    query: str
    relevant_chunks: List[SearchResult]
    context_text: str
    total_chunks_searched: int
    retrieval_time_seconds: float
    error_message: Optional[str] = None


class RetrievalService:
    """
    Service for retrieving relevant context from document chunks.
    
    Features:
    - Semantic similarity search
    - Context ranking and filtering
    - Configurable retrieval parameters
    - Context window management
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """Initialize the retrieval service with dependency injection for EmbeddingService."""
        self.embedding_service = embedding_service or EmbeddingService()
        self.max_context_length = settings.max_context_length
    
    async def retrieve_context(
        self,
        query: str,
        available_chunks: List[DocumentChunk],
        chunk_embeddings: Dict[str, List[float]],  # chunk_id -> embedding
        tenant_id: str,
        top_k: int = 5,
        similarity_threshold: float = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query or question
            available_chunks: List of available document chunks
            chunk_embeddings: Mapping of chunk IDs to embeddings
            tenant_id: Tenant identifier
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            RetrievalResult with relevant chunks and context
        """
        import time
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            if not query_embedding:
                return RetrievalResult(
                    success=False,
                    query=query,
                    relevant_chunks=[],
                    context_text="",
                    total_chunks_searched=0,
                    retrieval_time_seconds=time.time() - start_time,
                    error_message="Failed to generate query embedding"
                )
            
            # Prepare chunk-embedding pairs
            chunk_embedding_pairs = []
            for chunk in available_chunks:
                chunk_key = f"{chunk.source_file}_{chunk.chunk_index}"
                if chunk_key in chunk_embeddings:
                    chunk_embedding_pairs.append((chunk, chunk_embeddings[chunk_key]))
            
            # Find similar chunks
            similar_chunks = self.embedding_service.find_similar_chunks(
                query_embedding=query_embedding,
                chunk_embeddings=chunk_embedding_pairs,
                top_k=top_k,
                threshold=similarity_threshold
            )
            
            # Build context text
            context_text = self._build_context_text(similar_chunks)
            
            retrieval_time = time.time() - start_time
            
            logger.info(
                f"Retrieved {len(similar_chunks)} relevant chunks for tenant {tenant_id}: "
                f"query='{query[:50]}...' in {retrieval_time:.2f}s"
            )
            
            return RetrievalResult(
                success=True,
                query=query,
                relevant_chunks=similar_chunks,
                context_text=context_text,
                total_chunks_searched=len(chunk_embedding_pairs),
                retrieval_time_seconds=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return RetrievalResult(
                success=False,
                query=query,
                relevant_chunks=[],
                context_text="",
                total_chunks_searched=len(available_chunks),
                retrieval_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _build_context_text(self, search_results: List[SearchResult]) -> str:
        """
        Build context text from search results.
        
        Args:
            search_results: List of search results with chunks
            
        Returns:
            Formatted context text for chat completion
        """
        if not search_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in search_results:
            chunk = result.chunk
            
            # Format chunk with metadata
            chunk_text = f"[Source: {chunk.source_file}, Similarity: {result.similarity_score:.3f}]\n{chunk.content}"
            
            # Check if adding this chunk would exceed context limit
            if current_length + len(chunk_text) > self.max_context_length:
                break
                
            context_parts.append(chunk_text)
            current_length += len(chunk_text) + 2  # +2 for double newline
        
        return "\n\n".join(context_parts)
    
    def rank_chunks_by_relevance(
        self,
        chunks: List[SearchResult],
        boost_recent: bool = True,
        boost_longer: bool = False
    ) -> List[SearchResult]:
        """
        Re-rank chunks based on additional criteria.
        
        Args:
            chunks: List of search results to re-rank
            boost_recent: Give slight boost to more recent documents
            boost_longer: Give slight boost to longer chunks
            
        Returns:
            Re-ranked list of search results
        """
        if not chunks:
            return []
        
        # Create scoring function
        def calculate_score(result: SearchResult) -> float:
            base_score = result.similarity_score
            
            # Boost factors (small adjustments to maintain similarity dominance)
            if boost_longer and result.chunk.word_count > 100:
                base_score += 0.04  # Small boost for longer chunks
            
            # Note: boost_recent would require document timestamp in metadata
            # For now, just return base score
            
            return base_score
        
        # Re-rank and update rank numbers
        ranked_chunks = sorted(chunks, key=calculate_score, reverse=True)
        
        for i, chunk in enumerate(ranked_chunks):
            chunk.rank = i + 1
        
        return ranked_chunks
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate retrieval statistics."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.success]
        
        return {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": len(results) - len(successful_results),
            "average_retrieval_time": sum(r.retrieval_time_seconds for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_chunks_retrieved": sum(len(r.relevant_chunks) for r in successful_results) / len(successful_results) if successful_results else 0,
            "success_rate": len(successful_results) / len(results) * 100
        }
