"""
Base classes and interfaces for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    """Supported embedding providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class EmbeddingResponse:
    """Response from an embedding provider."""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any]
    processing_time_seconds: float


@dataclass
class ProviderConfig:
    """Configuration for embedding providers."""
    api_key: str
    model: str
    dimensions: Optional[int] = None
    max_batch_size: int = 100
    rate_limit_delay: float = 0.1
    timeout: int = 30


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Provider-specific parameters
            
        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        pass
    
    @abstractmethod
    async def embed_query(self, query: str, **kwargs) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string to embed
            **kwargs: Provider-specific parameters
            
        Returns:
            List of floats representing the embedding
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    @abstractmethod
    def calculate_cost(self, token_count: int) -> float:
        """
        Calculate the cost for embedding the given number of tokens.
        
        Args:
            token_count: Number of tokens to embed
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        """Return the maximum input tokens supported."""
        pass


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider errors."""
    pass


class ProviderConfigError(EmbeddingProviderError):
    """Raised when provider configuration is invalid."""
    pass


class ProviderAPIError(EmbeddingProviderError):
    """Raised when provider API returns an error."""
    pass


class ProviderRateLimitError(EmbeddingProviderError):
    """Raised when provider rate limit is exceeded."""
    pass
