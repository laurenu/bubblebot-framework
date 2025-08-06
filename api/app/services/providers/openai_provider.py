"""
OpenAI Embedding Provider implementation.
"""

import asyncio
import logging
import time
import tiktoken
from typing import List, Dict, Any

from openai import AsyncOpenAI, RateLimitError, APIError

from .base import (
    EmbeddingProvider, 
    EmbeddingResponse, 
    ProviderConfig, 
    ProviderType,
    ProviderConfigError,
    ProviderAPIError,
    ProviderRateLimitError
)
from .factory import EmbeddingProviderFactory

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for OpenAI models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=self.config.api_key, timeout=self.config.timeout)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI tokenizer: {e}")
            self.tokenizer = None

    def _validate_config(self) -> None:
        if not self.config.api_key:
            raise ProviderConfigError("OpenAI API key is required.")
        if not self.config.model:
            raise ProviderConfigError("OpenAI model name is required.")

    async def embed_texts(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        start_time = time.time()
        total_tokens = self._count_tokens(texts)
        all_embeddings = []

        for i in range(0, len(texts), self.config.max_batch_size):
            batch = texts[i:i + self.config.max_batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                    dimensions=self.config.dimensions,
                    **kwargs
                )
                all_embeddings.extend([item.embedding for item in response.data])
                if i + self.config.max_batch_size < len(texts):
                    await asyncio.sleep(self.config.rate_limit_delay)
            except RateLimitError as e:
                logger.error(f"OpenAI rate limit exceeded: {e}")
                raise ProviderRateLimitError(f"OpenAI rate limit exceeded: {e}") from e
            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise ProviderAPIError(f"OpenAI API error: {e}") from e

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=self.config.model,
            usage={'total_tokens': total_tokens},
            processing_time_seconds=time.time() - start_time
        )

    async def embed_query(self, query: str, **kwargs) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.config.model,
                input=[query],
                dimensions=self.config.dimensions,
                **kwargs
            )
            return response.data[0].embedding
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded on query: {e}")
            raise ProviderRateLimitError(f"OpenAI rate limit exceeded on query: {e}") from e
        except APIError as e:
            logger.error(f"OpenAI API error on query: {e}")
            raise ProviderAPIError(f"OpenAI API error on query: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "max_batch_size": self.config.max_batch_size,
            "max_input_tokens": self.max_input_tokens
        }

    def calculate_cost(self, token_count: int) -> float:
        # Pricing for text-embedding-3-small: $0.02 / 1M tokens
        # This can be made more sophisticated to handle different models
        cost_per_token = 0.02 / 1_000_000
        return token_count * cost_per_token

    def _count_tokens(self, texts: List[str]) -> int:
        if not self.tokenizer:
            return sum(len(text) for text in texts) // 4  # Rough fallback
        return sum(len(self.tokenizer.encode(text)) for text in texts)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def max_input_tokens(self) -> int:
        # Varies by model, 8192 is common for embedding models
        return 8192

# Register the provider with the factory
EmbeddingProviderFactory.register_provider(ProviderType.OPENAI, OpenAIEmbeddingProvider)
