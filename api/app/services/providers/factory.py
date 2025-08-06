"""
Factory for creating embedding providers.
"""

from typing import Dict, Type
from .base import EmbeddingProvider, ProviderConfig, ProviderType, ProviderConfigError


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    _providers: Dict[ProviderType, Type[EmbeddingProvider]] = {}
    
    @classmethod
    def register_provider(cls, provider_type: ProviderType, provider_class: Type[EmbeddingProvider]):
        """Register a new provider class."""
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create_provider(cls, provider_type: ProviderType, config: ProviderConfig) -> EmbeddingProvider:
        """
        Create an embedding provider instance.
        
        Args:
            provider_type: Type of provider to create
            config: Configuration for the provider
            
        Returns:
            Configured provider instance
            
        Raises:
            ProviderConfigError: If provider type is not supported
        """
        if provider_type not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ProviderConfigError(
                f"Unsupported provider type: {provider_type}. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list[ProviderType]:
        """Get list of available provider types."""
        return list(cls._providers.keys())


def create_provider_from_settings(provider_name: str, **kwargs) -> EmbeddingProvider:
    """
    Convenience function to create a provider from settings.
    
    Args:
        provider_name: Name of the provider (e.g., "openai", "gemini")
        **kwargs: Configuration parameters
        
    Returns:
        Configured provider instance
    """
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        available = [p.value for p in ProviderType]
        raise ProviderConfigError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )
    
    config = ProviderConfig(**kwargs)
    return EmbeddingProviderFactory.create_provider(provider_type, config)
