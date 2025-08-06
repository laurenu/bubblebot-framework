from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):

    # --- Embedding Provider Settings ---
    # Defines which embedding provider to use ("openai", "gemini", etc.)
    embedding_provider: str = "openai"
    embedding_provider_api_key: str = "test-key"
    embedding_provider_model: str = "text-embedding-3-small"
    embedding_provider_dimensions: Optional[int] = 1536
    embedding_batch_size: int = 100

    # --- Legacy OpenAI Settings (for backward compatibility) ---
    openai_api_key: Optional[str] = None
    openai_embedding_model: Optional[str] = None
    openai_embedding_dimensions: Optional[int] = None

    # --- Retrieval Settings ---
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    # --- Database Settings ---
    database_url: Optional[str] = None
    
    # --- API Settings ---
    api_v1_str: str = "/api/v1"
    project_name: str = "Bubblebot Framework"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
