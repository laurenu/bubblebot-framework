from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    embedding_batch_size: int = 100
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    
    database_url: Optional[str] = None
    
    api_v1_str: str = "/api/v1"
    project_name: str = "Bubblebot Framework"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
