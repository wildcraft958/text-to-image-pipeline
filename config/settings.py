import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Google Cloud Configuration
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "macro-dreamer-459709-p4")
    GOOGLE_LOCATION: str = os.getenv("GOOGLE_LOCATION", "asia-southeast1")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Model Configuration
    LLM_MODEL: str = "gemma-3n-e4b-it"
    IMAGE_MODEL: str = "imagen-3.0-generate-002"
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    SIMILARITY_THRESHOLD: float = 0.85
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings()
