import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Google Cloud Configuration
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "macro-dreamer-459709-p4")
    GOOGLE_LOCATION: str = os.getenv("GOOGLE_LOCATION", "asia-southeast1")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    GOOGLE_AI_STUDIO_API_KEY: str = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")

    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Model Configuration
    LLM_MODEL: str = "gemma-2-27b-it"
    IMAGE_MODEL: str = "imagen-3.0-generate-001"

    # Cache Configuration
    CACHE_TTL: int = 3600
    SIMILARITY_THRESHOLD: float = 0.85
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    model_config = ConfigDict(
        env_file='.env',
        case_sensitive=True,
        extra='ignore' 
    )

settings = Settings()
